"""
calendar_sync.py — Google Calendar integration for BigBrother.

Handles OAuth 2.0 authentication, event fetching, and task completion
for the Google Calendar API. Provides current/next event data to the
AI coach for contextual roasts.
"""

import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Google API scopes — calendar read/write + tasks read
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks.readonly",
]


class CalendarSync:
    """
    Synchronises with Google Calendar via the official API.

    Caches events locally and refreshes at a configurable interval.
    """

    def __init__(
        self,
        credentials_path: str = "config/credentials.json",
        token_path: str = "config/token.json",
        calendar_id: str = "all",
        refresh_interval: int = 300,
    ):
        """calendar_id:
            "all"      — query every calendar visible to the account (default)
            "primary"  — only the user's default calendar
            "<id>"     — a specific calendar id (e.g. an @group.calendar.google.com id)
        """
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.calendar_id = calendar_id
        self.refresh_interval = refresh_interval

        self._service = None
        self._tasks_service = None
        self._events_cache: List[Dict[str, Any]] = []      # today
        self._tomorrow_cache: List[Dict[str, Any]] = []    # tomorrow
        self._tasks_cache: List[Dict[str, Any]] = []       # Google Tasks
        # Cache the calendar-id list resolved from `calendar_id` setting so
        # we don't re-list every refresh. Refreshed when _maybe_refresh runs.
        self._calendar_ids: List[str] = []
        self._last_refresh: float = 0
        self._authenticated = False
        # googleapiclient's httplib2 transport is NOT thread-safe — concurrent
        # API calls on the same _service raise SSL WRONG_VERSION_NUMBER /
        # DECRYPTION_FAILED. This lock serializes every request that touches
        # _service or _tasks_service.
        self._api_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """
        Run the OAuth 2.0 flow. Opens a browser window on first run
        to authorise the app. Stores the token in config/token.json.
        """
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError:
            logger.error(
                "Google API libraries not installed. Run: "
                "pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"
            )
            return False

        creds = None

        # Try to load existing token
        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
            except Exception as exc:
                logger.warning("Could not load token: %s", exc)

        # Refresh or re-authenticate
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("Calendar token refreshed")
            except Exception:
                creds = None

        # Force re-auth if stored token is missing required scopes (e.g. tasks.readonly added later)
        if creds and creds.valid and creds.scopes:
            missing = set(SCOPES) - set(creds.scopes)
            if missing:
                logger.info(
                    "Stored token missing scopes %s — re-authenticating to include them", missing
                )
                creds = None

        if not creds or not creds.valid:
            if not self.credentials_path.exists():
                logger.error(
                    "credentials.json not found at %s. "
                    "Download it from Google Cloud Console.",
                    self.credentials_path,
                )
                return False

            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_path), SCOPES
            )
            creds = flow.run_local_server(port=0)
            logger.info("Calendar authentication completed")

        # Save token
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.token_path, "w") as f:
            f.write(creds.to_json())

        self._service = build("calendar", "v3", credentials=creds)
        try:
            self._tasks_service = build("tasks", "v1", credentials=creds)
            logger.info("Google Tasks service ready")
        except Exception as exc:
            logger.warning("Could not build Tasks service: %s", exc)
            self._tasks_service = None
        self._authenticated = True
        logger.info("Google Calendar service ready")
        return True

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    # ------------------------------------------------------------------
    # Calendar selection
    # ------------------------------------------------------------------

    def _resolve_calendar_ids(self) -> List[str]:
        """Resolve the configured calendar_id setting into actual ids.

        "all"     → every calendar visible to the account.
        "primary" → just primary.
        otherwise → treat the string as a literal calendar id.

        Result is cached on the instance — most accounts only have a handful
        of calendars and the list barely changes, so re-listing every poll
        is wasteful. _maybe_refresh clears it when the cache rolls over."""
        if self._calendar_ids:
            return self._calendar_ids
        if not self._authenticated or self._service is None:
            return ["primary"]
        target = (self.calendar_id or "all").strip()
        if target != "all":
            self._calendar_ids = [target]
            return self._calendar_ids
        try:
            with self._api_lock:
                result = self._service.calendarList().list().execute()
            ids = []
            for c in result.get("items", []):
                cid = c.get("id")
                if not cid:
                    continue
                # Skip purely informational read-only calendars (Holidays etc.).
                # Heuristic: if the user is just a reader AND the id looks
                # like a system feed, skip. We keep owned and shared-with-edit.
                if c.get("accessRole") == "reader" and (
                    cid.endswith("@group.v.calendar.google.com") or
                    cid.startswith("en.")
                ):
                    continue
                ids.append(cid)
            if not ids:
                ids = ["primary"]
            self._calendar_ids = ids
            logger.info("Calendar fetch will query %d calendars: %s",
                        len(ids), ", ".join(ids[:3]) + ("…" if len(ids) > 3 else ""))
            return ids
        except Exception as exc:
            logger.warning("calendarList().list() failed (%s) — falling back to primary", exc)
            self._calendar_ids = ["primary"]
            return self._calendar_ids

    def _fetch_day(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Pull events between start..end from every configured calendar and
        merge into one timestamp-sorted list. Used by all the day-bound
        fetch methods so they share a single multi-calendar pathway."""
        merged: List[Dict[str, Any]] = []
        for cid in self._resolve_calendar_ids():
            try:
                with self._api_lock:
                    result = self._service.events().list(
                        calendarId=cid,
                        timeMin=start.isoformat(),
                        timeMax=end.isoformat(),
                        singleEvents=True,
                        orderBy="startTime",
                    ).execute()
                for item in result.get("items", []):
                    merged.append({
                        "id": item.get("id"),
                        "calendar_id": cid,
                        "summary": item.get("summary", "Untitled"),
                        "start": item.get("start", {}).get("dateTime",
                                  item.get("start", {}).get("date", "")),
                        "end":   item.get("end",   {}).get("dateTime",
                                  item.get("end",   {}).get("date", "")),
                        "status": item.get("status", "confirmed"),
                        "description": item.get("description", ""),
                    })
            except Exception as exc:
                logger.warning("Calendar %s fetch failed: %s", cid, exc)
        # Sort by start so the merged view reads chronologically. Items
        # with no start come last.
        merged.sort(key=lambda e: e.get("start") or "9999")
        return merged


    # ------------------------------------------------------------------
    # Event fetching
    # ------------------------------------------------------------------

    def fetch_events(self, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch events for a given day (defaults to today, in LOCAL time).

        We deliberately bound the day by the user's local clock, not UTC.
        Doing it in UTC means a 23:00 local meeting today shows up as
        'tomorrow' and a 02:00 local meeting tomorrow shows up as 'today'
        — the kind of off-by-one that makes the dashboard look wrong.
        Google Calendar accepts a tz-aware ISO string and bounds the query
        correctly regardless of timezone offset.
        """
        if not self._authenticated or self._service is None:
            logger.warning("Calendar not authenticated — returning empty events")
            return []

        if date is None:
            date = datetime.now().astimezone()  # tz-aware in LOCAL tz

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        try:
            events = self._fetch_day(start_of_day, end_of_day)
            self._events_cache = events
            self._last_refresh = time.time()
            logger.info("Fetched %d calendar events for today across %d calendar(s)",
                        len(events), len(self._calendar_ids))
            return events
        except Exception as exc:
            logger.error("Failed to fetch calendar events: %s", exc)
            return self._events_cache  # return stale cache on error

    def fetch_tomorrow_events(self) -> List[Dict[str, Any]]:
        """Fetch tomorrow's events across every configured calendar."""
        if not self._authenticated or self._service is None:
            return []
        tomorrow = datetime.now().astimezone() + timedelta(days=1)
        start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
        try:
            events = self._fetch_day(start, end)
            for e in events:
                e["day"] = "tomorrow"
            self._tomorrow_cache = events
            logger.info("Fetched %d tomorrow events", len(events))
            return self._tomorrow_cache
        except Exception as exc:
            logger.error("Failed to fetch tomorrow events: %s", exc)
            return self._tomorrow_cache

    def fetch_tasks(self) -> List[Dict[str, Any]]:
        """
        Fetch incomplete Google Tasks due by end of the current week (Sunday).
        If fewer than 3 days remain in the week, extends to the following Sunday.
        Returns a list of task dicts with keys: id, title, due, status, tasklist.
        """
        if not self._authenticated or self._tasks_service is None:
            return []
        try:
            # Use LOCAL time so weekday/Sunday-end matches the user's clock.
            now = datetime.now().astimezone()

            # Advance to the end of this Sunday; if <3 days remain, go to next Sunday
            days_until_sunday = (6 - now.weekday()) % 7  # Monday=0 … Sunday=6
            if days_until_sunday < 3:
                days_until_sunday += 7  # extend to next Sunday
            week_end = (now + timedelta(days=days_until_sunday + 1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            with self._api_lock:
                task_lists = self._tasks_service.tasklists().list(maxResults=10).execute()
            tasks: List[Dict[str, Any]] = []
            for tl in task_lists.get("items", []):
                tl_id = tl["id"]
                with self._api_lock:
                    result = self._tasks_service.tasks().list(
                        tasklist=tl_id,
                        showCompleted=False,
                        dueMax=week_end.isoformat(),
                        maxResults=30,
                    ).execute()
                for item in result.get("items", []):
                    if item.get("status") == "completed":
                        continue
                    tasks.append({
                        "id": item.get("id"),
                        "title": item.get("title", "Untitled task"),
                        "due": item.get("due", ""),
                        "notes": item.get("notes", ""),
                        "status": item.get("status", "needsAction"),
                        "tasklist": tl.get("title", ""),
                    })
            # Sort by due date (tasks with no due date go last)
            tasks.sort(key=lambda t: t["due"] or "9999")
            self._tasks_cache = tasks
            logger.info("Fetched %d pending tasks for the week", len(tasks))
            return tasks
        except Exception as exc:
            logger.error("Failed to fetch tasks: %s", exc)
            return self._tasks_cache

    def _maybe_refresh(self):
        """Refresh events (today + tomorrow) and tasks if the cache is stale.

        Hot path (cache fresh) avoids the lock entirely so dashboard polls
        that just read cached data don't queue behind one another. Only
        when a refresh is actually needed do we take the lock and re-check
        under it to avoid two callers both refreshing."""
        if time.time() - self._last_refresh <= self.refresh_interval:
            return
        with self._api_lock:
            # Re-check under the lock: another thread may have refreshed
            # while we were waiting.
            if time.time() - self._last_refresh > self.refresh_interval:
                # Drop the cached calendar id list so a newly-added calendar
                # gets discovered on the next cycle (~5 min by default).
                self._calendar_ids = []
                self.fetch_events()          # populates _events_cache (today)
                self.fetch_tomorrow_events() # populates _tomorrow_cache
                self.fetch_tasks()           # populates _tasks_cache

    def get_today_events(self) -> List[Dict[str, Any]]:
        """Get today's events (cached, auto-refreshing)."""
        self._maybe_refresh()
        return self._events_cache

    def get_tomorrow_events(self) -> List[Dict[str, Any]]:
        """Get tomorrow's events (cached, auto-refreshing)."""
        self._maybe_refresh()
        return self._tomorrow_cache

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get pending Google Tasks due today or tomorrow (cached, auto-refreshing)."""
        self._maybe_refresh()
        return self._tasks_cache

    def get_week_plan(self) -> List[Dict[str, Any]]:
        """
        Return a day-by-day plan for the rest of the week (today → Sunday).
        Each entry: { date, label, events, tasks }
        """
        self._maybe_refresh()

        # Week-plan boundaries are LOCAL — see fetch_events for the rationale.
        now = datetime.now().astimezone()
        local_tz = now.tzinfo
        today = now.date()

        # How many days until Sunday (0 = today is Sunday → show current week only)
        days_until_sunday = (6 - today.weekday()) % 7
        if days_until_sunday == 0:
            days_until_sunday = 6  # Sunday → show the next 6 days

        days: List[Dict[str, Any]] = []

        for offset in range(days_until_sunday + 1):
            target_date = today + timedelta(days=offset)
            date_str = target_date.isoformat()  # "2026-02-24"

            # Label
            if offset == 0:
                label = "Today"
            elif offset == 1:
                label = "Tomorrow"
            else:
                label = target_date.strftime("%A")  # Monday, Tuesday…

            # Events for this day
            if offset == 0:
                day_events = [
                    {"summary": e.get("summary", ""), "start": e.get("start", ""),
                     "end": e.get("end", ""), "id": e.get("id", "")}
                    for e in self._events_cache
                ]
            elif offset == 1:
                day_events = [
                    {"summary": e.get("summary", ""), "start": e.get("start", ""),
                     "end": e.get("end", ""), "id": e.get("id", "")}
                    for e in self._tomorrow_cache
                ]
            else:
                # Fetch on demand across all calendars (week view doesn't
                # cache further-out days — too many to keep warm).
                day_events = []
                if self._authenticated and self._service is not None:
                    try:
                        target_dt = datetime(target_date.year, target_date.month,
                                             target_date.day, tzinfo=local_tz)
                        events = self._fetch_day(target_dt, target_dt + timedelta(days=1))
                        day_events = [
                            {"summary": e.get("summary", ""),
                             "start": e.get("start", ""),
                             "end":   e.get("end", ""),
                             "id":    e.get("id", "")}
                            for e in events
                        ]
                    except Exception as exc:
                        logger.warning("Could not fetch events for %s: %s", date_str, exc)

            # Tasks due on this day
            day_tasks = [
                {"title": t.get("title", ""), "id": t.get("id", ""),
                 "due": t.get("due", ""), "tasklist": t.get("tasklist", "")}
                for t in self._tasks_cache
                if t.get("due", "")[:10] == date_str
            ]
            # Also include tasks with no due date on the "Today" bucket
            if offset == 0:
                day_tasks += [
                    {"title": t.get("title", ""), "id": t.get("id", ""),
                     "due": "", "tasklist": t.get("tasklist", "")}
                    for t in self._tasks_cache
                    if not t.get("due")
                ]

            days.append({
                "date":   date_str,
                "label":  label,
                "events": day_events,
                "tasks":  day_tasks,
            })

        return days

    @staticmethod
    def _parse_event_dt(value: str) -> Optional[datetime]:
        """Parse a Google Calendar start/end value.

        Two shapes from the API:
          - "2026-05-14T15:30:00+03:00"  (tz-aware)
          - "2026-05-14"                  (all-day events, naive date string)

        We return a tz-AWARE datetime in both cases: the all-day form is
        anchored at local midnight. Mixing aware and naive datetimes in
        comparisons raises TypeError, so we have to canonicalise here."""
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
        if dt.tzinfo is None:
            # All-day event — Google returns just "YYYY-MM-DD". Treat it as
            # local-day-start so it compares cleanly against an aware now().
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return dt

    def get_current_event(self) -> Optional[Dict[str, Any]]:
        """Get the event happening right now, if any."""
        self._maybe_refresh()
        now = datetime.now(timezone.utc)

        for event in self._events_cache:
            try:
                start = self._parse_event_dt(event.get("start", ""))
                end   = self._parse_event_dt(event.get("end", ""))
                if start is None or end is None:
                    continue
                if start <= now <= end:
                    return event
            except Exception:
                continue
        return None

    def get_next_event(self) -> Optional[Dict[str, Any]]:
        """Get the next upcoming event, searching today then tomorrow."""
        self._maybe_refresh()
        now = datetime.now(timezone.utc)

        for event in self._events_cache + self._tomorrow_cache:
            try:
                start = self._parse_event_dt(event.get("start", ""))
                if start is not None and start > now:
                    return event
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Task completion
    # ------------------------------------------------------------------

    def complete_task(self, event_id: str) -> bool:
        """
        Mark a calendar event as completed by prefixing "[done]" to its title.
        With multi-calendar support we have to know WHICH calendar the event
        lives on — look it up from the cached payloads where we recorded it.
        """
        if not self._authenticated or self._service is None:
            return False

        # Find the calendar this event came from. Cached entries record
        # calendar_id; if the event is from get_week_plan (which doesn't
        # cache far-out days) we won't find it and have to bail.
        owning_cid: Optional[str] = None
        for cached in self._events_cache + self._tomorrow_cache:
            if cached.get("id") == event_id:
                owning_cid = cached.get("calendar_id")
                break
        if not owning_cid:
            # Last resort: try primary, but warn so it's discoverable in the log.
            owning_cid = "primary"
            logger.warning("complete_task(%s): no cached calendar id, falling back to primary", event_id)

        try:
            with self._api_lock:
                event = self._service.events().get(
                    calendarId=owning_cid, eventId=event_id
                ).execute()

                event["colorId"] = "11"  # Green in Google Calendar
                summary = event.get("summary", "")
                if not summary.startswith("[done]"):
                    event["summary"] = "[done] " + summary

                self._service.events().update(
                    calendarId=owning_cid, eventId=event_id, body=event
                ).execute()

            logger.info("Marked event %s on calendar %s as complete",
                        event_id, owning_cid)

            # Refresh cache
            self.fetch_events()
            return True

        except Exception as exc:
            logger.error("Failed to complete task %s: %s", event_id, exc)
            return False
