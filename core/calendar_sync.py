"""
calendar_sync.py — Google Calendar integration for BigBrother.

Handles OAuth 2.0 authentication, event fetching, and task completion
for the Google Calendar API. Provides current/next event data to the
AI coach for contextual roasts.
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Google API scopes — read/write for task completion
SCOPES = ["https://www.googleapis.com/auth/calendar"]


class CalendarSync:
    """
    Synchronises with Google Calendar via the official API.

    Caches events locally and refreshes at a configurable interval.
    """

    def __init__(
        self,
        credentials_path: str = "config/credentials.json",
        token_path: str = "config/token.json",
        calendar_id: str = "primary",
        refresh_interval: int = 300,
    ):
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.calendar_id = calendar_id
        self.refresh_interval = refresh_interval

        self._service = None
        self._events_cache: List[Dict[str, Any]] = []
        self._last_refresh: float = 0
        self._authenticated = False

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
        self._authenticated = True
        logger.info("Google Calendar service ready")
        return True

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    # ------------------------------------------------------------------
    # Event fetching
    # ------------------------------------------------------------------

    def fetch_events(self, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch events for a given day (defaults to today).
        Returns a list of event dicts with keys: id, summary, start, end, status.
        """
        if not self._authenticated or self._service is None:
            logger.warning("Calendar not authenticated — returning empty events")
            return []

        if date is None:
            date = datetime.now(timezone.utc)

        # Start / end of the day in UTC
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        try:
            result = self._service.events().list(
                calendarId=self.calendar_id,
                timeMin=start_of_day.isoformat(),
                timeMax=end_of_day.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            ).execute()

            events = []
            for item in result.get("items", []):
                events.append({
                    "id": item.get("id"),
                    "summary": item.get("summary", "Untitled"),
                    "start": item.get("start", {}).get("dateTime", item.get("start", {}).get("date", "")),
                    "end": item.get("end", {}).get("dateTime", item.get("end", {}).get("date", "")),
                    "status": item.get("status", "confirmed"),
                    "description": item.get("description", ""),
                })

            self._events_cache = events
            self._last_refresh = time.time()
            logger.info("Fetched %d calendar events", len(events))
            return events

        except Exception as exc:
            logger.error("Failed to fetch calendar events: %s", exc)
            return self._events_cache  # return stale cache on error

    def _maybe_refresh(self):
        """Refresh events if the cache is stale."""
        if time.time() - self._last_refresh > self.refresh_interval:
            self.fetch_events()

    def get_today_events(self) -> List[Dict[str, Any]]:
        """Get today's events (cached, auto-refreshing)."""
        self._maybe_refresh()
        return self._events_cache

    def get_current_event(self) -> Optional[Dict[str, Any]]:
        """Get the event happening right now, if any."""
        self._maybe_refresh()
        now = datetime.now(timezone.utc)

        for event in self._events_cache:
            try:
                start = datetime.fromisoformat(event["start"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(event["end"].replace("Z", "+00:00"))
                if start <= now <= end:
                    return event
            except (ValueError, KeyError):
                continue
        return None

    def get_next_event(self) -> Optional[Dict[str, Any]]:
        """Get the next upcoming event."""
        self._maybe_refresh()
        now = datetime.now(timezone.utc)

        for event in self._events_cache:
            try:
                start = datetime.fromisoformat(event["start"].replace("Z", "+00:00"))
                if start > now:
                    return event
            except (ValueError, KeyError):
                continue
        return None

    # ------------------------------------------------------------------
    # Task completion
    # ------------------------------------------------------------------

    def complete_task(self, event_id: str) -> bool:
        """
        Mark a calendar event as completed by changing its status.
        Uses colorId to visually mark it as done (green = 11).
        """
        if not self._authenticated or self._service is None:
            return False

        try:
            event = self._service.events().get(
                calendarId=self.calendar_id, eventId=event_id
            ).execute()

            event["colorId"] = "11"  # Green in Google Calendar
            event["summary"] = "✅ " + event.get("summary", "")

            self._service.events().update(
                calendarId=self.calendar_id, eventId=event_id, body=event
            ).execute()

            logger.info("Marked event %s as complete", event_id)

            # Refresh cache
            self.fetch_events()
            return True

        except Exception as exc:
            logger.error("Failed to complete task %s: %s", event_id, exc)
            return False
