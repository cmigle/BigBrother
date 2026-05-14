"""
generate_audit.py — produce BigBrother_Audit_Report.xlsx from the live system.

Usage (from repo root):
    python scripts/generate_audit.py [--url http://127.0.0.1:5000] [--out file.xlsx]

The script probes the running server, reads data/*.json, and writes a fresh
7-sheet workbook. Server-off probes are recorded as 'OFFLINE' so the report
is still useful when the app isn't running.

Sheets, in order:
    Overview       — totals, pass rate, generation timestamp
    Endpoints      — HTTP probes (method, path, status, latency, notes)
    UI Inventory   — every interactive element + what backs it
    Live State     — current /api/status dump with field meanings
    Stats          — daily kills + hourly distribution from data/stats.json
    Roast History  — full roast log from data/roast_history.json
    Environment    — python/library versions + degraded-mode notes
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import urllib.request
import urllib.error


# ── Styling helpers ──────────────────────────────────────────────────────────
HEADER_FILL = PatternFill("solid", fgColor="1F2A3A")
HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
TITLE_FONT  = Font(bold=True, size=14, color="1F2A3A")
SUBTITLE_FONT = Font(italic=True, color="666666", size=9)
THIN = Side(style="thin", color="D0D0D0")
CELL_BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

STATUS_FILLS = {
    "PASS":     PatternFill("solid", fgColor="D4EDDA"),
    "OK":       PatternFill("solid", fgColor="D4EDDA"),
    "EXPECTED": PatternFill("solid", fgColor="FFF3CD"),
    "DEGRADED": PatternFill("solid", fgColor="FFF3CD"),
    "FAIL":     PatternFill("solid", fgColor="F8D7DA"),
    "OFFLINE":  PatternFill("solid", fgColor="E2E3E5"),
}


def style_header(ws, row: int, ncols: int):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="left", vertical="center")
        cell.border = CELL_BORDER


def style_status_cell(ws, row: int, col: int):
    cell = ws.cell(row=row, column=col)
    value = (cell.value or "").upper() if isinstance(cell.value, str) else ""
    if value in STATUS_FILLS:
        cell.fill = STATUS_FILLS[value]
        cell.font = Font(bold=True, color="1F2A3A")


def auto_size(ws, max_width: int = 60):
    """Best-effort column auto-sizing — openpyxl can't measure rendered width,
    so we approximate from the longest line of text in each column."""
    for col_cells in ws.columns:
        col_letter = get_column_letter(col_cells[0].column)
        longest = 0
        for cell in col_cells:
            if cell.value is None:
                continue
            # splitlines on '\n' alone returns []; max() over empty is an error
            for line in str(cell.value).splitlines() or [""]:
                longest = max(longest, len(line))
        ws.column_dimensions[col_letter].width = min(max_width, max(10, longest + 2))


# ── HTTP probing ─────────────────────────────────────────────────────────────
@dataclass
class ProbeResult:
    method: str
    path: str
    status: str          # PASS / EXPECTED / FAIL / OFFLINE
    http_status: Optional[int]
    latency_ms: Optional[float]
    content_type: str
    note: str


def probe(base_url: str, method: str, path: str, body: Optional[dict] = None,
          expect_status: int = 200, expect_failure_ok: bool = False) -> ProbeResult:
    """Hit one endpoint and classify the outcome.

    expect_failure_ok=True flags routes that legitimately 503/404 when a
    subsystem is down (e.g. calendar without auth). Those map to EXPECTED."""
    url = base_url.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            ctype = resp.headers.get("Content-Type", "")
            # Read just enough to confirm payload exists; don't blow memory on MJPEG.
            try:
                head = resp.read(1024)
            except Exception:
                head = b""
            note = ""
            if "json" in ctype:
                try:
                    note = json.dumps(json.loads(head.decode()))[:140]
                except Exception:
                    pass
            elif "multipart" in ctype:
                note = "multipart MJPEG stream"
            return ProbeResult(method, path, "PASS", resp.status,
                               round(elapsed_ms, 1), ctype, note)
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        status = "EXPECTED" if expect_failure_ok else "FAIL"
        return ProbeResult(method, path, status, e.code, round(elapsed_ms, 1),
                           e.headers.get_content_type() if e.headers else "",
                           f"HTTP {e.code}")
    except urllib.error.URLError as e:
        # Server not running, DNS fail, refused — all map to OFFLINE.
        return ProbeResult(method, path, "OFFLINE", None, None, "",
                           f"{type(e).__name__}: {e.reason}")
    except Exception as e:
        return ProbeResult(method, path, "FAIL", None, None, "",
                           f"{type(e).__name__}: {e}")


def fetch_json(base_url: str, path: str) -> Optional[dict]:
    """Best-effort JSON fetch used to populate Live State / Stats sheets."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=8) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


# ── Sheet builders ───────────────────────────────────────────────────────────
def build_overview(wb, probes: list[ProbeResult], base_url: str):
    ws = wb.create_sheet("Overview")
    ws["A1"] = "BigBrother — System Audit"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:D1")
    ws["A2"] = f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}   ·   Audit URL: {base_url}"
    ws["A2"].font = SUBTITLE_FONT
    ws.merge_cells("A2:D2")

    # KPI block
    counts = {"PASS": 0, "EXPECTED": 0, "FAIL": 0, "OFFLINE": 0}
    for p in probes:
        counts[p.status] = counts.get(p.status, 0) + 1
    total = sum(counts.values())
    pass_rate = (counts["PASS"] / total * 100) if total else 0.0

    ws["A4"] = "Result"
    ws["B4"] = "Count"
    style_header(ws, 4, 2)

    rows = [
        ("Total probes", total),
        ("PASS", counts["PASS"]),
        ("EXPECTED (graceful failure)", counts["EXPECTED"]),
        ("FAIL", counts["FAIL"]),
        ("OFFLINE (server unreachable)", counts["OFFLINE"]),
        ("Pass rate", f"{pass_rate:.1f}%"),
    ]
    for i, (k, v) in enumerate(rows, start=5):
        ws.cell(row=i, column=1, value=k)
        ws.cell(row=i, column=2, value=v)
        # Color-code the status rows
        if k in STATUS_FILLS:
            ws.cell(row=i, column=1).fill = STATUS_FILLS[k]

    # Subsystem health derived from probes/notes
    ws.cell(row=13, column=1, value="Subsystem health")
    ws.cell(row=13, column=1).font = Font(bold=True, size=11)
    ws.cell(row=14, column=1, value="Subsystem")
    ws.cell(row=14, column=2, value="State")
    ws.cell(row=14, column=3, value="Evidence")
    style_header(ws, 14, 3)

    health = _derive_health(probes, base_url)
    for i, (name, state, evidence) in enumerate(health, start=15):
        ws.cell(row=i, column=1, value=name)
        ws.cell(row=i, column=2, value=state)
        ws.cell(row=i, column=3, value=evidence)
        style_status_cell(ws, i, 2)

    auto_size(ws)


def _derive_health(probes: list[ProbeResult], base_url: str) -> list[tuple]:
    """Compact health summary for the Overview sheet."""
    status_probe = next((p for p in probes if p.path == "/api/status"), None)
    cal_probe    = next((p for p in probes if p.path == "/api/calendar"), None)
    feed_probe   = next((p for p in probes if p.path == "/api/camera/feed"), None)

    rows: list[tuple] = []

    if status_probe is None or status_probe.status == "OFFLINE":
        rows.append(("Web server", "OFFLINE", "No response on /api/status"))
        return rows
    rows.append(("Web server", "PASS", f"/api/status {status_probe.http_status} "
                                       f"in {status_probe.latency_ms} ms"))

    # Calendar — auth state is encoded in the response JSON
    cal_state = "OFFLINE"
    cal_evidence = ""
    cal_json = fetch_json(base_url, "/api/calendar")
    if cal_json is not None:
        if cal_json.get("authenticated"):
            cal_state = "PASS"
            cal_evidence = (f"authenticated — "
                            f"{len(cal_json.get('events', []))} today, "
                            f"{len(cal_json.get('tomorrow_events', []))} tomorrow, "
                            f"{len(cal_json.get('tasks', []))} tasks")
        else:
            cal_state = "EXPECTED"
            cal_evidence = "not authenticated (credentials.json missing or token expired)"
    rows.append(("Google Calendar", cal_state, cal_evidence))

    # Camera feed
    if feed_probe and feed_probe.status == "PASS":
        rows.append(("Camera (MJPEG feed)", "PASS",
                     f"{feed_probe.content_type} in {feed_probe.latency_ms} ms"))
    else:
        rows.append(("Camera (MJPEG feed)", "DEGRADED",
                     "Feed endpoint not responding 200"))

    # Ollama: derived from server log isn't easy, so we just note it's probably off
    # if the latest roast text doesn't look LLM-generated. Heuristic only.
    status_json = fetch_json(base_url, "/api/status")
    if status_json:
        roast = (status_json.get("latest_roast") or "").lower()
        if not roast:
            rows.append(("AI Coach (Ollama)", "EXPECTED", "no roast generated yet"))
        else:
            rows.append(("AI Coach", "PASS",
                         f"latest roast: {status_json['latest_roast'][:80]}…"))

    # Distraction monitor
    if status_json:
        if status_json.get("distraction_detected"):
            rows.append(("Process monitor", "PASS",
                         f"active — {status_json.get('distraction_label')} "
                         f"({status_json.get('distraction_source')})"))
        else:
            rows.append(("Process monitor", "PASS", "idle (no distraction)"))

    return rows


def build_endpoints(wb, probes: list[ProbeResult]):
    ws = wb.create_sheet("Endpoints")
    ws["A1"] = "HTTP Endpoint Probes"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:G1")

    headers = ["ID", "Method", "Path", "Status", "HTTP", "Latency (ms)", "Content-Type", "Notes"]
    for c, h in enumerate(headers, 1):
        ws.cell(row=3, column=c, value=h)
    style_header(ws, 3, len(headers))

    for i, p in enumerate(probes, start=4):
        ws.cell(row=i, column=1, value=f"E{i-3:02d}")
        ws.cell(row=i, column=2, value=p.method)
        ws.cell(row=i, column=3, value=p.path)
        ws.cell(row=i, column=4, value=p.status)
        ws.cell(row=i, column=5, value=p.http_status)
        ws.cell(row=i, column=6, value=p.latency_ms)
        ws.cell(row=i, column=7, value=p.content_type)
        ws.cell(row=i, column=8, value=p.note)
        style_status_cell(ws, i, 4)
    auto_size(ws)


def build_ui_inventory(wb, base_url: str):
    """UI elements + their backing endpoint. Static catalogue maintained here —
    if you add a UI element, add a row. Status comes from cross-referencing
    the live endpoint probes where applicable."""
    ws = wb.create_sheet("UI Inventory")
    ws["A1"] = "UI Inventory — every element + what backs it"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:F1")

    headers = ["Page", "Element", "Type", "What it does", "Backed by", "Notes"]
    for c, h in enumerate(headers, 1):
        ws.cell(row=3, column=c, value=h)
    style_header(ws, 3, len(headers))

    rows = [
        # Nav
        ("All pages",  "Brand text (BIGBROTHER)", "Label",   "Site title in the nav",                          "Static markup",                "No glitch effect after the polish pass."),
        ("All pages",  "Nav links",               "Links",   "Dashboard / Camera / Stats / Settings",          "GET /, /camera, /stats, /settings", ""),
        ("All pages",  "Live clock",              "Widget",  "HH:MM:SS, client-side, ticks every second",      "JS setInterval",               ""),

        # Dashboard — top
        ("Dashboard",  "Live-detection toggle",   "Toggle",  "Enable/disable camera + inline feed",            "POST /api/settings {camera_enabled}", "Closes MJPEG stream when off."),
        ("Dashboard",  "Inline mini-feed",        "Stream",  "Live MJPEG preview, 360×240",                    "GET /api/camera/feed",        "Cache-busted on each ON toggle."),
        ("Dashboard",  "Status banner",           "Indicator", "Dot + status text + confidence",               "GET /api/status",             "Three states: all-clear, on-table (warning), active (danger). Dot pulses on danger only."),
        ("Dashboard",  "Distraction card",        "Indicator", "Web/app distraction with duration",            "GET /api/status (distraction_*)", "WEB/APP text tag instead of icon."),
        ("Dashboard",  "Drill Sergeant roast",    "Widget",  "Latest roast + Speak button",                    "GET /api/status (latest_roast, latest_roast_id)", "Only speaks new roasts since page load."),
        ("Dashboard",  "Speak button",            "Button",  "Replay roast audio via edge-tts MP3",            "GET /api/roast/audio?text=...", "3-voice rotation: Brian / Ryan / Andrew."),
        ("Dashboard",  "Recommended focus card",  "Widget",  "Calendar-aware 'what to do now'",                "GET /api/status (latest_guess)", "Names current event or next upcoming."),

        # Dashboard — stats row
        ("Dashboard",  "Kills Today",             "Stat",    "Counter of triggered scares today",              "GET /api/status (kill_count_today)", ""),
        ("Dashboard",  "Clean Streak",            "Stat",    "Time since last kill",                           "GET /api/status (current_streak)", ""),
        ("Dashboard",  "Armed/Cooldown",          "Stat",    "Whether scare is on cooldown",                   "GET /api/status (on_cooldown)", ""),

        # Dashboard — calendar
        ("Dashboard",  "Today's Calendar",        "List",    "Google Calendar events for today",               "GET /api/calendar",           "Strike-through + [done] marker on completed events."),
        ("Dashboard",  "Tomorrow",                "List",    "Tomorrow's events, hidden if empty",             "GET /api/calendar",           ""),
        ("Dashboard",  "This Week's Tasks",       "List",    "Google Tasks due this week",                     "GET /api/calendar",           "Today/Tomorrow/Overdue badges."),
        ("Dashboard",  "This Week",               "Grid",    "Day-by-day plan",                                "GET /api/weekly-plan",        ""),
        ("Dashboard",  "Event checkbox",          "Button",  "Mark a Google Calendar event done",              "POST /api/calendar/complete", "Writes '[done] ' prefix to event title."),

        # Camera page
        ("Camera",     "Live MJPEG image",        "Stream",  "Full YOLO-annotated camera feed",                "GET /api/camera/feed",        ""),
        ("Camera",     "Detection overlay",       "Label",   "MONITORING vs PHONE DETECTED",                   "GET /api/status",             ""),
        ("Camera",     "Status / Confidence / FPS rows", "Labels", "Side panel summary",                       "GET /api/status",             ""),

        # Stats page
        ("Stats",      "Total Kills",             "Stat",    "Lifetime counter",                               "GET /api/stats",              ""),
        ("Stats",      "This Week",               "Stat",    "7-day total",                                    "GET /api/stats (weekly)",     ""),
        ("Stats",      "Daily Average",           "Stat",    "Weekly mean",                                    "GET /api/stats (weekly)",     ""),
        ("Stats",      "Longest Streak",          "Stat",    "Best ever phone-free streak",                    "GET /api/stats",              ""),
        ("Stats",      "Weekly bar chart",        "Chart",   "Per-weekday Chart.js bars",                      "GET /api/stats (weekly.days)", ""),
        ("Stats",      "Worst Hours bar chart",   "Chart",   "Per-hour Chart.js bars",                         "GET /api/stats (worst_hours)", ""),
        ("Stats",      "Greatest Hits feed",      "List",    "Reverse-chronological roast log",                "GET /api/stats (roast_history)", ""),

        # Settings
        ("Settings",   "Camera On toggle",        "Toggle",  "Master enable for the camera",                   "POST /api/settings {camera_enabled}", ""),
        ("Settings",   "Active Camera dropdown",  "Select",  "Pick a webcam index",                            "GET /api/cameras / POST {camera_index}", "Hot-swaps via switch_camera thread."),
        ("Settings",   "Suppress Monitors",       "Toggle",  "filter_screens flag",                            "POST /api/settings {filter_screens}", ""),
        ("Settings",   "Confidence slider",       "Slider",  "YOLO confidence threshold",                      "POST /api/settings {confidence_threshold}", ""),
        ("Settings",   "Min duration input",      "Input",   "Sec of phone visibility before scare",           "(client only)",               "Not persisted server-side. Read at startup from settings.yaml only."),
        ("Settings",   "Strict Mode toggle",      "Toggle",  "Fire on any visible phone, skip motion gate",    "POST /api/settings {strict_mode}", ""),
        ("Settings",   "Volume slider",           "Slider",  "Scare sound volume",                             "POST /api/settings {volume}", ""),
        ("Settings",   "Cooldown input",          "Input",   "Seconds between scares",                         "POST /api/settings {cooldown_seconds}", ""),
        ("Settings",   "Escalation toggle",       "Toggle",  "Alarm escalates if phone persists",              "POST /api/settings {escalation_enabled}", ""),
        ("Settings",   "Process Monitor toggle",  "Toggle",  "Watch processes & browser titles",               "POST /api/settings {pm_enabled}", ""),
        ("Settings",   "Detect Browser Tabs",     "Toggle",  "Match Insta/TikTok/etc. in window titles",       "POST /api/settings {pm_check_web}", ""),
        ("Settings",   "Detect Distracting Apps", "Toggle",  "Match Steam/Discord/games by process name",      "POST /api/settings {pm_check_apps}", "Ignores apps with no visible window (tray-only)."),
        ("Settings",   "Text-to-Speech toggle",   "Toggle",  "Server-side AICoach.speak",                      "(form-only)",                 "No POST hook — change settings.yaml + restart."),
        ("Settings",   "Ollama Model input",      "Input",   "Local LLM model name",                           "(form-only)",                 "No POST hook — change settings.yaml + restart."),
        ("Settings",   "Serial Gun toggle",       "Toggle",  "Enable USB-serial trigger",                      "(form-only)",                 "No POST hook — change settings.yaml + restart."),
        ("Settings",   "Serial Port input",       "Input",   "COM port name",                                  "(form-only)",                 "No POST hook — change settings.yaml + restart."),
        ("Settings",   "Save Settings button",    "Button",  "Submit form",                                    "POST /api/settings",          ""),
    ]
    for i, row in enumerate(rows, start=4):
        for c, value in enumerate(row, start=1):
            ws.cell(row=i, column=c, value=value)
    auto_size(ws)


def build_live_state(wb, base_url: str):
    ws = wb.create_sheet("Live State")
    ws["A1"] = "Live /api/status snapshot"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:C1")

    headers = ["Field", "Value", "Meaning"]
    for c, h in enumerate(headers, 1):
        ws.cell(row=3, column=c, value=h)
    style_header(ws, 3, len(headers))

    meanings = {
        "phone_detected":      "True when YOLO sees a phone (real detection OR within the loss-tolerance grace window).",
        "actively_on_phone":   "True when the phone is moving enough to count as held (or when Strict Mode is on).",
        "confidence":          "YOLO confidence of the latest phone bbox.",
        "face_visible":        "Legacy field — inverse of actively_on_phone.",
        "current_streak":      "Seconds since last kill (clean streak).",
        "on_cooldown":         "Scare system is in its cooldown window.",
        "kill_count_today":    "Number of scares triggered today.",
        "latest_roast":        "Most recent AI Coach roast text.",
        "latest_roast_id":     "Monotonic id — dashboard uses this to detect a new roast since page load.",
        "latest_guess":        "Calendar-aware 'what to do now' string.",
        "distraction_detected": "True when a distracting browser tab / app is open with a visible window.",
        "distraction_label":   "Friendly name (e.g. Instagram, Discord).",
        "distraction_source":  "'web' (browser tab) or 'app' (process).",
        "distraction_duration": "Seconds the current distraction has persisted.",
        "camera_enabled":      "Master on/off for the detector.",
    }
    status = fetch_json(base_url, "/api/status") or {}
    row = 4
    for k in sorted(status.keys()):
        ws.cell(row=row, column=1, value=k)
        v = status[k]
        # Render dicts/lists as JSON; everything else as-is
        ws.cell(row=row, column=2, value=v if not isinstance(v, (dict, list)) else json.dumps(v))
        ws.cell(row=row, column=3, value=meanings.get(k, ""))
        row += 1
    if not status:
        ws.cell(row=4, column=1, value="(server unreachable)")
    auto_size(ws, max_width=80)


def build_stats(wb):
    ws = wb.create_sheet("Stats")
    ws["A1"] = "Stats — from data/stats.json"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:C1")

    stats_path = Path("data/stats.json")
    if not stats_path.exists():
        ws["A3"] = "data/stats.json not found"
        return
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception as e:
        ws["A3"] = f"Could not parse stats.json: {e}"
        return

    # Block 1: daily totals (newest first)
    ws["A3"] = "Daily kills"
    ws["A3"].font = Font(bold=True, size=11)
    headers = ["Date", "Kills", "Total duration (s)"]
    for c, h in enumerate(headers, 1):
        ws.cell(row=4, column=c, value=h)
    style_header(ws, 4, len(headers))

    daily = data.get("daily", {})
    if isinstance(daily, dict):
        items = sorted(daily.items(), reverse=True)
    else:
        items = []
    row = 5
    for date, info in items:
        kills = info.get("kill_count", 0) if isinstance(info, dict) else info
        dur   = info.get("total_duration", 0) if isinstance(info, dict) else 0
        ws.cell(row=row, column=1, value=date)
        ws.cell(row=row, column=2, value=kills)
        ws.cell(row=row, column=3, value=round(dur, 2) if isinstance(dur, (int, float)) else dur)
        row += 1

    # Block 2: hour distribution
    row += 1
    ws.cell(row=row, column=1, value="Hourly distribution").font = Font(bold=True, size=11)
    row += 1
    for c, h in enumerate(["Hour", "Kills"], 1):
        ws.cell(row=row, column=c, value=h)
    style_header(ws, row, 2)
    row += 1

    hours = data.get("hourly_distribution", {})
    for hour in range(24):
        key = f"{hour:02d}"
        count = hours.get(key, 0) if isinstance(hours, dict) else 0
        ws.cell(row=row, column=1, value=f"{hour:02d}:00")
        ws.cell(row=row, column=2, value=count)
        row += 1

    # Block 3: lifetime totals
    row += 1
    ws.cell(row=row, column=1, value="Lifetime totals").font = Font(bold=True, size=11)
    row += 1
    for k in ("total_kills", "total_duration_seconds", "longest_streak_seconds"):
        if k in data:
            ws.cell(row=row, column=1, value=k)
            v = data[k]
            ws.cell(row=row, column=2, value=round(v, 2) if isinstance(v, float) else v)
            row += 1
    auto_size(ws)


def build_roast_history(wb):
    ws = wb.create_sheet("Roast History")
    ws["A1"] = "Roast History — data/roast_history.json"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:E1")

    headers = ["#", "Timestamp", "Kill #", "Context (current event)", "Roast"]
    for c, h in enumerate(headers, 1):
        ws.cell(row=3, column=c, value=h)
    style_header(ws, 3, len(headers))

    hist_path = Path("data/roast_history.json")
    if not hist_path.exists():
        ws.cell(row=4, column=1, value="(no roast_history.json yet)")
        return
    try:
        entries = json.loads(hist_path.read_text(encoding="utf-8"))
    except Exception as e:
        ws.cell(row=4, column=1, value=f"Parse error: {e}")
        return

    for i, entry in enumerate(entries, start=1):
        r = 3 + i
        ws.cell(row=r, column=1, value=i)
        ws.cell(row=r, column=2, value=entry.get("timestamp", ""))
        ws.cell(row=r, column=3, value=entry.get("kill_count", ""))
        ws.cell(row=r, column=4, value=entry.get("current_event", ""))
        ws.cell(row=r, column=5, value=entry.get("roast", ""))
    auto_size(ws, max_width=80)


def build_environment(wb):
    ws = wb.create_sheet("Environment")
    ws["A1"] = "Environment"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:B1")

    headers = ["Component", "Version / Status"]
    for c, h in enumerate(headers, 1):
        ws.cell(row=3, column=c, value=h)
    style_header(ws, 3, len(headers))

    def version_of(modname: str) -> str:
        try:
            mod = __import__(modname)
            return getattr(mod, "__version__", "(installed, no __version__)")
        except Exception as e:
            return f"NOT INSTALLED ({e.__class__.__name__})"

    rows = [
        ("Python",         f"{platform.python_version()} ({platform.python_implementation()})"),
        ("Platform",       f"{platform.system()} {platform.release()}"),
        ("Flask",          version_of("flask")),
        ("openpyxl",       version_of("openpyxl")),
        ("OpenCV",         version_of("cv2")),
        ("Ultralytics",    version_of("ultralytics")),
        ("numpy",          version_of("numpy")),
        ("psutil",         version_of("psutil")),
        ("requests",       version_of("requests")),
        ("edge-tts",       version_of("edge_tts")),
        ("google-api-python-client", version_of("googleapiclient")),
        ("Working dir",    str(Path.cwd())),
        ("Report file",    "BigBrother_Audit_Report.xlsx"),
    ]
    for i, (k, v) in enumerate(rows, start=4):
        ws.cell(row=i, column=1, value=k)
        ws.cell(row=i, column=2, value=v)
    auto_size(ws, max_width=80)


# ── Main ─────────────────────────────────────────────────────────────────────
ENDPOINTS_TO_PROBE = [
    # method,  path,                         body,    expect_failure_ok
    ("GET",    "/",                          None,    False),
    ("GET",    "/camera",                    None,    False),
    ("GET",    "/stats",                     None,    False),
    ("GET",    "/settings",                  None,    False),
    ("GET",    "/static/style.css",          None,    False),
    ("GET",    "/api/status",                None,    False),
    ("GET",    "/api/settings",              None,    False),
    ("POST",   "/api/settings",              {"camera_enabled": True}, False),
    ("GET",    "/api/cameras",               None,    False),
    ("GET",    "/api/calendar",              None,    False),
    ("GET",    "/api/weekly-plan",           None,    False),
    ("GET",    "/api/stats",                 None,    False),
    ("GET",    "/api/roast/latest",          None,    False),
    ("GET",    "/api/roast/history",         None,    False),
    ("POST",   "/api/roast/speak",           {},      True),   # 404 when no roast yet
    ("GET",    "/api/guess",                 None,    False),
    ("GET",    "/api/camera/feed",           None,    False),
    ("POST",   "/api/calendar/complete",     {"event_id": "audit-noop"}, True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000",
                        help="Base URL of the running BigBrother server")
    parser.add_argument("--out", default="BigBrother_Audit_Report.xlsx",
                        help="Output xlsx filename")
    args = parser.parse_args()

    print(f"Probing {args.url} ...")
    probes: list[ProbeResult] = []
    for method, path, body, expect_fail in ENDPOINTS_TO_PROBE:
        r = probe(args.url, method, path, body, expect_failure_ok=expect_fail)
        flag = r.status
        print(f"  {flag:9s} {method:5s} {path:32s} "
              f"{r.http_status or '-':>4}  {r.latency_ms or '-':>7} ms")
        probes.append(r)

    print(f"Building workbook → {args.out}")
    wb = openpyxl.Workbook()
    # openpyxl ships with one default "Sheet" — remove it.
    default = wb.active
    wb.remove(default)

    build_overview(wb, probes, args.url)
    build_endpoints(wb, probes)
    build_ui_inventory(wb, args.url)
    build_live_state(wb, args.url)
    build_stats(wb)
    build_roast_history(wb)
    build_environment(wb)

    wb.save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
