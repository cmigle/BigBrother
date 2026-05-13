"""
server.py — Flask web server for BigBrother dashboard.

Serves the dashboard UI and exposes JSON API endpoints consumed
by the frontend via AJAX polling.
"""

import time
import logging
import threading
from io import BytesIO
from typing import Optional

import cv2
from flask import (
    Flask, render_template, jsonify, request, Response, send_from_directory,
)

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# These will be injected by main.py before the server starts
_detector = None
_scare = None
_coach = None
_calendar = None
_stats = None
_config = {}
_process_monitor = None
_lock = threading.Lock()


def init_app(detector, scare_system, ai_coach, calendar_sync, stats_tracker, config, process_monitor=None):
    """Inject dependencies into the Flask app."""
    global _detector, _scare, _coach, _calendar, _stats, _config, _process_monitor
    _detector = detector
    _scare = scare_system
    _coach = ai_coach
    _calendar = calendar_sync
    _stats = stats_tracker
    _config = config
    _process_monitor = process_monitor


# ======================================================================
# Page routes
# ======================================================================

@app.route("/")
def dashboard():
    """Main dashboard page."""
    return render_template("dashboard.html")


@app.route("/stats")
def stats_page():
    """Statistics page."""
    return render_template("stats.html")


@app.route("/settings")
def settings_page():
    """Settings page."""
    return render_template("settings.html")


@app.route("/camera")
def camera_page():
    """Camera feed page."""
    return render_template("camera.html")


# ======================================================================
# API endpoints
# ======================================================================

@app.route("/api/status")
def api_status():
    """Current detection state, cooldown, kill count, latest roast."""
    with _lock:
        today = _stats.get_today_stats() if _stats else {}
        streak = _stats.get_streak() if _stats else 0

        phone_detected = False
        confidence = 0.0
        face_visible = True
        actively_on_phone = False
        on_cooldown = False

        if _detector and _detector._latest_result:
            phone_detected    = _detector._latest_result.detected
            confidence        = _detector._latest_result.confidence
            face_visible      = _detector._latest_result.face_visible
            actively_on_phone = _detector._latest_result.actively_on_phone

        if _scare:
            on_cooldown = _scare.is_on_cooldown()

        latest_roast = _coach.get_latest_roast() if _coach else ""
        latest_guess = _coach.get_guess() if _coach else ""

        # Process / browser distraction
        distraction_detected = False
        distraction_label    = ""
        distraction_source   = ""
        distraction_duration = 0.0
        if _process_monitor:
            pm = _process_monitor.get_latest()
            distraction_detected = pm.detected
            distraction_label    = pm.label
            distraction_source   = pm.source
            distraction_duration = round(pm.detection_duration, 1)

    return jsonify({
        "phone_detected": phone_detected,
        "confidence": round(confidence, 3),
        "face_visible": face_visible,
        "actively_on_phone": actively_on_phone,
        "on_cooldown": on_cooldown,
        "kill_count_today": today.get("kill_count", 0),
        "current_streak": round(streak, 1),
        "latest_roast": latest_roast,
        "latest_guess": latest_guess,
        "distraction_detected": distraction_detected,
        "distraction_label":    distraction_label,
        "distraction_source":   distraction_source,
        "distraction_duration": distraction_duration,
        "camera_enabled":      bool(_detector.enabled) if _detector else False,
    })


@app.route("/api/stats")
def api_stats():
    """Full stats data + roast history."""
    if _stats is None:
        return jsonify({"error": "Stats not available"}), 503

    all_stats = _stats.get_all_stats()
    roast_history = _coach.get_roast_history() if _coach else []

    return jsonify({
        **all_stats,
        "roast_history": roast_history,
    })


@app.route("/api/calendar")
def api_calendar():
    """Today's + tomorrow's calendar events, plus pending tasks."""
    if _calendar is None or not _calendar.is_authenticated:
        return jsonify({"events": [], "tomorrow_events": [], "tasks": [], "authenticated": False})

    events          = _calendar.get_today_events()
    tomorrow_events = _calendar.get_tomorrow_events()
    tasks           = _calendar.get_tasks()
    return jsonify({
        "events": events,
        "tomorrow_events": tomorrow_events,
        "tasks": tasks,
        "authenticated": True,
    })


@app.route("/api/weekly-plan")
def api_weekly_plan():
    """Day-by-day plan for the rest of the week (events + tasks grouped by day)."""
    if _calendar is None or not _calendar.is_authenticated:
        return jsonify({"days": [], "authenticated": False})
    days = _calendar.get_week_plan()
    return jsonify({"days": days, "authenticated": True})


@app.route("/api/calendar/complete", methods=["POST"])
def api_calendar_complete():
    """Mark a calendar event as completed."""
    if _calendar is None or not _calendar.is_authenticated:
        return jsonify({"error": "Calendar not available"}), 503

    data = request.get_json()
    event_id = data.get("event_id") if data else None
    if not event_id:
        return jsonify({"error": "event_id required"}), 400

    success = _calendar.complete_task(event_id)
    return jsonify({"success": success})


@app.route("/api/roast/latest")
def api_roast_latest():
    """Latest AI coach roast text."""
    roast = _coach.get_latest_roast() if _coach else ""
    return jsonify({"roast": roast})


@app.route("/api/roast/history")
def api_roast_history():
    """All roasts with timestamps."""
    history = _coach.get_roast_history() if _coach else []
    return jsonify({"history": history})


@app.route("/api/roast/speak", methods=["POST"])
def api_roast_speak():
    """Trigger server-side TTS for the latest roast."""
    if _coach is None:
        return jsonify({"error": "Coach not available"}), 503
    roast = _coach.get_latest_roast()
    if not roast:
        return jsonify({"error": "No roast to speak"}), 404
    threading.Thread(target=_coach.speak, args=(roast,), daemon=True).start()
    return jsonify({"success": True, "roast": roast})


# In-memory MP3 cache keyed by (voice, text) so repeated dashboard polls
# don't re-render the same roast. Hard cap so we don't leak forever.
_TTS_CACHE: dict = {}
_TTS_CACHE_MAX = 32
_TTS_LOCK = threading.Lock()


def _render_edge_tts(text: str, voice: str = "en-US-GuyNeural") -> Optional[bytes]:
    """Synchronously render `text` to MP3 bytes via edge-tts. Cached."""
    key = (voice, text)
    with _TTS_LOCK:
        if key in _TTS_CACHE:
            return _TTS_CACHE[key]
    try:
        import asyncio, edge_tts
        async def _do():
            comm = edge_tts.Communicate(text, voice)
            chunks = bytearray()
            async for ev in comm.stream():
                if ev.get("type") == "audio":
                    chunks.extend(ev["data"])
            return bytes(chunks)
        loop = asyncio.new_event_loop()
        try:
            mp3 = loop.run_until_complete(_do())
        finally:
            loop.close()
    except Exception as exc:
        logger.error("edge-tts render failed: %s", exc)
        return None
    with _TTS_LOCK:
        if len(_TTS_CACHE) >= _TTS_CACHE_MAX:
            _TTS_CACHE.pop(next(iter(_TTS_CACHE)))
        _TTS_CACHE[key] = mp3
    return mp3


@app.route("/api/roast/audio")
def api_roast_audio():
    """Return the latest roast as MP3 audio for browser playback.

    The dashboard polls /api/status and discovers a new roast; it then loads
    /api/roast/audio?key=<roast-text> and plays the MP3. Caching is keyed on
    the text so the same roast is rendered exactly once."""
    text = request.args.get("text", "").strip()
    if not text and _coach is not None:
        text = _coach.get_latest_roast()
    if not text:
        return jsonify({"error": "No text"}), 404
    voice = request.args.get("voice", "en-US-GuyNeural")
    mp3 = _render_edge_tts(text, voice)
    if mp3 is None:
        return jsonify({"error": "TTS render failed"}), 500
    return Response(mp3, mimetype="audio/mpeg")


@app.route("/api/guess")
def api_guess():
    """Educated guess about what the user should be doing (no calendar event)."""
    if _coach is None:
        return jsonify({"guess": "You should be working. Obviously."}), 200
    # Check if there's a current event — only generate guess if nothing scheduled
    has_event = False
    if _calendar and _calendar.is_authenticated:
        ev = _calendar.get_current_event()
        has_event = ev is not None
    guess = _coach.generate_guess() if not has_event else ""
    return jsonify({"guess": guess, "has_current_event": has_event})


@app.route("/api/cameras")
def api_cameras():
    """List all available camera indices detected on this machine."""
    from core.phone_detector import PhoneDetector
    cameras = PhoneDetector.list_cameras()
    current = _detector.camera_index if _detector else 0
    return jsonify({"cameras": cameras, "current": current})


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """Get or update settings."""
    if request.method == "GET":
        payload = dict(_config)
        payload["camera_enabled"] = bool(_detector.enabled) if _detector else True
        return jsonify(payload)

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Apply settings
    if "confidence_threshold" in data and _detector:
        _detector.update_settings(confidence=float(data["confidence_threshold"]))
    if "volume" in data and _scare:
        _scare.update_settings(volume=float(data["volume"]))
    if "cooldown_seconds" in data and _scare:
        _scare.update_settings(cooldown=float(data["cooldown_seconds"]))
    if "escalation_enabled" in data and _scare:
        _scare.update_settings(escalation_enabled=bool(data["escalation_enabled"]))
    if "filter_screens" in data and _detector:
        _detector.update_settings(filter_screens=bool(data["filter_screens"]))
    if "camera_index" in data and _detector:
        new_cam = int(data["camera_index"])
        if new_cam != _detector.camera_index:
            import threading as _t
            _t.Thread(target=_detector.switch_camera, args=(new_cam,), daemon=True).start()
    if "camera_enabled" in data and _detector:
        want = bool(data["camera_enabled"])
        if want != bool(_detector.enabled):
            threading.Thread(target=_detector.set_enabled, args=(want,), daemon=True).start()
    if "pm_enabled" in data and _process_monitor:
        _process_monitor.update_settings(enabled=bool(data["pm_enabled"]))
    if "pm_check_web" in data and _process_monitor:
        _process_monitor.update_settings(check_web=bool(data["pm_check_web"]))
    if "pm_check_apps" in data and _process_monitor:
        _process_monitor.update_settings(check_apps=bool(data["pm_check_apps"]))

    return jsonify({"success": True})


# ======================================================================
# Camera feed (MJPEG stream)
# ======================================================================

_OFF_PLACEHOLDER_JPEG: Optional[bytes] = None


def _get_off_placeholder() -> bytes:
    """Return a cached 'camera off' JPEG so the MJPEG stream stays alive
    even when the detector is disabled. Without this, the <img> tag holds
    a stale first-frame because no new boundary ever arrives."""
    global _OFF_PLACEHOLDER_JPEG
    if _OFF_PLACEHOLDER_JPEG is None:
        import numpy as _np
        img = _np.zeros((360, 640, 3), dtype=_np.uint8)
        img[:] = (20, 20, 20)  # near-black
        cv2.putText(img, "CAMERA OFF", (170, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (90, 90, 90), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        _OFF_PLACEHOLDER_JPEG = buf.tobytes() if ok else b""
    return _OFF_PLACEHOLDER_JPEG


def _generate_mjpeg():
    """Generator that yields MJPEG frames from the detector."""
    last_jpeg: Optional[bytes] = None
    while True:
        jpeg_bytes: Optional[bytes] = None

        if _detector is not None and _detector.enabled:
            frame = _detector.get_annotated_frame()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    jpeg_bytes = buf.tobytes()
        else:
            # Yield a static placeholder so the browser keeps the stream
            # alive and immediately repaints when the camera comes back on.
            jpeg_bytes = _get_off_placeholder()

        if jpeg_bytes is None:
            time.sleep(0.05)
            continue

        # Skip emitting an identical frame back-to-back — avoids re-encoding
        # and re-sending the same placeholder bytes 15× per second.
        if jpeg_bytes is last_jpeg:
            time.sleep(0.2)
            continue
        last_jpeg = jpeg_bytes

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )
        time.sleep(0.066)  # ~15 FPS cap when frames are flowing


@app.route("/api/camera/feed")
def camera_feed():
    """MJPEG stream with YOLOv8 bounding box overlay."""
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ======================================================================
# Server runner
# ======================================================================

def run_server(host: str = "0.0.0.0", port: int = 5000):
    """Run the Flask development server (called from main.py in a thread)."""
    logger.info("Starting web server on %s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
