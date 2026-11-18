"""
server.py — Flask web server for BigBrother dashboard.

Serves the dashboard UI and exposes JSON API endpoints consumed
by the frontend via AJAX polling.
"""

import time
import logging
import threading
from typing import Optional

import cv2
import numpy as np
from flask import (
    Flask, render_template, jsonify, request, Response,
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
        latest_roast_id = getattr(_coach, "roast_id", 0) if _coach else 0
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
        "latest_roast_id": latest_roast_id,
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


# In-memory MP3 cache keyed by (voice, rate, pitch, volume, text) so repeated
# dashboard polls don't re-render the same roast. Hard cap so we don't leak.
_TTS_CACHE: dict = {}
_TTS_CACHE_MAX = 32
_TTS_LOCK = threading.Lock()

# Voice rotation. edge-tts voices are inherently TTS-flat, so the way to
# fight monotony is variety, not louder prosody. Each roast picks a voice
# deterministically from its roast_id so the same roast always renders the
# same way (so the MP3 cache hits) but a sequence of roasts cycles voices.
#
# Each tuple: (voice, rate, pitch, volume). Hand-tuned by ear.
_VOICE_ROTATION = [
    # Brian: most expressive of the US set, slight rasp
    ("en-US-BrianNeural",              "+10%", "-3Hz",  "+30%"),
    # Ryan: British sneer, very different rhythm — breaks monotony hard
    ("en-GB-RyanNeural",               "+5%",  "-3Hz",  "+30%"),
    # Andrew: deep authoritative news-anchor, lands the longer cinema lines
    ("en-US-AndrewMultilingualNeural", "+10%", "-5Hz",  "+30%"),
]
_VOICE_DEFAULT  = _VOICE_ROTATION[0][0]
_RATE_DEFAULT   = _VOICE_ROTATION[0][1]
_PITCH_DEFAULT  = _VOICE_ROTATION[0][2]
_VOLUME_DEFAULT = _VOICE_ROTATION[0][3]


def _pick_voice(roast_id: int) -> tuple:
    """Deterministically rotate voices by roast_id so the audio for a given
    roast is stable (cacheable) but a series of roasts cycles voices."""
    return _VOICE_ROTATION[roast_id % len(_VOICE_ROTATION)]


def _render_edge_tts(
    text: str,
    voice: str = _VOICE_DEFAULT,
    rate: str = _RATE_DEFAULT,
    pitch: str = _PITCH_DEFAULT,
    volume: str = _VOLUME_DEFAULT,
) -> Optional[bytes]:
    """Synchronously render `text` to MP3 bytes via edge-tts. Cached."""
    key = (voice, rate, pitch, volume, text)
    with _TTS_LOCK:
        if key in _TTS_CACHE:
            return _TTS_CACHE[key]
    try:
        import asyncio, edge_tts
        async def _do():
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
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
    /api/roast/audio?text=<roast-text> and plays the MP3. Caching is keyed on
    (voice, rate, pitch, volume, text) so the same roast is rendered once."""
    text = request.args.get("text", "").strip()
    if not text and _coach is not None:
        text = _coach.get_latest_roast()
    if not text:
        return jsonify({"error": "No text"}), 404

    # Voice rotation: if the caller specifies a voice explicitly, honour it;
    # otherwise pick by roast_id (or by text hash as a fallback so manual
    # ▶ SPEAK on the same roast gets the same voice every time).
    if "voice" in request.args:
        voice  = request.args.get("voice",  _VOICE_DEFAULT)
        rate   = request.args.get("rate",   _RATE_DEFAULT)
        pitch  = request.args.get("pitch",  _PITCH_DEFAULT)
        volume = request.args.get("volume", _VOLUME_DEFAULT)
    else:
        rid = getattr(_coach, "roast_id", 0) if _coach else 0
        # If the text matches the current roast, use roast_id. Otherwise
        # derive a stable rotation slot from the text itself so a re-render
        # of an old roast always sounds the same.
        if _coach and text == _coach.get_latest_roast():
            slot = rid
        else:
            slot = hash(text) & 0x7fffffff
        voice, rate, pitch, volume = _pick_voice(slot)
    mp3 = _render_edge_tts(text, voice, rate, pitch, volume)
    if mp3 is None:
        return jsonify({"error": "TTS render failed"}), 500
    return Response(mp3, mimetype="audio/mpeg")


@app.route("/api/guess")
def api_guess():
    """Educated guess about what the user should be doing.

    Calendar-aware: when a current event is active the guess names it; when
    one is starting soon it mentions the runway; otherwise it falls back to
    time-of-day phrasing. The coach handles all that internally now."""
    if _coach is None:
        return jsonify({"guess": "You should be working. Obviously."}), 200
    has_event = False
    if _calendar and _calendar.is_authenticated:
        has_event = _calendar.get_current_event() is not None
    return jsonify({"guess": _coach.generate_guess(), "has_current_event": has_event})


_CAMERAS_CACHE: dict = {"at": 0.0, "data": None}
_CAMERAS_CACHE_TTL = 30.0   # seconds


@app.route("/api/cameras")
def api_cameras():
    """List available camera indices.

    Cached because probing on Windows means briefly opening each /dev/videoN —
    and opening a camera the live detector is using steals the device. We
    serve from a short-lived cache, and even when we DO probe we skip the
    detector's active index so the live feed doesn't break."""
    now = time.time()
    if (_CAMERAS_CACHE["data"] is not None
            and now - _CAMERAS_CACHE["at"] < _CAMERAS_CACHE_TTL):
        return jsonify(_CAMERAS_CACHE["data"])

    from core.phone_detector import PhoneDetector
    current = _detector.camera_index if _detector else 0
    # Tell list_cameras to skip the live detector's index and inject what we
    # already know about it so the UI still sees a row for the active camera.
    in_use = {}
    if _detector and _detector._cap is not None and _detector._cap.isOpened():
        try:
            in_use[current] = {
                "width":  int(_detector._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(_detector._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps":    _detector._cap.get(cv2.CAP_PROP_FPS),
            }
        except Exception:
            in_use[current] = {}
    cameras = PhoneDetector.list_cameras(
        skip_indices=[current] if _detector else None,
        in_use=in_use,
    )
    payload = {"cameras": cameras, "current": current}
    _CAMERAS_CACHE["data"] = payload
    _CAMERAS_CACHE["at"]   = now
    return jsonify(payload)


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """Get or update settings."""
    if request.method == "GET":
        payload = dict(_config)
        payload["camera_enabled"] = bool(_detector.enabled) if _detector else True
        # Surface strict_mode at the top level for the settings UI. It's also
        # under detection.* in the config dict, but the settings page reads
        # flat fields elsewhere — keep it consistent.
        payload["strict_mode"] = bool(getattr(_detector, "strict_mode", False)) if _detector else False
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
    if "strict_mode" in data and _detector:
        _detector.update_settings(strict_mode=bool(data["strict_mode"]))
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
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[:] = (20, 20, 20)  # near-black
        cv2.putText(img, "CAMERA OFF", (170, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (90, 90, 90), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        _OFF_PLACEHOLDER_JPEG = buf.tobytes() if ok else b""
    return _OFF_PLACEHOLDER_JPEG


def _generate_mjpeg():
    """Generator yielding multipart MJPEG frames for the browser.

    The format here matters more than you'd think — Chromium tiles partial
    frames in a 3×3 grid when it loses track of part boundaries, and a
    missing Content-Length on a payload that happens to contain the
    boundary token can trip it. We always include Content-Length and we
    never sleep without yielding so the parser doesn't desync.
    """
    BOUNDARY = b"--frame\r\n"
    last_frame_id: int = -1
    last_sent: float = 0.0
    keepalive_seconds = 1.0

    while True:
        frame = None
        if _detector is not None and _detector.enabled:
            frame = _detector.get_annotated_frame()

        # Validate the frame before handing it to imencode. A wrong-shape
        # or non-uint8 buffer (e.g. greyscale, partially-written, or YUV
        # left over from a backend hiccup) can encode into a gray JPEG.
        valid = (
            frame is not None
            and getattr(frame, "ndim", 0) == 3
            and frame.shape[2] == 3
            and frame.dtype == np.uint8
        )

        if not valid:
            jpeg_bytes = _get_off_placeholder()
            this_id = -1  # placeholder uses a sentinel id
        else:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                jpeg_bytes = _get_off_placeholder()
                this_id = -1
            else:
                jpeg_bytes = buf.tobytes()
                # id() of the numpy array is stable while the array is alive
                # and changes when the detector replaces _latest_frame, so we
                # can use it as a cheap "is this a new frame?" signal.
                this_id = id(frame)

        # If nothing new has happened, sleep briefly without emitting —
        # but only for a short window so the browser's MJPEG parser stays
        # warm. After the window we re-emit the same frame to keep the
        # connection alive (important when the camera is off and the
        # placeholder is the only thing to send).
        if this_id == last_frame_id and time.time() - last_sent < keepalive_seconds:
            time.sleep(0.066)
            continue
        last_frame_id = this_id
        last_sent = time.time()

        header = (
            BOUNDARY
            + b"Content-Type: image/jpeg\r\n"
            + b"Content-Length: " + str(len(jpeg_bytes)).encode("ascii") + b"\r\n"
            + b"\r\n"
        )
        yield header + jpeg_bytes + b"\r\n"
        time.sleep(0.066)  # ~15 FPS cap


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

def run_server(host: str = "127.0.0.1", port: int = 5000):
    """Run the Flask development server (called from main.py in a thread)."""
    logger.info("Starting web server on %s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
