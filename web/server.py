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
_lock = threading.Lock()


def init_app(detector, scare_system, ai_coach, calendar_sync, stats_tracker, config):
    """Inject dependencies into the Flask app."""
    global _detector, _scare, _coach, _calendar, _stats, _config
    _detector = detector
    _scare = scare_system
    _coach = ai_coach
    _calendar = calendar_sync
    _stats = stats_tracker
    _config = config


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
        on_cooldown = False

        if _detector and _detector._latest_result:
            phone_detected = _detector._latest_result.detected
            confidence = _detector._latest_result.confidence

        if _scare:
            on_cooldown = _scare.is_on_cooldown()

        latest_roast = _coach.get_latest_roast() if _coach else ""
        latest_guess = _coach.get_guess() if _coach else ""

    return jsonify({
        "phone_detected": phone_detected,
        "confidence": round(confidence, 3),
        "on_cooldown": on_cooldown,
        "kill_count_today": today.get("kill_count", 0),
        "current_streak": round(streak, 1),
        "latest_roast": latest_roast,
        "latest_guess": latest_guess,
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
    """Today's calendar events."""
    if _calendar is None or not _calendar.is_authenticated:
        return jsonify({"events": [], "authenticated": False})

    events = _calendar.get_today_events()
    return jsonify({
        "events": events,
        "authenticated": True,
    })


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
        return jsonify(_config)

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

    return jsonify({"success": True})


# ======================================================================
# Camera feed (MJPEG stream)
# ======================================================================

def _generate_mjpeg():
    """Generator that yields MJPEG frames from the detector."""
    while True:
        frame = None
        if _detector is not None:
            frame = _detector.get_annotated_frame()

        if frame is None:
            time.sleep(0.1)
            continue

        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )
        time.sleep(0.066)  # ~15 FPS


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
