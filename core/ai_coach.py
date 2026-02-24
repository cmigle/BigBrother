"""
ai_coach.py — The Drill Sergeant AI Coach for BigBrother.

Generates sarcastic, drill-sergeant-style roasts when the user is caught
picking up their phone. Uses a local LLM via Ollama for dynamic roasts,
falls back to pre-written templates from config/roasts.yaml.
"""

import time
import json
import random
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
import yaml

logger = logging.getLogger(__name__)

DRILL_SERGEANT_PROMPT = """You are a sarcastic, over-the-top drill sergeant AI coach in an anti-phone-distraction app called BigBrother.
The user just got caught picking up their phone. A loud gunshot sound was played to scare them.
Now deliver a short (2-3 sentences max), funny, biting roast telling them what they should be doing instead.

Context:
- Current time: {current_time}
- Current calendar event: {current_event}
- Next upcoming event: {next_event}
- Times caught today: {kill_count}
- Time since last caught: {time_since_last}

Rules:
- Be funny and sarcastic but not mean-spirited
- Reference their actual calendar event by name
- If they've been caught many times today, roast them harder
- Keep it under 3 sentences
- Start with a reaction to the gunshot (e.g., "BANG!", "Another one!", "Again?!")
"""


class AICoach:
    """
    Generates sarcastic drill-sergeant roasts about the user's phone usage.

    Tries Ollama first; falls back to template roasts from roasts.yaml.
    Speaks the roast via TTS (non-blocking) and logs to roast_history.json.
    """

    def __init__(
        self,
        calendar_sync,
        stats_tracker,
        ollama_model: str = "llama3",
        ollama_url: str = "http://localhost:11434",
        tts_enabled: bool = True,
        tts_engine: str = "pyttsx3",
        fallback_roasts_path: str = "config/roasts.yaml",
        roast_history_path: str = "data/roast_history.json",
    ):
        self.calendar = calendar_sync
        self.stats = stats_tracker
        self.model = ollama_model
        self.url = ollama_url.rstrip("/")
        self.tts_enabled = tts_enabled
        self.tts_engine = tts_engine
        self.last_roast: str = ""
        self.roast_history_path = Path(roast_history_path)

        self._fallback_roasts: List[str] = []
        self._tts_lock = threading.Lock()
        self._last_roast_time: float = 0

        self._load_fallback_roasts(fallback_roasts_path)

    # ------------------------------------------------------------------
    # Fallback roast loading
    # ------------------------------------------------------------------

    def _load_fallback_roasts(self, path: str):
        """Load pre-written roast templates from YAML."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._fallback_roasts = data.get("roasts", [])
            logger.info("Loaded %d fallback roasts from %s", len(self._fallback_roasts), path)
        except FileNotFoundError:
            logger.warning("Fallback roasts file not found: %s", path)
            self._fallback_roasts = [
                "BANG! Put the phone down. You have things to do.",
                "Again?! That's {kill_count} times today. Get back to work.",
                "BANG! Your calendar is crying right now.",
            ]
        except Exception as exc:
            logger.error("Error loading fallback roasts: %s", exc)
            self._fallback_roasts = ["BANG! Put the phone down!"]

    # ------------------------------------------------------------------
    # Roast generation
    # ------------------------------------------------------------------

    def generate_roast(self) -> str:
        """
        Generate a sarcastic roast based on current calendar and stats.
        Tries Ollama first, falls back to template roasts.
        """
        # Gather context
        current_event = self._get_current_event_text()
        next_event = self._get_next_event_text()
        kill_count = self.stats.get_today_stats().get("kill_count", 0)
        time_since = self._format_time_since_last()
        current_time = datetime.now().strftime("%I:%M %p")

        # Try Ollama
        roast = self._try_ollama(current_time, current_event, next_event, kill_count, time_since)

        # Fallback
        if roast is None:
            roast = self._fallback_roast(current_event, next_event, kill_count)

        self.last_roast = roast
        self._last_roast_time = time.time()

        # Log to history
        self._log_roast(roast, current_event, next_event, kill_count)

        return roast

    def _try_ollama(
        self, current_time: str, current_event: str, next_event: str,
        kill_count: int, time_since: str,
    ) -> Optional[str]:
        """Attempt to generate a roast via Ollama. Returns None on failure."""
        prompt = DRILL_SERGEANT_PROMPT.format(
            current_time=current_time,
            current_event=current_event,
            next_event=next_event,
            kill_count=kill_count,
            time_since_last=time_since,
        )

        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            roast = data.get("response", "").strip()
            if roast:
                logger.info("Ollama roast generated successfully")
                return roast
        except requests.ConnectionError:
            logger.warning("Ollama not reachable at %s — using fallback", self.url)
        except requests.Timeout:
            logger.warning("Ollama timed out — using fallback")
        except Exception as exc:
            logger.warning("Ollama error: %s — using fallback", exc)

        return None

    def _fallback_roast(self, current_event: str, next_event: str, kill_count: int) -> str:
        """Pick a random template roast and fill in placeholders."""
        template = random.choice(self._fallback_roasts)

        # Compute minutes until next event (rough)
        minutes_until = "??"
        if self.calendar is not None:
            nxt = self.calendar.get_next_event()
            if nxt and nxt.get("start"):
                try:
                    from datetime import datetime as dt
                    start = dt.fromisoformat(nxt["start"].replace("Z", "+00:00"))
                    diff = (start - datetime.now(start.tzinfo)).total_seconds() / 60
                    minutes_until = str(max(1, int(diff)))
                except Exception:
                    pass

        return template.format(
            event=current_event,
            next_event=next_event,
            kill_count=kill_count,
            minutes_until=minutes_until,
        )

    # ------------------------------------------------------------------
    # Calendar helpers
    # ------------------------------------------------------------------

    def _get_current_event_text(self) -> str:
        if self.calendar is None:
            return "Nothing scheduled right now"
        ev = self.calendar.get_current_event()
        if ev:
            return ev.get("summary", "Untitled event")
        return "Nothing scheduled right now"

    def _get_next_event_text(self) -> str:
        if self.calendar is None:
            return "Nothing coming up"
        ev = self.calendar.get_next_event()
        if ev:
            return ev.get("summary", "Untitled event")
        return "Nothing coming up"

    def _format_time_since_last(self) -> str:
        if self._last_roast_time == 0:
            return "First time today — let's not make it a habit"
        delta = time.time() - self._last_roast_time
        if delta < 60:
            return f"{int(delta)} seconds ago"
        elif delta < 3600:
            return f"{int(delta // 60)} minutes ago"
        else:
            return f"{int(delta // 3600)} hours ago"

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def speak(self, text: str):
        """Speak the roast out loud via TTS (non-blocking, in a thread)."""
        if not self.tts_enabled:
            return
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text: str):
        """Synchronous TTS — runs in background thread."""
        with self._tts_lock:
            try:
                if self.tts_engine == "pyttsx3":
                    self._speak_pyttsx3(text)
                elif self.tts_engine == "edge-tts":
                    self._speak_edge_tts(text)
                else:
                    logger.warning("Unknown TTS engine: %s", self.tts_engine)
            except Exception as exc:
                logger.error("TTS error: %s", exc)

    def _speak_pyttsx3(self, text: str):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def _speak_edge_tts(self, text: str):
        import asyncio
        import edge_tts
        import tempfile
        import pygame

        async def _generate():
            communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            await communicate.save(tmp_path)
            return tmp_path

        loop = asyncio.new_event_loop()
        tmp_path = loop.run_until_complete(_generate())
        loop.close()

        # Play with pygame
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def fire(self):
        """Full pipeline: generate roast → display → speak."""
        roast = self.generate_roast()
        logger.info("AI Coach says: %s", roast)
        self.speak(roast)

    # ------------------------------------------------------------------
    # History / logging
    # ------------------------------------------------------------------

    def _log_roast(self, roast: str, current_event: str, next_event: str, kill_count: int):
        """Append roast to history JSON."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "roast": roast,
            "current_event": current_event,
            "next_event": next_event,
            "kill_count": kill_count,
        }

        history = self._load_history()
        history.append(entry)

        # Keep only last 500 roasts
        if len(history) > 500:
            history = history[-500:]

        try:
            self.roast_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.roast_history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save roast history: %s", exc)

    def _load_history(self) -> list:
        """Load existing roast history."""
        try:
            if self.roast_history_path.exists():
                with open(self.roast_history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def get_roast_history(self, limit: int = 50) -> list:
        """Return the most recent roasts."""
        history = self._load_history()
        return history[-limit:]

    def get_latest_roast(self) -> str:
        """Return the last generated roast."""
        return self.last_roast
