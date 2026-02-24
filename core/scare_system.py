"""
scare_system.py — Sound-based scare system for BigBrother.

Plays loud gunshot / alarm sounds when a phone is detected, manages
cooldown timers, and orchestrates the full scare pipeline including
AI coach roasts and optional hardware triggers.
"""

import os
import time
import random
import logging
import threading
from typing import Optional, List

logger = logging.getLogger(__name__)


class ScareSystem:
    """
    Manages the scare pipeline: sound playback, cooldown, escalation,
    and wiring to the AI coach and stats tracker.
    """

    def __init__(
        self,
        ai_coach,
        stats_tracker,
        sound_folder: str = "sounds/",
        cooldown: float = 15.0,
        volume: float = 1.0,
        serial_gun=None,
        escalation_enabled: bool = True,
        escalation_delay: float = 10.0,
    ):
        self.coach = ai_coach
        self.stats = stats_tracker
        self.sound_folder = sound_folder
        self.cooldown = cooldown
        self.volume = volume
        self.serial_gun = serial_gun
        self.escalation_enabled = escalation_enabled
        self.escalation_delay = escalation_delay

        self._last_fire_time: float = 0
        self._trigger_start: Optional[float] = None
        self._sounds: list = []
        self._alarm_sound = None
        self._reload_sound = None
        self._mixer_ready = False
        self._escalation_active = False

        self._init_audio()

    # ------------------------------------------------------------------
    # Audio initialisation
    # ------------------------------------------------------------------

    def _init_audio(self):
        """Initialise pygame mixer and pre-load sound files."""
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(8)
            self._mixer_ready = True
            logger.info("Pygame mixer initialised")
        except Exception as exc:
            logger.warning("Could not initialise pygame mixer: %s", exc)
            self._mixer_ready = False
            return

        self._load_sounds()

    def _load_sounds(self):
        """Load .wav files from the sound folder."""
        import pygame

        if not os.path.isdir(self.sound_folder):
            logger.warning("Sound folder not found: %s", self.sound_folder)
            return

        for filename in sorted(os.listdir(self.sound_folder)):
            filepath = os.path.join(self.sound_folder, filename)
            if not filename.lower().endswith(".wav"):
                continue
            try:
                snd = pygame.mixer.Sound(filepath)
                if filename.startswith("alarm"):
                    self._alarm_sound = snd
                elif filename.startswith("reload"):
                    self._reload_sound = snd
                else:
                    self._sounds.append(snd)
                logger.info("Loaded sound: %s", filename)
            except Exception as exc:
                logger.warning("Could not load %s: %s", filename, exc)

        if not self._sounds:
            logger.warning("No gunshot sounds loaded from %s", self.sound_folder)

    # ------------------------------------------------------------------
    # Trigger / cooldown
    # ------------------------------------------------------------------

    def is_on_cooldown(self) -> bool:
        """Check if the cooldown timer is still active."""
        return (time.time() - self._last_fire_time) < self.cooldown

    def trigger(self, detection_duration: float = 0.0):
        """
        Full scare sequence:
          1. Check cooldown
          2. Play random gunshot sound
          3. Record kill in StatsTracker
          4. Call AI Coach to generate + speak a roast
          5. Fire serial gun if connected
          6. Update cooldown timer
        """
        if self.is_on_cooldown():
            logger.debug("Scare on cooldown — skipping trigger")
            return

        logger.info("=== SCARE TRIGGERED === (phone visible for %.1fs)", detection_duration)

        # 1. Play sound
        self._play_random_gunshot()

        # 2. Record kill
        self.stats.record_kill(detection_duration)

        # 3. AI Coach roast (in separate thread so we don't block detection)
        threading.Thread(target=self._run_coach, daemon=True).start()

        # 4. Hardware trigger
        if self.serial_gun is not None:
            try:
                self.serial_gun.fire()
            except Exception as exc:
                logger.warning("Serial gun fire failed: %s", exc)

        # 5. Cooldown
        self._last_fire_time = time.time()
        self._trigger_start = time.time()
        self._escalation_active = False

    def escalate(self):
        """
        More aggressive response when the phone is STILL detected after
        the initial scare + escalation delay.
        """
        if not self.escalation_enabled:
            return

        logger.info("=== ESCALATION ===")
        self._escalation_active = True

        # Rapid-fire sounds
        if self._alarm_sound and self._mixer_ready:
            import pygame
            self._alarm_sound.set_volume(self.volume)
            self._alarm_sound.play(loops=-1)  # loop until stopped

        # Harder roast
        threading.Thread(target=self._run_coach, daemon=True).start()

    def stop_escalation(self):
        """Stop continuous alarm when phone is put away."""
        if self._escalation_active and self._alarm_sound:
            self._alarm_sound.stop()
            self._escalation_active = False
            logger.info("Escalation stopped — phone put away")

            # Play reload sound if available
            if self._reload_sound and self._mixer_ready:
                self._reload_sound.set_volume(self.volume * 0.5)
                self._reload_sound.play()

    def should_escalate(self, detection_duration: float) -> bool:
        """Check if we should escalate based on time since initial trigger."""
        if not self.escalation_enabled:
            return False
        if self._trigger_start is None:
            return False
        elapsed = time.time() - self._trigger_start
        return (
            elapsed > self.escalation_delay
            and not self._escalation_active
        )

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _play_random_gunshot(self):
        """Play a random gunshot sound from the loaded pool."""
        if not self._mixer_ready or not self._sounds:
            logger.warning("No sounds available to play")
            return

        sound = random.choice(self._sounds)
        sound.set_volume(self.volume)
        sound.play()
        logger.info("Gunshot sound played")

    def _run_coach(self):
        """Run AI coach roast (intended to run in a thread)."""
        try:
            self.coach.fire()
        except Exception as exc:
            logger.error("AI Coach error: %s", exc)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def update_settings(
        self,
        volume: Optional[float] = None,
        cooldown: Optional[float] = None,
        escalation_enabled: Optional[bool] = None,
    ):
        if volume is not None:
            self.volume = max(0.0, min(1.0, volume))
        if cooldown is not None:
            self.cooldown = cooldown
        if escalation_enabled is not None:
            self.escalation_enabled = escalation_enabled
