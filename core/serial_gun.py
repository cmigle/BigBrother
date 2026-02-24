"""
serial_gun.py — Optional Arduino/serial hardware interface for BigBrother.

Sends fire commands over a serial port to trigger a physical scare mechanism
(e.g., an Arduino-controlled nerf gun, solenoid, or airsoft mechanism).
Gracefully falls back if no hardware is connected.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SerialGun:
    """
    Optional serial interface for physical scare hardware.

    Sends simple ASCII commands over a serial connection.
    Designed for Arduino-based trigger mechanisms.
    """

    def __init__(
        self,
        port: str = "COM3",
        baud_rate: int = 9600,
        enabled: bool = False,
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.enabled = enabled
        self._serial = None
        self._connected = False

        if self.enabled:
            self.connect()

    def connect(self) -> bool:
        """Attempt to connect to the serial port."""
        if not self.enabled:
            return False

        try:
            import serial
            self._serial = serial.Serial(
                self.port,
                self.baud_rate,
                timeout=1,
            )
            self._connected = True
            logger.info("Serial gun connected on %s @ %d baud", self.port, self.baud_rate)
            return True
        except ImportError:
            logger.warning("pyserial not installed — serial gun disabled")
        except Exception as exc:
            logger.warning("Could not connect to serial port %s: %s", self.port, exc)

        self._connected = False
        return False

    def fire(self):
        """Send fire command to the hardware."""
        if not self._connected or self._serial is None:
            logger.debug("Serial gun not connected — skipping fire")
            return

        try:
            self._serial.write(b"FIRE\n")
            logger.info("Serial gun FIRE command sent")
        except Exception as exc:
            logger.error("Serial gun fire failed: %s", exc)
            self._connected = False

    def disconnect(self):
        """Close the serial connection."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
            self._connected = False
            logger.info("Serial gun disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected
