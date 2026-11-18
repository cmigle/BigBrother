"""
serial_gun.py — Optional Arduino hardware interface for BigBrother.

Supports two transport modes:
  • WiFi  — sends GET /trigger to the Arduino's HTTP server (preferred).
  • Serial — sends "FIRE\\n" over a USB/serial connection (legacy fallback).

Gracefully falls back if no hardware is connected.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class SerialGun:
    """
    Optional hardware interface for physical scare hardware.

    WiFi mode (recommended): sends an HTTP GET request to the Arduino's
    /trigger endpoint.  Configure with wifi_enabled=True and arduino_url.

    Serial mode (legacy): sends ASCII commands over a serial connection.
    """

    def __init__(
        self,
        port: str = "COM3",
        baud_rate: int = 115200,
        enabled: bool = False,
        wifi_enabled: bool = False,
        arduino_url: str = "http://192.168.1.100",
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.enabled = enabled
        self.wifi_enabled = wifi_enabled
        # Ensure the URL ends with /trigger
        base = arduino_url.rstrip("/")
        self.trigger_url = base + "/trigger"

        self._serial = None
        self._connected = False

        if self.wifi_enabled:
            # No persistent connection needed for HTTP — mark as ready
            self._connected = True
            logger.info("Arduino WiFi trigger ready → %s", self.trigger_url)
            # Background probe so we log reachability without blocking startup.
            threading.Thread(target=self._wifi_probe, daemon=True).start()
        elif self.enabled:
            self.connect()

    def _wifi_probe(self):
        """Best-effort reachability check at startup. Pure diagnostics — does
        not affect runtime behaviour, just makes failures discoverable."""
        try:
            import urllib.request
            # Hit the base URL, not /trigger, so we don't fire the gun on boot.
            base = self.trigger_url.rsplit("/trigger", 1)[0] or self.trigger_url
            with urllib.request.urlopen(base, timeout=4) as resp:
                logger.info("Gun reachability probe OK (HTTP %d at %s)", resp.status, base)
        except Exception as exc:
            logger.warning("Gun reachability probe failed (%s) — will retry on next fire", exc)

    # ------------------------------------------------------------------
    # Serial (legacy)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Fire
    # ------------------------------------------------------------------

    def fire(self):
        """Send fire command to the hardware (non-blocking)."""
        if self.wifi_enabled:
            threading.Thread(target=self._wifi_fire, daemon=True).start()
        elif self._connected and self._serial is not None:
            self._serial_fire()
        else:
            logger.debug("Gun not connected — skipping fire")

    def _wifi_fire(self):
        """Send HTTP GET /trigger to the ESP32, fire-and-forget.

        The ESP32 keeps the HTTP response open until the servo physically
        finishes moving (~3 s). urllib.request would block that whole time
        even in a thread, so threads pile up under rapid fire and the
        'trigger sent' log lands seconds after the servo already moved.

        Instead we open a raw TCP socket, write the request line, then
        close without reading the response. The kernel TCP stack delivers
        the bytes to the ESP32, which fires immediately — we just never
        see the 'OK' it sends back. Round-trip drops from ~3 s to <50 ms."""
        import socket
        from urllib.parse import urlparse
        parsed = urlparse(self.trigger_url)
        host = parsed.hostname
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Connection: close\r\n"
            f"User-Agent: BigBrother\r\n"
            f"\r\n"
        ).encode("ascii")
        sock = None
        try:
            # Short connect timeout — if the ESP32 is unreachable we want
            # to know fast, not stall a thread. send() is non-blocking
            # enough on LAN that we don't bother with a separate send timeout.
            sock = socket.create_connection((host, port), timeout=1.5)
            sock.sendall(request)
            logger.info("Arduino WiFi trigger sent → %s (fire-and-forget)", self.trigger_url)
        except Exception as exc:
            logger.warning("Arduino WiFi trigger failed: %s", exc)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    def _serial_fire(self):
        """Send FIRE command over serial."""
        try:
            self._serial.write(b"FIRE\n")
            logger.info("Serial gun FIRE command sent")
        except Exception as exc:
            logger.error("Serial gun fire failed: %s", exc)
            self._connected = False

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    def disconnect(self):
        """Close the serial connection (no-op for WiFi mode)."""
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
