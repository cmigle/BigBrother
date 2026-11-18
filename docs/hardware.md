# Hardware: the ESP32 trigger

The physical part of the scare: an ESP32 drives a servo that pulls the trigger of a Nerf-style blaster. The PC fires it over Wi-Fi, or over USB serial as a fallback.

```
PC (BigBrother) ──Wi-Fi: GET /trigger──▶ ESP32 ──PWM──▶ servo ──▶ blaster trigger
                └─USB:   FIRE\n ────────┘
```

## Parts

| Part | Notes |
|---|---|
| ESP32 dev board | Any board supported by the Arduino ESP32 core, e.g. an ESP32 DevKit v1 |
| SG90 or MG90S servo | MG90S has metal gears and copes better with stiff triggers |
| Nerf-style blaster | The servo horn sits behind the trigger |
| USB cable and jumper wires | USB powers the board and carries the serial fallback |

## Wiring

| Servo wire | ESP32 pin |
|---|---|
| Signal (orange/yellow) | GPIO 23 |
| Power (red) | 5V / VIN |
| Ground (brown/black) | GND |

The sketch drives the servo with 500–2400 µs pulses. It rests at 40°, swings to 170° to pull the trigger, holds for 500 ms and returns. Change `servoPin`, `startAngle` and `endAngle` at the top of the sketch to match your mount.

## Flashing

1. Install the [Arduino IDE](https://www.arduino.cc/en/software), the **esp32** board package (Boards Manager) and the **ESP32Servo** library (Library Manager).
2. Open [`firmware/trigger_servo/trigger_servo.ino`](../firmware/trigger_servo/trigger_servo.ino) and set `ssid` and `password` to your Wi-Fi network. Don't commit these.
3. Select your board and port, then upload.
4. Open the Serial Monitor at **115200 baud**. The board prints its IP address and announces itself as `http://servo.local`.

## Connecting it to BigBrother

### Wi-Fi (preferred)

In `config/local.yaml`:

```yaml
hardware:
  wifi_enabled: true
  arduino_url: "http://servo.local"   # or the IP from the Serial Monitor
```

On a scare the app sends `GET /trigger` in a background thread, so a slow network never blocks detection. At startup it requests the base URL, without `/trigger`, to check the board is reachable without firing it. If `servo.local` doesn't resolve on your network, use the IP address instead.

### USB serial

```yaml
hardware:
  wifi_enabled: false
  serial_enabled: true
  serial_port: "COM3"        # /dev/ttyUSB0 on Linux
  baud_rate: 115200
```

Install `pyserial` (`pip install pyserial`). The app sends `FIRE\n`, and the board fires on `FIRE` or `1`. If both modes are enabled, Wi-Fi wins.

## Testing without the app

```bash
curl http://servo.local/trigger     # should print OK and pull the trigger
```

Or type `1` and press Enter in the Serial Monitor.

## Safety

Foam darts still hurt at close range. Point the blaster at your chest or back, never at your face, and don't leave it loaded and armed around people who haven't agreed to it.
