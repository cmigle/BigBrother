#include <ESP32Servo.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>

// ── WiFi credentials ──────────────────────────────────────────────
const char* ssid     = "xxx";
const char* password = "xxx";
// ──────────────────────────────────────────────────────────────────

WebServer server(80);

Servo myServo;
const int servoPin   = 23;
const int startAngle = 40;
const int endAngle   = 170;

void fireTrigger() {
  Serial.println("Trigger received! Moving...");
  myServo.write(endAngle);
  delay(500);
  myServo.write(startAngle);
  Serial.println("Action complete. Waiting for trigger...");
}

void setup() {
  Serial.begin(115200);

  // Allocate timers for ESP32 PWM
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  myServo.setPeriodHertz(50);
  myServo.attach(servoPin, 500, 2400);
  myServo.write(startAngle);

  // Connect to WiFi
  Serial.printf("Connecting to %s", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP address: ");
  Serial.println(WiFi.localIP());
  Serial.println("Enter this IP in the web settings panel.");

  // Advertise as servo.local so the PC can find the trigger without a fixed IP
  if (MDNS.begin("servo")) {
    MDNS.addService("http", "tcp", 80);
    Serial.println("Reachable as http://servo.local");
  }

  // HTTP endpoint: GET /trigger
  server.on("/trigger", HTTP_GET, []() {
    fireTrigger();
    server.send(200, "text/plain", "OK");
  });

  server.begin();
  Serial.println("--- Trigger Servo Control (WiFi) ---");
  Serial.println("Listening for GET /trigger");
}

void loop() {
  server.handleClient();

  // USB serial trigger: "FIRE" from the PC app, or "1" + Enter from the Serial Monitor
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "FIRE" || cmd == "1") fireTrigger();
  }
}
