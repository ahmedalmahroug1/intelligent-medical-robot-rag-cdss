/*
 * firmware_esp32.ino
 * ------------------
 * "مستقبِل" بسيط على الـESP32: يقود 6 سيرفوات عبر PCA9685 (I2C) بأوامر نصّية.
 * العقل (بايثون + LLM) على الحاسوب؛ هذا الكود ينفّذ الأوامر فقط.
 *
 * يعمل بطريقتين في آنٍ واحد:
 *   - USB Serial (للاختبار/التطوير)
 *   - WiFi: خادم TCP على المنفذ 8080 (للتحكّم اللاسلكي)
 * كلاهما يستدعي نفس handleCommand().
 *
 * التوصيل:  PCA9685  VCC->3V3  GND->GND  SDA->GPIO21  SCL->GPIO22
 *           بطارية 5.6V:  + -> V+   - -> GND  (أرضي مشترك مع ESP32)
 *           السيرفوات على القنوات: 1, 3, 9, 11, 13, 15
 *
 * البروتوكول (سطر نصّي ينتهي بـ\n):
 *     S<id> <angle>   مثال: "S0 90"   -> حرّك السيرفو id إلى angle درجة
 *     A a0 a1 ... a5                   -> اضبط السيرفوات الستة دفعةً واحدة
 *     PING                             -> يردّ PONG (فحص الاتصال)
 *
 * يتطلّب مكتبة: Adafruit PWM Servo Driver Library
 *
 * ⚠️ ESP32 الكلاسيكي يدعم WiFi 2.4GHz فقط (لا 5GHz).
 */
#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_PWMServoDriver.h>

// ====== عدّل هذين السطرين باسم شبكتك وكلمة مرورها (شبكة 2.4GHz) ======
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const uint16_t TCP_PORT    = 8080;
// =====================================================================

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
WiFiServer server(TCP_PORT);
WiFiClient client;

const int NUM_SERVOS = 6;

// خريطة: السيرفو id في الكود -> القناة الفعلية على PCA9685
const int CH[NUM_SERVOS] = {1, 3, 9, 11, 13, 15};

// معايرة لكل سيرفو (بالميكروثانية). نبدأ بمدى آمن ضيّق ثم نوسّعه تدريجياً.
int usMin[NUM_SERVOS] = {1000, 1000, 1000, 1000, 1000, 1000};
int usMax[NUM_SERVOS] = {2000, 2000, 2000, 2000, 2000, 2000};

void setServo(int id, int angle) {
  if (id < 0 || id >= NUM_SERVOS) return;
  angle = constrain(angle, 0, 180);
  int us = map(angle, 0, 180, usMin[id], usMax[id]);
  pwm.writeMicroseconds(CH[id], us);
}

// معالج الأوامر — مستقل عن وسيلة النقل (Serial و WiFi يستدعيانه).
String handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return "ERR empty";
  char c = line.charAt(0);

  if (c == 'S') {                       // "S<id> <angle>"
    int sp = line.indexOf(' ');
    if (sp < 0) return "ERR fmt";
    int id  = line.substring(1, sp).toInt();
    int ang = line.substring(sp + 1).toInt();
    setServo(id, ang);
    return "OK S" + String(id) + " " + String(ang);
  }

  if (c == 'A') {                       // "A a0 a1 a2 a3 a4 a5"
    int idx = 1;
    for (int i = 0; i < NUM_SERVOS; i++) {
      while (idx < (int)line.length() && line.charAt(idx) == ' ') idx++;
      int sp = line.indexOf(' ', idx);
      String tok = (sp == -1) ? line.substring(idx) : line.substring(idx, sp);
      if (tok.length()) setServo(i, tok.toInt());
      idx = (sp == -1) ? line.length() : sp + 1;
    }
    return "OK A";
  }

  if (line == "PING") return "PONG";
  return "ERR " + line;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();                         // ESP32: SDA=21, SCL=22 افتراضياً
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);                   // السيرفوات التماثلية = 50Hz
  delay(10);
  for (int i = 0; i < NUM_SERVOS; i++) setServo(i, 90);  // توسيط

  // الاتصال بالشبكة
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    server.begin();
    Serial.print("WiFi OK  IP: ");
    Serial.print(WiFi.localIP());       // <-- استخدم هذا الـIP في بايثون
    Serial.print("  PORT: ");
    Serial.println(TCP_PORT);
  } else {
    Serial.println("WiFi FAILED (USB Serial still works)");
  }
  Serial.println("READY");
}

void loop() {
  // 1) USB Serial
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    Serial.println(handleCommand(line));
  }

  // 2) WiFi TCP (عميل واحد في كل مرة)
  if (!client || !client.connected()) {
    client = server.available();        // اقبل اتصالاً جديداً إن وُجد
  }
  if (client && client.connected() && client.available()) {
    String line = client.readStringUntil('\n');
    if (line.length()) client.println(handleCommand(line));
  }
}

