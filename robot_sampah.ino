// robot_sampah.ino
#include <Servo.h>

Servo lidServo;

const int enaA = 5;   // enable A
const int mA  = 4;   // motor A direction
const int enaB = 7;   // enable B
const int mB  = 6;   // motor B direction
const int pwm  = 100;

const int servoPin = 9;
const int pingPin = 8; // trig/echo style (same approach as earlier sketch)

long duration_cm;
int distance_cm;

const int openDistance = 30;  // cm threshold for opening lid

const int openAngle = 180;
const int closeAngle = 0;

bool lidOpen = false;

// Current movement state for debugging
String currentState = "STOP";

void setMotorsStop() {
  digitalWrite(enaA, LOW);
  digitalWrite(enaB, LOW);
  analogWrite(mA, 0);
  analogWrite(mB, 0);
  currentState = "STOP";
}

void setMotorsForward() {
  digitalWrite(enaA, HIGH); analogWrite(mA, pwm);
  digitalWrite(enaB, LOW); analogWrite(mB, pwm);
  currentState = "FORWARD";
}

void setMotorsBackward() {
  digitalWrite(enaA, LOW); analogWrite(mA, pwm);
  digitalWrite(enaB, HIGH); analogWrite(mB, pwm);
  currentState = "BACK";
}

void setTurnRightContinuous() {
  // right turn by stopping right motor, left motor forward
  digitalWrite(enaA, LOW); analogWrite(mA, pwm); 
  digitalWrite(enaB, LOW); analogWrite(mB, pwm);   
  currentState = "TURN_RIGHT";
}

void setTurnLeftContinuous() {
  // left turn by stopping left motor, right motor forward
  digitalWrite(enaA, HIGH); analogWrite(mA, pwm);   
  digitalWrite(enaB, HIGH); analogWrite(mB, pwm); 
  currentState = "TURN_LEFT";
}

void setup() {
  pinMode(enaA, OUTPUT);
  pinMode(mA, OUTPUT);
  pinMode(enaB, OUTPUT);
  pinMode(mB, OUTPUT);
  pinMode(pingPin, OUTPUT); // we'll toggle to trigger then read pulse

  lidServo.attach(servoPin);

  Serial.begin(9600);
  delay(200);
  lidServo.write(closeAngle);
  setMotorsStop();

  Serial.println("ARDUINO READY");
}

unsigned long lastDistMillis = 0;
const unsigned long DIST_INTERVAL_MS = 200; // send distance every 200ms

int measureDistanceCm() {
  // ping
  pinMode(pingPin, OUTPUT);
  digitalWrite(pingPin, LOW);
  delayMicroseconds(2);
  digitalWrite(pingPin, HIGH);
  delayMicroseconds(5);
  digitalWrite(pingPin, LOW);

  pinMode(pingPin, INPUT);
  long duration = pulseIn(pingPin, HIGH, 30000); // timeout 30ms
  if (duration <= 0) return -1;
  int cm = duration * 0.034 / 2;
  return cm;
}
/*
void openLid() {
  for (int pos = 180; pos >= 40; pos--) {
    lidServo.write(pos);
    delay(15);
  }
  lidOpen = true;
}

void closeLid() {
  for (int pos = 40; pos <= 180; pos++) {
    myservo.write(pos);
    delay(15);
  }
  lidOpen = false;
}
*/
void doServoCycle() {
  // open -> wait -> close
  lidServo.write(openAngle);
  delay(1500);
  lidServo.write(closeAngle);
  delay(500);
}

// handle a simple command (line without newline)
void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd == "STOP") {
    setMotorsStop();
    Serial.println("ACK:STOP");
  } else if (cmd == "FORWARD") {
    setMotorsForward();
    Serial.println("ACK:FORWARD");
  } else if (cmd == "BACK") {
    setMotorsBackward();
    Serial.println("ACK:BACK");
  } else if (cmd == "TURN_RIGHT") {
    setTurnRightContinuous();
    Serial.println("ACK:TURN_RIGHT");
  } else if (cmd == "TURN_LEFT") {
    setTurnLeftContinuous();
    Serial.println("ACK:TURN_LEFT");
  } else if (cmd == "SERVO") {
    Serial.println("ACK:SERVO");
    doServoCycle();
    Serial.println("ACK:SERVO_DONE");
  } else if (cmd == "SEARCH_STEP") {
    // small search step: rotate right briefly then stop
    setTurnRightContinuous();
    delay(600);
    setMotorsStop();
    Serial.println("ACK:SEARCH_STEP");
  } else {
    // unknown: ignore
    Serial.print("UNKNOWN_CMD:");
    Serial.println(cmd);
  }
}

String serialBuffer = "";

void loop() {
  // read serial: commands end with '\n'
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(serialBuffer);
      serialBuffer = "";
    } else if (c != '\r') {
      serialBuffer += c;
    }
  }

  // periodic distance measurement + publish
  unsigned long now = millis();
  if (now - lastDistMillis >= DIST_INTERVAL_MS) {
    lastDistMillis = now;
    int cm = measureDistanceCm();
    if (cm < 0) {
      Serial.println("DIST:-1");
    } else {
      distance_cm = cm;
      Serial.print("DIST:");
      Serial.println(distance_cm);
      // auto lid behaviour (optional): keep commented out so PC controls when servo runs
      // if (distance_cm > 0 && distance_cm <= openDistance && !lidOpen) { lidServo.write(openAngle); lidOpen=true; }
      // else if (distance_cm > openDistance && lidOpen) { lidServo.write(closeAngle); lidOpen=false; }
    }
  }

  // nothing else blocking in loop, commands are handled asynchronously
}
