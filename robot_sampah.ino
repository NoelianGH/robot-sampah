#include <Servo.h>

Servo lidServo;

const int enaA = 10;
const int inA  = 11;
const int enaB = 12;
const int inB  = 13;

const int servoPin = 9;
const int pingPin = 8;

int distance_cm;

const int openAngle = 180;
const int closeAngle = 0;

void setMotorsStop() {
  digitalWrite(enaA, LOW);
  digitalWrite(enaB, LOW);
  digitalWrite(inA, LOW);
  digitalWrite(inB, LOW);
}

void setMotorsForward() {
  digitalWrite(enaA, HIGH); digitalWrite(inA, HIGH);
  digitalWrite(enaB, HIGH); digitalWrite(inB, HIGH);
}

void setMotorsBackward() {
  digitalWrite(enaA, HIGH); digitalWrite(inA, LOW);
  digitalWrite(enaB, HIGH); digitalWrite(inB, LOW);
}

void setTurnRightContinuous() {
  digitalWrite(enaA, HIGH); digitalWrite(inA, HIGH);
  digitalWrite(enaB, LOW); digitalWrite(inB, LOW);
}

void setTurnLeftContinuous() {
  digitalWrite(enaA, LOW); digitalWrite(inA, LOW);
  digitalWrite(enaB, HIGH); digitalWrite(inB, HIGH);
}

void setup() {
  pinMode(enaA, OUTPUT);
  pinMode(inA, OUTPUT);
  pinMode(enaB, OUTPUT);
  pinMode(inB, OUTPUT);
  pinMode(pingPin, OUTPUT); 

  lidServo.attach(servoPin);

  Serial.begin(9600);
  delay(200);
  lidServo.write(closeAngle);
  setMotorsStop();

  Serial.println("ARDUINO READY");
}

unsigned long lastDistMillis = 0;
const unsigned long DIST_INTERVAL_MS = 200;

int measureDistanceCm() {
  pinMode(pingPin, OUTPUT);
  digitalWrite(pingPin, LOW);
  delayMicroseconds(2);
  digitalWrite(pingPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(pingPin, LOW);

  pinMode(pingPin, INPUT);
  long duration = pulseIn(pingPin, HIGH, 30000);
  if (duration <= 0) return -1;
  int cm = duration * 0.034 / 2;
  return cm;
}

void doServoCycle() {
  lidServo.write(openAngle);
  delay(1500);
  lidServo.write(closeAngle);
  delay(500);
}

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
  } else {
    Serial.print("UNKNOWN_CMD:");
    Serial.println(cmd);
  }
}

String serialBuffer = "";

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(serialBuffer);
      serialBuffer = "";
    } else if (c != '\r') {
      serialBuffer += c;
    }
  }

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
    }
  }
}