#include <Servo.h>

Servo lidServo;

//const int trigPin = 7;
//const int echoPin = 6;

const int pingPin = 9;

long duration;
int distance;

// Define distance threshold for detection (in cm)
const int openDistance = 30;

// Define servo positions
const int openAngle = 180;
const int closeAngle = 0;

// For smooth behavior
bool lidOpen = false;

void setup() {
  lidServo.attach(10);
  //pinMode(trigPin, OUTPUT);
  //pinMode(echoPin, INPUT);
  Serial.begin(9600);

  // Start with lid closed
  lidServo.write(closeAngle);
}

void loop() {
  // --- Measure distance ---
  pinMode(pingPin, OUTPUT);
  digitalWrite(pingPin, LOW);
  delayMicroseconds(2);
  digitalWrite(pingPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(pingPin, LOW);
  
  pinMode(pingPin, INPUT);
  duration = pulseIn(pingPin, HIGH);
  distance = duration * 0.034 / 2;  // convert to cm

  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // --- Control lid based on distance ---
  if (distance > 0 && distance <= openDistance && !lidOpen) {
    // Open lid
    lidServo.write(openAngle);
    lidOpen = true;
    Serial.println("Lid opened");
  } 
  else if (distance > openDistance && lidOpen) {
    // Close lid
    lidServo.write(closeAngle);
    lidOpen = false;
    Serial.println("Lid closed");
  }

  delay(200);  // Small delay for stability
}