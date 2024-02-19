#include <Servo.h>

Servo shoulder;  // 180-degree Servo
Servo elbow;     // 180-degree Servo
Servo base;      // 180-degree Servo
Servo gripper;   // 180-degree Servo

const int IDLE_CODE = -1;
const int SHOULDER_CODE = 3;
const int ELBOW_CODE = 5;
const int GRIPPER_CODE = 10;
const int BASE_CODE = 9;


String DONE_MSG = "okay";


unsigned long previousMillis = 0;
const long interval = 250;

int code = -1;
int angle = -1;
int ang_vel = 100;




void setup() {
  Serial.begin(9600);

  Serial.println("Setup started!");

  shoulder.attach(3);
  elbow.attach(5);
  base.attach(9);
  gripper.attach(10);

  rest();

  Serial.println("Setup completed!");
}





void loop() {

  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    int old_code = code;

    if (code == IDLE_CODE){
      Serial.println("I'm idle...");

      if (Serial.available() >= 2){
        Serial.println("Received new code");
        code = Serial.read();
        angle = Serial.read();
      }

    }

    else if (code == SHOULDER_CODE){
      Serial.println("Working on Shoulder");
      code = shoulderTo(angle, ang_vel);
    }

    else if (code == ELBOW_CODE){
      Serial.println("Working on Elbow");
      code = elbowTo(angle, ang_vel);
    }

    else if (code == GRIPPER_CODE){
      Serial.println("Working on Gripper");
      code = gripperTo(angle, ang_vel);
    }

    else if (code == BASE_CODE){
      Serial.println("Working on Base");
      code = baseTo(angle, ang_vel);
    }

    else{
        Serial.println("Unknown command. Going in rest position.");
        rest();
        code = IDLE_CODE;
        write_done();
    }

    if ((old_code != IDLE_CODE) && (code == IDLE_CODE)){
      Serial.println("Task done!");
      write_done();
    }
  }

}



int shoulderTo(int target, int vel) {
  int pos = shoulder.read();

  if (pos <= target){
    pos += vel;

    if (pos >= target){
      shoulder.write(target);
      return IDLE_CODE;
    }
    else{
      shoulder.write(pos);
    }

  }

  else{
    pos -= vel;

    if (pos <= target){
      shoulder.write(target);
      return IDLE_CODE;

    }
    else{
      shoulder.write(pos);
    }

  }

  return SHOULDER_CODE;

}



int elbowTo(int target, int vel) {
  int pos = elbow.read();

  if (pos <= target){
    pos += vel;

    if (pos >= target){
      elbow.write(target);
      return IDLE_CODE;
    }
    else{
      elbow.write(pos);
    }

  }

  else{
    pos -= vel;

    if (pos <= target){
      elbow.write(target);
      return IDLE_CODE;

    }
    else{
      elbow.write(pos);
    }

  }

  return ELBOW_CODE;
}


int baseTo(int target, int vel) {
  int pos = base.read();

  if (pos <= target){
    pos += vel;

    if (pos >= target){
      base.write(target);
      return IDLE_CODE;
    }
    else{
      base.write(pos);
    }

  }

  else{
    pos -= vel;

    if (pos <= target){
      base.write(target);
      return IDLE_CODE;

    }
    else{
      base.write(pos);
    }

  }

  return BASE_CODE;
}


int gripperTo(int target, int vel) {
  int pos = gripper.read();

  if (pos <= target){
    pos += vel;

    if (pos >= target){
      gripper.write(target);
      return IDLE_CODE;
    }
    else{
      gripper.write(pos);
    }

  }

  else{
    pos -= vel;

    if (pos <= target){
      gripper.write(target);
      return IDLE_CODE;
    }
    else{
      gripper.write(pos);
    }

  }

  return GRIPPER_CODE;
}



void rest(){
  Serial.println("Going to rest position.");
  bool is_moving = true;

  while (is_moving){
    unsigned long currentMillis = millis();

    if (currentMillis - previousMillis >= interval) {
      Serial.println("working on Gripper Rest position");
      previousMillis = currentMillis;

      int g = gripperTo(180, ang_vel);

      if (g == IDLE_CODE){
        is_moving = false;
      }
    }
  }

  is_moving = true;
  while (is_moving){
    unsigned long currentMillis = millis();

    if (currentMillis - previousMillis >= interval) {
      Serial.println("working on Elbow Rest position");
      previousMillis = currentMillis;

      int e = elbowTo(30, ang_vel);

      if (e == IDLE_CODE){
        is_moving = false;
      }
    }
  }

  is_moving = true;
  while (is_moving){
    unsigned long currentMillis = millis();

    if (currentMillis - previousMillis >= interval) {
      Serial.println("working on Shoulder Rest position");
      previousMillis = currentMillis;

      int s = shoulderTo(90, ang_vel);

      if (s == IDLE_CODE){
        is_moving = false;
      }
    }
  }

  is_moving = true;
  while (is_moving){
    unsigned long currentMillis = millis();

    if (currentMillis - previousMillis >= interval) {
      Serial.println("working on Base Rest position");
      previousMillis = currentMillis;

      int b = baseTo(90, ang_vel);

      if (b == IDLE_CODE){
        is_moving = false;
      }
    }
  }

  Serial.println("Resting position reached.");
}


void write_done(){
  Serial.println(DONE_MSG);
}
