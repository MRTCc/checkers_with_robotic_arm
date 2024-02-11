#include <Servo.h>

Servo shoulder;  // 180-degree MicroServo
Servo elbow;     // 180-degree MicroServo
Servo base;      // 180-degree MicroServo
Servo gripper;   //180-degree MicroServo

const int REST_CODE = 0;
const int SHOULDER_CODE = 1;
const int ELBOW_CODE = 2;
const int ELBOW_UP_CODE = 3;
const int CLOSE_GRIPPER_CODE = 4;
const int OPEN_GRIPPER_CODE = 5;
const int BASE_CODE = 6;

String DONE_MSG = "okay";

void setup() {
  Serial.begin(9600);

  //Attach and initialize servos.
  // TODO: controllare a quali pin attaccare i servo
  shoulder.attach(8);
  elbow.attach(9);
  base.attach(10);
  gripper.attach(11);
  gripper.write(70);
  base.write(100);

  rest();

  delay(500);
}




void loop() {

  if (Serial.available() >= 2){ // Check if at least 2 bytes are available
    byte code, arg;
    
    // Read the command_code and the argument from the serial port
    code = Serial.read(); // Read command code
    arg = Serial.read(); // Read argument

    if (code == REST_CODE){
      Serial.println("Received REST_CODE command.");
      // rest();
    }
    else if (code == SHOULDER_CODE){
      Serial.println("Received SHOULDER_CODE command.");
      // shoulderTo();
    }
    else if (code == ELBOW_CODE){
      Serial.println("Received ELBOW_CODE command.");
      // Handle ELBOW_CODE
    }
    else if (code == CLOSE_GRIPPER_CODE){
      Serial.println("Received CLOSE_GRIPPER_CODE command.");
      // Handle CLOSE_GRIPPER_CODE
    }
    else if (code == OPEN_GRIPPER_CODE){
      Serial.println("Received OPEN_GRIPPER_CODE command.");
      // Handle OPEN_GRIPPER_CODE
    }
    else if (code == BASE_CODE){
      Serial.println("Received BASE_CODE command.");
      // Handle BASE_CODE
    }
    else{
      Serial.println("Received unknown command.");
      // Handle unknown command
    }

    write_done();
  }

}




void shoulderTo(float target) {
  // TODO: verificare che questo aggiustamento abbia senso per il mio robot
  target += 12; // Adjusting for mechanical imperfections
  target = 180 * target / 270; // Adjusting for 270-degree servo

  float pos = shoulder.read();
  if (pos <= target){
    for (pos = pos; pos <= target; pos += 1){
      shoulder.write(pos);
      delay(30);
    }
  }
  else {
    for (pos = pos; pos >= target; pos -= 1){
      shoulder.write(pos);
      delay(30);
    }
  }
}


void elbowTo(float target) {
  // TODO: verificare che questo aggiustamento abbia senso per il mio robot
  target += 3; // Adjusting for mechanical imperfections
  target = 180 * target / 270; // Adjusting for 270-degree servo

  float pos = elbow.read();
  if (pos <= target){
    for (pos = pos; pos <= target; pos += 1){
      elbow.write(pos);
      delay(20);
    }
  }
  else {
    for (pos = pos; pos >= target; pos -= 1){
      elbow.write(pos);
      delay(20);
    }
  }
}



void rest(){
  // TODO: impostare i valori corretti per il mio robot
  shoulderTo(120);
  elbowTo(40);
}



void write_done(){
  Serial.println("done");
}