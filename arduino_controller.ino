#include <Servo.h>

Servo shoulder;  // 180-degree MicroServo
Servo elbow;     // 180-degree MicroServo
Servo base;      // 180-degree MicroServo
Servo gripper;   // 180-degree MicroServo

const int IDLE_CODE = 0;
const int REST_CODE = 0;
const int SHOULDER_CODE = 1;
const int ELBOW_CODE = 2;
const int ELBOW_UP_CODE = 3;
const int GRIPPER_CODE = 4;
const int BASE_CODE = 5;

String DONE_MSG = "okay";




void setup() {
  Serial.begin(9600);

  shoulder.attach(3);
  elbow.attach(5);
  base.attach(9);
  gripper.attach(10);
  
  base.write(50);
  shoulder.write(50);
  elbow.write(50);
  gripper.write(50);
  //Serial.println();

 // rest();
}

void loop() {

  //Serial.print('i');

  if (Serial.available() >= 2){
  
    Serial.print("Something available ");

    long code = Serial.read();
    long angle = Serial.read();

    Serial.print("Received code and angle: ");
    Serial.print(code);
    Serial.print(" , ");
    Serial.print(angle);
  
    if (code == IDLE_CODE){
      Serial.print("Nothing to do ");
    }

    else if (code == SHOULDER_CODE){
            long pos = shoulder.read();

            Serial.println("Executing shoulder ");

            if (pos <= angle){
              for (long i = pos; i <= angle; i++){
                shoulder.write(i);
                delay(30);

                      Serial.print(i);
              }
            }
            else {
              for (long i = pos; i >= angle; i--){
                shoulder.write(i);
                delay(30);

                      Serial.print(i);
              }
            }
    }

    else if (code == ELBOW_CODE){
            long pos = elbow.read();

            Serial.println("Executing elbow ");
            
            if (pos <= angle){
              for (long i = pos; i <= angle; i++){
                elbow.write(i);
                delay(30);

                      Serial.print(i);
              }
            }
            else {
              for (long i = pos; i >= angle; i--){
                elbow.write(i);
                delay(30);

                      Serial.print(i);
              }
            }
    }

    else if (code == GRIPPER_CODE){
                    long pos = gripper.read();
              
            Serial.println("Executing gripper ");

              if (pos <= angle){
                for (long i = pos; i <= angle; i++){
                  gripper.write(i);
                  delay(30);

                        Serial.print(i);
                }
              }
              else {
                for (long i = pos; i >= angle; i--){
                  gripper.write(i);
                  
                  delay(30);

                        Serial.print(i);
                }
              }
    }

    else if (code == BASE_CODE){
                    long pos = base.read();
              
            Serial.println("Executing base ");

              if (pos <= angle){
                for (long i = pos; i <= angle; i++){
                  base.write(i);
                  delay(30);

                        Serial.print(i);
                }
              }
              else {
                for (long i = pos; i >= angle; i--){
                  base.write(i);
                  delay(30);

                        Serial.print(i);
                }
              }
    }

    else{
        Serial.println("Unknown command ");
    }

    Serial.println(DONE_MSG);
  }

}




void write_done(){
  Serial.println(DONE_MSG);
}
