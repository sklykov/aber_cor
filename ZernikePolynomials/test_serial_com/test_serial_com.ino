/*
 * Test of timeouts for pyserial library and simple communication with python code.
*/
void setup()
{
  Serial.begin(115200); // send and receive at 115200 baud
  delay(2);
  if(Serial){
    Serial.println("Testing connection opened");
  }
  delay(2); 
}


void loop(){
  delay(2); 
  if (Serial.available() > 0){
    String readString = Serial.readString();
    Serial.println("Received command: " + readString);
    if (readString == "?"){
      Serial.println("Request of the state received, the state is ok, if you read this!");
    }
  } 
  delay(3); 
}
