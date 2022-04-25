/*
 * Test of timeouts for pyserial library and simple communication with python code.
*/
void setup()
{
  Serial.begin(115200); // send and receive at 115200 baud
  delay(50);
  // Serial.setTimeout(10); // sets small timeout for waiting the string buffer
  if(Serial){
    // Serial.write("Testing connection opened\n");
    Serial.println("Connection opened");
  }
}


void loop(){
  delay(5);
  if (Serial.available() > 0){
    // String readString = Serial.readString();  // by default, readString has 1 s timeout
    // Serial.println("Received command: " + readString);
    char readChar = Serial.read();
    if (readChar == '?'){
      Serial.println("Request of the state received, the state is ok, if you read this.");
    }
    if (readChar == '!'){
      Serial.write('!');
    }
    delay(10);
  } 

}
