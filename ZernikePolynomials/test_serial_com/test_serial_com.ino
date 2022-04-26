/*
 * Test of timeouts for pyserial library and simple communication with python code.
*/
void setup()
{
  Serial.begin(115200); // send and receive at 115200 baud
  Serial.setTimeout(10); // sets small timeout for waiting the string buffer
  delay(10);
  if(Serial){
    Serial.write("Testing connection opened\n");
    // Serial.println("Connection opened");
  }
}


void loop(){
  delay(10);
  if (Serial.available() > 0){
     // reading of an entire string much less effective, but still affordable
     String readString = Serial.readString();  // by default, readString has 1 s timeout
     Serial.println("Echoed command: " + readString);
     if (readString == "?"){
        delay(2); 
        Serial.println("Special request of the state received. The state is ok.");
     }

    // reading of single char - fast and effictive, tested
//    char readChar = Serial.read();
//    if (readChar == '?'){
//      Serial.print("Request of the state received, the state is ok, if you read this.");
//    }
//    if (readChar == '!'){
//      Serial.write('!');
//    }
    delay(5);
  } 

}
