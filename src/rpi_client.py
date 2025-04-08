import argparse
import socket
import RPi.GPIO as GPIO

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi client for flashbang detection")
    parser.add_argument("--host", type=str, required=True, help="Host to connect to")
    parser.add_argument("--port", type=int, required=True, help="Port to connect to")
    parser.add_argument("--pin", type=int, required=True, help="GPIO pin to connect to in BCM mode")
    args = parser.parse_args()
    
    PIN = args.pin
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN, GPIO.OUT)
    GPIO.output(PIN, GPIO.LOW)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                s.connect((args.host, args.port))
                print(f"Connected to server at {args.host}:{args.port}")
                
                while True:
                    data = s.recv(1024)
                    if not data:
                        break
                    
                    flashbang_detected = data.decode()
                    print(f"Flashbang detected: {flashbang_detected}")
                    
                    if flashbang_detected == "1":
                        GPIO.output(PIN, GPIO.HIGH)
                    else:
                        GPIO.output(PIN, GPIO.LOW)
            except BlockingIOError:
                pass
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()
        s.close()
        print("Socket closed")
if __name__ == "__main__":
    main()