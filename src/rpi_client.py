import argparse
import socket
import RPi.GPIO as GPIO

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi client for flashbang detection")
    parser.add_argument("--host", type=str, required=True, help="Host to connect to")
    parser.add_argument("--port", type=int, required=True, help="Port to connect to")
    parser.add_argument("--pin", type=int, required=True, help="GPIO pin to connect to")
    args = parser.parse_args()
    
    # connect to the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.host, args.port))
        
        print(f"Connected to server at {args.host}:{args.port}")
        
        while True:
            data = s.recv(1024)
            if not data:
                break
            
            flashbang_detected = data.decode()
            print(f"Flashbang detected: {flashbang_detected}")
            
            # Here you can add code to control the GPIO pin based on the flashbang detection
            # For example, using RPi.GPIO library to control the pin
            # import RPi.GPIO as GPIO
            # GPIO.setmode(GPIO.BCM)
            # GPIO.setup(args.pin, GPIO.OUT)
            # GPIO.output(args.pin, GPIO.HIGH if flashbang_detected == "True" else GPIO.LOW)
            # GPIO.cleanup()


if __name__ == "__main__":
    main()