import argparse
import socket

import torch
from dotenv import load_dotenv

from flash_checker import capture_screen, detect_flashbang
from machine_learning import FlashbangModel, test_transform


def main():
    parser = argparse.ArgumentParser(description="Detect flashbangs and send the result over the network") 
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    args = parser.parse_args()

    load_dotenv()

    PORT = args.port
    HOST_NAME = socket.gethostname()
    HOST = socket.gethostbyname(HOST_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FlashbangModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device ,weights_only=True))
    model.eval()
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        
        print(f"Server started at {HOST}:{PORT}")
        
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                
                image = capture_screen()
                predicted = detect_flashbang(image, model, device)
                
                conn.sendall(str(predicted).encode())
                
                print(f"Flashbang detected: {predicted}")
                
    print("Server closed")
    

    
    
if __name__ == "__main__":
    main()