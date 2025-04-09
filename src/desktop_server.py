import argparse
import socket
import sys
import torch
import time

from flash_checker import capture_screen, detect_flashbang
from machine_learning import FlashbangModel


def main():
    parser = argparse.ArgumentParser(description="Detect flashbangs and send the result over the network") 
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    args = parser.parse_args()

    temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        temp_socket.connect(("8.8.8.8", 80))
        HOST = temp_socket.getsockname()[0]
    except Exception as e:
        print(f"Error determining local IP: {e}")
        sys.exit(1)
    finally:
        temp_socket.close()

    PORT = args.port
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlashbangModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    server_socket.setblocking(False)

    clients = []

    print(f"Server started at {HOST}:{PORT}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            try:
                conn, addr = server_socket.accept()
                conn.setblocking(False)
                clients.append(conn)
                print(f"Client connected: {addr}")
            except BlockingIOError:
                pass

            image = capture_screen()
            predicted = detect_flashbang(image, model, device)
            print(f"Flashbang detected: {predicted}")

            disconnected_clients = []
            for client in clients:
                try:
                    client.sendall(str(predicted).encode())
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
                    print(f"Client disconnected or error during send: {e}")
                    disconnected_clients.append(client)
                    client.close()

            for client in disconnected_clients:
                if client in clients:
                    clients.remove(client)

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nServer interrupted. Shutting down...")
    finally:
        for client in clients:
            client.close()
        server_socket.close()
        print("Server closed.")

if __name__ == "__main__":
    main()