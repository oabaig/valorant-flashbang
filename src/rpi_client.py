import argparse

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi client for flashbang detection")
    parser.add_argument("--host", type=str, required=True, help="Host to connect to")
    parser.add_argument("--port", type=int, required=True, help="Port to connect to")
    parser.add_argument("--pin", type=int, required=True, help="GPIO pin to connect to")
    args = parser.parse_args()


if __name__ == "__main__":
    main()