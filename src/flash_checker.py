import argparse
import time

import cv2
import mss
import numpy as np
import torch
from PIL import Image

from machine_learning import FlashbangModel, test_transform


def capture_screen(monitor_capture=1) -> Image.Image:
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_capture]

        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)
        return img

def display_image(image: Image.Image, flashbangDetected: bool) -> None:
    frame = np.array(image)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if flashbangDetected:
        cv2.putText(frame, "Flashbang Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No Flashbang", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Flashbang Detection", frame)
    
def detect_flashbang(image: Image.Image, model: FlashbangModel, device: torch.device) -> bool:
    transformed_image = test_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(transformed_image)
        _, predicted = torch.max(output, 1)
        
    return predicted.item() == 0 # 0 is flashbang class

    
def main():
    parser = argparse.ArgumentParser(description="Detect flashbangs in real-time")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--display", action="store_true", help="Display the screen", default=False)
    parser.add_argument("--monitor", type=int, help="Monitor region to capture", default=1)
    parser.add_argument("--fps", type=int, help="Frames per second", default=60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FlashbangModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    if args.display:
        print("Real-time flashbang detection started. Press 'q' to quit.")

    while True:
        image = capture_screen(monitor_capture=args.monitor)

        predicted = detect_flashbang(image, model, device)

        if args.display:
            display_image(image, predicted)             

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        time.sleep(1 / args.fps)

    if args.display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()