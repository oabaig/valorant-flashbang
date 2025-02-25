import torch
import cv2
import argparse
import numpy as np
from PIL import Image
from machine_learning import FlashbangModel, test_transform
import mss  

def capture_screen(region=None) -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[2]

        if region:
            monitor = {
                "top": region[1],
                "left": region[0],
                "width": region[2],
                "height": region[3]
            }

        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

def display_image(frame: np.ndarray, flashbangDetected: bool) -> None:
    if flashbangDetected:
        cv2.putText(frame, "Flashbang Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No Flashbang", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Flashbang Detection", frame)

def main():
    parser = argparse.ArgumentParser(description="Detect flashbangs in real-time")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--display", action="store_true", help="Display the screen", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FlashbangModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print("Real-time flashbang detection started. Press 'q' to quit.")

    while True:
        frame = capture_screen()

        image = Image.fromarray(frame)
        image = test_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        if args.display:
            display_image(frame, predicted.item() == 0) # 0 is flashbang class
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()