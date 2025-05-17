import argparse
import time

import cv2
import dxcam
import numpy as np
import torch
from PIL import Image

from machine_learning import FlashbangModel, test_transform
from multiprocessing import Queue

def initialize_screen_capture(virtual_camera=0):
    capture = cv2.VideoCapture(virtual_camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    return capture

def capture_screen(queue: Queue, fps=20, monitor=0):
    camera = dxcam.create(output_idx=monitor)
    camera.start(target_fps=fps)
    
    while True:
        frame = camera.get_latest_frame()
        if frame is not None and not queue.full():
            queue.put(frame)
        time.sleep(1 / fps)
    
def display_image(image: Image.Image, flashbangDetected: bool) -> None:
    frame = np.array(image)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if flashbangDetected:
        cv2.putText(frame, "Flashbang Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No Flashbang", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Flashbang Detection", frame)

def detect_flashbang(image: np.ndarray, model: FlashbangModel, device: torch.device) -> bool:
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