from machine_learning import FlashbangModel, test_transform
import argparse
import torch
import pathlib
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Train a model to detect flashbangs")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training")
    args = parser.parse_args()
    
    MODEL = pathlib.Path(args.model)
    
    device = torch.device(args.device)
    model = FlashbangModel()
    model.load_state_dict(torch.load(MODEL, map_location=device))
    
    # test the input image
    image = Image.open(args.image)
    image = test_transform(image).unsqueeze(0)
    image = image.to(device)
    
    # test the model
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        
        if predicted.item() == 0:
            print("Flashbang detected")
        else:
            print("No flashbang detected")

if __name__ == "__main__":
    main()