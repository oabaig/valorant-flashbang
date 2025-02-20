import argparse
import pathlib
import cv2

""" Gathers all the images of a folder and its subfolders and saves them in a new folder. 
    Only supports jpeg images for now.
    Args:
    input (str): path to the starting folder
    output (str): path to the output folder
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", '-i', type=str, help="Path to the starting folder", required=True)
    parser.add_argument("--output", '-o', type=str, help="Path to the output folder", required=True)
    parser.add_argument("--rename", '-r', action="store_true", help="Rename the files to a sequential number")
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")
    
    if not output_path.exists():
        output_path.mkdir(parents=True)
        

    count = 0
    for path in input_path.rglob("*.jpg"):
        img = cv2.imread(str(path))
        if args.rename:
            output_file = output_path / f"{count:04d}.jpg"
        else:
            output_file = output_path / path.relative_to(input_path)
            
        cv2.imwrite(str(output_file), img)
        count += 1
            
    print("Done!")
    