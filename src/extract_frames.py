from pathlib import Path
import cv2
import argparse

def extract_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        print(f'Extracting frame {count}')
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = output_path / f'{count:04d}.jpg'
        cv2.imwrite(str(frame_path), frame)
        
        count += 1
    cap.release()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract frames from a video')
    parser.add_argument('--input', '-i', required=True, help='Path to the video file')
    parser.add_argument('--output', '-o', help='Folder to save the extracted frames')
    args = parser.parse_args()
    
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f'File {input_path} does not exist')
        exit(1)
        
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.parent / input_path.stem
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f'Extracting frames from {input_path} to {output_path}')
    
    extract_frames(input_path, output_path)
    