import sys
import os
import cv2

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        images.append(frame)
    cap.release()
    
    return images
    
def main():
    video_path = sys.argv[1]
    if sys.argv[2]:
        output_folder = sys.argv[2] 
    else:
        output_folder = './frames'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    extracted_images = extract_frames(video_path)
    
    for i, image in enumerate(extracted_images):
        cv2.imwrite(os.path.join(output_folder, f'{i}.jpg'), image)
     