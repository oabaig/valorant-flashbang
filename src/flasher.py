import os
import numpy as np
from PIL import ImageGrab
import cv2
import torch
from dotenv import load_dotenv

import socket

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')
HOST_IP = os.getenv('HOST_IP')
PORT = int(os.getenv('PORT'))

def flashlight_on(s: socket.socket):
    s.send('FLASH'.encode('utf-8'))
    print('flashlight_on')


def main():

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(HOST_IP, PORT)
    s.connect((HOST_IP, PORT))

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=MODEL_PATH)
    model.conf = 0.4

    while True:
        video = ImageGrab.grab()
        gray_video = cv2.cvtColor(np.array(video), cv2.COLOR_BGR2RGB)
        result = model(gray_video)

        results = result.pandas().xyxy

        if not results[0].empty:
            flashlight_on(s)

        out = np.squeeze(np.array(result.render()))

        cv2.imshow('window', out)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


main()
