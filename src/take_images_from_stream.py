# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-09-24 19:57:33
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-09-24 21:26:01


import os
from uuid import uuid4

import click
import cv2

def main():
    save_dir = 'assets/frames'
    video = cv2.VideoCapture('/dev/video0')

    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = video.read()
        if ret == False: break
        # frame = frame[:,::-1,:]

        cv2.imshow('vide-stream', frame)
        k = cv2.waitKey(1) & 0xFF

        if k == 27 or k == ord('q'):
            break

        if k == ord('s'):
            filename = '{}.jpg'.format(uuid4().hex)
            filename = os.path.join(save_dir, filename)

            print('Saving {} at {}'.format(filename, save_dir))
            cv2.imwrite(filename, frame)

if __name__ == "__main__":
    main()