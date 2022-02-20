"""
    Records frames from camera

    Client side driver workhorse process
"""
import argparse
import sys
import os
import time
import cv2
import torch
sys.path.append('../../')
from pipe import timestamp_name

def parse_args():
    parser = argparse.ArgumentParser(description='Recording settings.')

    parser.add_argument('--num_frames', default=30000, type=int,
        help='')
    return parser.parse_args()


def start_record(driver_cls):
    pipe = driver_cls.pipe

    args = parse_args()

    video_cap = cv2.VideoCapture(0)
    cap_prop = lambda x : int(video_cap.get(x))
    width, height = \
        cap_prop(cv2.CAP_PROP_FRAME_WIDTH), cap_prop(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Camera dimensions: {}x{}".format(height, width))

    frames = []
    start_time = time.time()
    while True:
        success, frame = video_cap.read()
        if not success or len(frames) > args.num_frames - 1:
            break
        #frames.append(frame)
        '''
        pipe.put(torch.from_numpy(frame))
        '''
        client_path = f'perception_{timestamp_name()}.txt'
        torch.save(torch.from_numpy(frame), client_path)
        pipe.put(client_path)


    print("Recording time taken : {0} seconds".format(time.time() - start_time))


if __name__ == '__main__':
    main()
