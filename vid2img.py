import cv2 as cv
import os
from tqdm import tqdm

vid_path = 'sample_video.mp4'
output_dir = 'imgs_sample'
os.makedirs(output_dir, exist_ok=True)

cap = cv.VideoCapture(vid_path)
tot = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
for i in tqdm(range(tot)):
    frame = cap.read()[1] 
    cv.imwrite(f'{output_dir}/frame_{i}.jpg', cv.resize(frame, (224, 224)))
