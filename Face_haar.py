import os
import shutil
import subprocess
import cv2
import numpy as np
import pdb

def search(d_name,li):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.mp4':
                
                li.append(os.path.join(os.path.abspath(d_name), filename))


READ_ROOT   =   './'
SAVE_ROOT   =   './save_output/'
MP4_LIST    =   []
search(READ_ROOT, MP4_LIST)

# Initialize
face_cascade= cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
fps         = 30
duration_s  = 10
for mp4_file in MP4_LIST:
    try:
        pdb.set_trace()
        print(mp4_file)
        # load video
        cap         = cv2.VideoCapture(mp4_file)
        fps         = cap.get(cv2.CAP_PROP_FPS)
        frame_num   = 0
        images      = []
        buffer_face_find = np.ones(7501)
        buffer_face_find = buffer_face_find*(-1)
        # extract image frame
        while 1:
            frame_num += 1
            ret, image = cap.read()
             # color image to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # face detection using default harr detector
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            
    except:
        pass

