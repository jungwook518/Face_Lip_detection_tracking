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
            
            if frame_num == 3000 :
                frame_num = -1
                break
    
            # last video frame and no faces
            if faces is not ():
                buffer_face_find[frame_num-1]=1
            elif faces is ():
                if ret == 1:
                    buffer_face_find[frame_num-1]=0
                else :
                    frame_num = -1
                    print("can not detect")
                    break
            
            #if ret == 0 and faces is ():
            #    frame_num = -1
            #    print("Can not detect")
            #    break
        pdb.set_trace()
        for i in range(2750):
            a = np.sum(buffer_face_find[0:0+250])
            maxframe = a
            want_frame = 0
            b = np.sum(buffer_face_find[i:i+250])
            if maxframe < b:
                maxframe = b
                want_frame = i        
        # cheack video duration
        cap         = cv2.VideoCapture(mp4_file)
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
        last_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # save video information
        fps             = round(fps)
        #start_frame     = frame_num - 1
        start_frame     = want_frame 
        #end_frame       = round(start_frame + duration_s * fps)
        end_frame       = want_frame + 250
        last_frame      = round(last_frame)
        if end_frame > last_frame:
            end_frame = last_frame
        start_time      = round(start_frame / fps, 3)
        end_time        = round(end_frame / fps, 3)
        duration_ss     = end_time - start_time
        # crop & save video
        pdb.set_trace()
        file_name       = mp4_file.split('/')[-1]
        origin_path     = mp4_file.split(file_name)[0]
        save_path       = origin_path.replace('youtube_download', SAVE_ROOT)
        save_file       = os.path.join(save_path, file_name)
        command         = "ffmpeg -ss {} -t 10 -i {} {}".format(start_time, mp4_file, save_file)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        subprocess.call(command, shell=True, stdout=None)
        
        if frame_num == -1:
            continue
    except:
        pass

