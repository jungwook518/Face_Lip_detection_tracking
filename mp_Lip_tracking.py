import face_alignment
import collections
import numpy as np
# from imutils.video import VideoStream
import argparse
# import imutils
import time
import cv2
import os, sys
import subprocess
from skimage import io
from os.path import isfile, join
from os import listdir
from utils import find_q2k, find_outliers
import pdb
import pickle
import skvideo.io
import json
from glob import glob
import multiprocessing
from multiprocessing import Process

def main(video):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create,
        "goturn":cv2.TrackerGOTURN_create
    }
                            
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu',flip_input=False, face_detector='sfd')
    fa_probs_threshold  = 0.95
    fps=30
    

    video = video
    resize_face=(88,88)
    border_face = 100

    lip_mp4_SAVE_ROOT   =   '/home/nas3/user/jungwook/Face_Lip_detection_tracking/Lip_MP4_save_output/'
    lip_label_SAVE_ROOT   =   '/home/nas3/user/jungwook/Face_Lip_detection_tracking/Lip_label_save_output/'
    lip_npy_SAVE_ROOT   =   '/home/nas3/user/jungwook/Face_Lip_detection_tracking/Lip_npy_save_output/'

    
    
  
    label_save_video_name = video.split('/')[-1][:-4]
    label_out_path = lip_label_SAVE_ROOT +label_save_video_name+'.json'

    npy_face_save_video_name = video.split('/')[-1][:-4]
    npy_out_path = lip_npy_SAVE_ROOT +label_save_video_name+'.npy'

    check_save_video_name = video.split('/')[-1][:-4]
    check_out_path = lip_mp4_SAVE_ROOT +check_save_video_name+'.mp4'
    
    if not os.path.exists(lip_label_SAVE_ROOT):
        os.makedirs(lip_label_SAVE_ROOT)
    if not os.path.exists(lip_npy_SAVE_ROOT):
        os.makedirs(lip_npy_SAVE_ROOT)
    if not os.path.exists(lip_mp4_SAVE_ROOT):
        os.makedirs(lip_mp4_SAVE_ROOT)

    start_time=time.time()
    reader = skvideo.io.FFmpegReader(video)
    video_shape = reader.getShape()
    (num_frames, h, w, c) = video_shape    
    # loop over frames from the video stream
    n_frame=0
    files=[]
    overlapped_list=[]
    lip_box = dict()
    lip_box['Lip_bounding_box']={}
    for frame in reader.nextFrame():     
        # frame = frame[:,3840//4:3840//4*3,:] 
        print(label_save_video_name, ' ',n_frame,' ', num_frames)
        ############################ face detect at first frame ############################
        if n_frame == 0:                   
            pred, probs = fa.get_landmarks(frame)
            if len(probs) > 1:
                for prob in probs:
                    overlapped_list.append(prob)
                min_index=overlapped_list.index(max(overlapped_list))
                pred=[pred[min_index]]
                overlapped_list=[]
            
            pred = np.squeeze(pred)
            x = pred[48:,0]
            y = pred[48:,1]
            min_x = min(x)-border_face
            min_y = min(y)-border_face
            max_x = max(x)+border_face
            max_y = max(y)+border_face
            if min_x < 0. :
                min_x = 0.
            if min_y < 0. :
                min_y = 0.
            height = int((max_y-min_y)/2)
            width = int((max_x-min_x)/2)
            standard=max(height,width)
            box = [int(min_x), int(min_y), int(max_x), int(max_y)]
            box = tuple(box)
            tracker = OPENCV_OBJECT_TRACKERS["medianflow"]()
            tracker.init(frame, box)
            prev_frame = frame
        else:
            (success, boxes) = tracker.update(frame)
            if success == False:
                boxes = previous_box
            box=[]
            for i in range(len(boxes)):
                box.append(int(boxes[i]))
            box=tuple(box)

            (success, boxes) = tracker.update(frame)
            

        (x, y, w, h) = [int(v) for v in box]        
        left_boundary=int((h+y)/2)-standard
        right_boundary=int((h+y)/2)+standard
        top_boundary=int((w+x)/2)-standard
        bottom_boundary=int((w+x)/2)+standard

        
        crop_img = frame[left_boundary:right_boundary,top_boundary:bottom_boundary]
        resized_crop_img=cv2.resize(crop_img, dsize=resize_face,interpolation=cv2.INTER_LINEAR)
        files.append(resized_crop_img)
        n_frame += 1
        previous_box = box

        face_box['Lip_bounding_box']['frame_'+str(n_frame-1)]={}
        face_box['Lip_bounding_box']['frame_'+str(n_frame-1)]['xtl']=left_boundary
        face_box['Lip_bounding_box']['frame_'+str(n_frame-1)]['ytl']=top_boundary
        face_box['Lip_bounding_box']['frame_'+str(n_frame-1)]['xbl']=right_boundary
        face_box['Lip_bounding_box']['frame_'+str(n_frame-1)]['ybl']=bottom_boundary
    
    print("mpg vs mpg_crop: {} vs {}".format(n_frame,len(files)))
    # pdb.set_trace()
    if num_frames == len(files):
        print("Good crop: ", video)
        f_c = open('/home/nas3/user/jungwook/Face_Lip_detection_tracking/Lip_crop_list.txt','a')
        f_c.write(video)
        f_c.write('\n')
        f_c.close()
    else:
        print("No crop: ", video)
        f_e = open('/home/nas3/user/jungwook/Face_Lip_detection_tracking/Lip_no_crop_list.txt','a')
        f_e.write(video)
        f_e.write('\n')
        f_e.close()
        
    

    with open(label_out_path, 'w', encoding='utf-8') as make_file:
        json.dump(face_box, make_file, indent="\t")
    
    np.save(npy_out_path,files)
    print("time: ",time.time()-start_time) 

    if face_mp4_SAVE_ROOT :
        out = cv2.VideoWriter(
                check_out_path,
                # cv2.VideoWriter_fourcc(*'DIVX'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                resize_face,
            ) 
        print("now starting to save cropped video")
        for k in range(len(files)):
            out.write(files[k])
        out.release()
        print(video, " saved")
    



def search(d_name,li,ext):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.mp4':
                li.append(
                        os.path.join(
                            os.path.join(
                                os.path.abspath(d_name),paths
                                ), 
                            filename
                            )
                        )

if __name__ == "__main__":
    
    READ_ROOT   =   '/home/nas/user/jungwook/Face_Lip_detection_tracking/NIA_full/'
    
    videos_list    =   []
    ext = '.mp4'
    # pdb.set_trace()
    search(READ_ROOT, videos_list,ext)
    videos_list = sorted(videos_list)

    #'./no_crop_list.txt' and 'crop_list.txt' 만들어주기
    f = open('/home/nas3/user/jungwook/Face_Lip_detection_tracking/Lip_no_crop_list.txt','w')
    f.close()

    f = open('/home/nas3/user/jungwook/Face_Lip_detection_tracking/Lip_crop_list.txt','w')
    f.close()
    start_time=time.time()
    procs = []
    for video in videos_list:
        proc = Process(target=main, args=(video,))
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()

    

    print("Total time: ",time.time()-start_time) 
    print("hi")
