import face_alignment
import collections
import numpy as np
from imutils.video import VideoStream
import argparse
import imutils
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




os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video_path", type=str, required=True, help="path to input video file")
ap.add_argument("-sl", "--save_label_path", type=str, required=True, help="path to output video file")

ap.add_argument("-sv", "--save_video_path", type=str, required=True, help="path to output video file")
ap.add_argument("-cv", "--check_video", type=bool, required=True, help="check to save video file, True, False")
ap.add_argument("-m", "--mode", type=str,required=True, help="tracking or detection", )
ap.add_argument("-t", "--tracker", type=str, default="medianflow", help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
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
                        

fa=face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda',flip_input=False, face_detector='sfd')

fa_probs_threshold  = 0.95

# initialize OpenCV's special multi-object tracker
#trackers            = cv2.MultiTracker_create()


fps=25
size=(88,88)
border = 0



video_list=[x for x in listdir(args["video_path"]) if ".avi" in x]
video_list=sorted(video_list)
video_count=0
f = open("./no_process_videos.txt", 'w')
for video_name in video_list:
    try:
        video=args["video_path"]+str(video_name)
        out_path = args["save_video_path"] + '/Lip_'+str(video_name)   
        label_out_path = args["save_label_path"] +'/Lip_'+ str(video_name)[:-4]+'.json'
        
        if not os.path.exists(args["save_video_path"]):
            os.makedirs(args["save_video_path"])
        if not os.path.exists(args["save_label_path"]):
            os.makedirs(args["save_label_path"])
        
        reader = skvideo.io.FFmpegReader(video)
        video_shape = reader.getShape()
        (num_frames, h, w, c) = video_shape


        start_time=time.time()
        vs = cv2.VideoCapture(video)
        
        # loop over frames from the video stream
        n_frame=0
        files=[]
        count=0
        video_count+=1
        overlapped_list=[]
        
        while True:        
            hasFrame, frame = vs.read()
            if not hasFrame:
                break
            count+=1      
            if args["mode"] == "tracking":
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
                    min_x = min(x)-border
                    min_y = min(y)-border
                    max_x = max(x)+border
                    max_y = max(y)+border
                    if min_x < 0. :
                        min_x = 0.
                    if min_y < 0. :
                        min_y = 0.
                    height = int((max_y-min_y)/2)
                    width = int((max_x-min_x)/2)
                    standard=max(height,width)
                    box = [int(min_x), int(min_y), int(max_x), int(max_y)]
                    box = tuple(box)
                    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                    tracker.init(frame, box)

                else:
                    (success, boxes) = tracker.update(frame)
                    box=[]
                    for i in range(len(boxes)):
                        box.append(int(boxes[i]))
                    box=tuple(box)

            if args["mode"] == "detection":       
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
                min_x = min(x)-border
                min_y = min(y)-border
                max_x = max(x)+border
                max_y = max(y)+border
                if min_x < 0. :
                    min_x = 0.
                if min_y < 0. :
                    min_y = 0.
                height = int((max_y-min_y)/2)
                width = int((max_x-min_x)/2)
                standard=max(height,width)
                box = [int(min_x), int(min_y), int(max_x), int(max_y)]
                box = tuple(box)


            (x, y, w, h) = [int(v) for v in box]
            
            left_boundary=int((h+y)/2)-standard
            right_boundary=int((h+y)/2)+standard
            top_boundary=int((w+x)/2)-standard
            bottom_boundary=int((w+x)/2)+standard

            
            crop_img = frame[left_boundary:right_boundary,top_boundary:bottom_boundary]
            resized_crop_img=cv2.resize(crop_img, dsize=size,interpolation=cv2.INTER_LINEAR)
            files.append(resized_crop_img)
            n_frame += 1
        
           
        print("mpg vs mpg_crop: {} vs {}".format(count,len(files)))
        
        if num_frames == len(files):
            print("Good crop: ", video)
            lip_box = dict()
            lip_box['Lip_bounding_box']={}
            for i in range(num_frames):
                lip_box['Lip_bounding_box']['frame_'+str(i)]={}
                lip_box['Lip_bounding_box']['frame_'+str(i)]['xtl']=left_boundary
                lip_box['Lip_bounding_box']['frame_'+str(i)]['ytl']=top_boundary
                lip_box['Lip_bounding_box']['frame_'+str(i)]['xbl']=right_boundary
                lip_box['Lip_bounding_box']['frame_'+str(i)]['ybl']=bottom_boundary

        else:
            print("No crop: ", video)
            f.write(video)
            f.write('\n')
            continue
        
        with open(label_out_path, 'w', encoding='utf-8') as make_file:
            json.dump(lip_box, make_file, indent="\t")
        
        
        if args["check_video"] == True:
            out = cv2.VideoWriter(
                    out_path,
                    #cv2.VideoWriter_fourcc(*'DIVX'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    size,
                ) 
            print("now starting to save cropped video")
            for k in range(len(files)):
                out.write(files[k])
            out.release()
        
            vs.release()
            print(video_name, " saved",video_count)

        print("time: ",time.time()-start_time) 
        f.close()
    except Exception as ex:
        print("ERROR:",ex,video_name)


        
