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

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video_path", type=str, required=True,
	help="path to input video file")
ap.add_argument("-sv", "--save_video_path", type=str, required=True,
	help="path to output video file")
ap.add_argument("-m", "--mode", type=str, default="tracking",
        help="select detect, tracking or both", )
ap.add_argument("-t", "--tracker", type=str, default="medianflow",
	help="OpenCV object tracker type")
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


fps=30
size=(224,224)

video_list=[x for x in listdir(args["video_path"]) if ".mp4" in x]
video_list=sorted(video_list)
video_count=0

for video_name in video_list:

    try:
        video=args["video_path"]+str(video_name)
        out_path = args["save_video_path"] + str(video_name)   
    
        if os.path.exists(out_path):
            print("already existed: ", video_name)
            continue
        
        # reader = skvideo.io.FFmpegReader(video)
        # video_shape = reader.getShape()
        # (num_frames, h, w, c) = video_shape
        # print("input_video shape : ",num_frames, h, w, c)

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
                    bbox = []
                    pred, probs = fa.get_landmarks(frame)
                    # pdb.set_trace()
                    if len(probs) > 1:
                        for prob in probs:
                            overlapped_list.append(prob)
                        min_index=overlapped_list.index(max(overlapped_list))
                        pred=[pred[min_index]]
                        overlapped_list=[]
                    
                    pred = np.squeeze(pred)
                    x = pred[:,0]
                    y = pred[:,1]
                    min_x = min(x)
                    min_y = min(y)
                    max_x = max(x)
                    max_y = max(y)
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
        
            else:
                print("NotImplementMode Error")
                sys.exit()

            (x, y, w, h) = [int(v) for v in box]
            
            left_boundary=int((h+y)/2)-standard
            right_boundary=int((h+y)/2)+standard
            top_boundary=int((w+x)/2)-standard
            bottom_boundary=int((w+x)/2)+standard


            crop_img = frame[left_boundary:right_boundary,top_boundary:bottom_boundary]
            resized_crop_img=cv2.resize(crop_img, dsize=(224,224),interpolation=cv2.INTER_LINEAR)
            files.append(resized_crop_img) 
            n_frame += 1
        
           
        print("mpg vs mpg_crop: {} vs {}".format(count,len(files)))

        out = cv2.VideoWriter(
                out_path,
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
    except Exception as ex:
        print("ERROR:",ex,video_name)


        
