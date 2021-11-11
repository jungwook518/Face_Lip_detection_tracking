import face_alignment
import collections
import numpy as np
import argparse
import csv
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
from multiprocessing import Process, Pool, cpu_count
from tqdm import tqdm

def process(idx):
    try :
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
        video = videos_list[idx]     
        fa=face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu',flip_input=False, face_detector='sfd')
        fa_probs_threshold  = 0.95
        fps=30

        resize_face=(224,224)
        border_face = 100

        face_mp4_SAVE_ROOT   =   save_where+'Face_MP4_save_output/'
        face_label_SAVE_ROOT   =   save_where+'Face_label_save_output/'
        face_npy_SAVE_ROOT   =   save_where+'Face_npy_save_output/'

    
        label_save_video_name = video.split('/')[-1][:-4]
        label_out_path = face_label_SAVE_ROOT +label_save_video_name+'.json'

        npy_face_save_video_name = video.split('/')[-1][:-4]
        npy_out_path = face_npy_SAVE_ROOT +label_save_video_name+'.npy'

        check_save_video_name = video.split('/')[-1][:-4]
        check_out_path = face_mp4_SAVE_ROOT +check_save_video_name+'.mp4'
        
        if not os.path.exists(face_label_SAVE_ROOT):
            os.makedirs(face_label_SAVE_ROOT)
        if not os.path.exists(face_npy_SAVE_ROOT):
            os.makedirs(face_npy_SAVE_ROOT)
        if not os.path.exists(face_mp4_SAVE_ROOT):
            os.makedirs(face_mp4_SAVE_ROOT)

        start_time=time.time()
        reader = skvideo.io.FFmpegReader(video)
        video_shape = reader.getShape()
        (num_frames, h, w, c) = video_shape    

        n_frame=0
        files=[]
        overlapped_list=[]
        face_box = dict()
        face_box['Face_bounding_box']={}
        face_box['Face_bounding_box']['xtl_ytl_xbr_ybr']=[]
        for frame in reader.nextFrame():     
            if n_frame % 1000 ==0:
                print(label_save_video_name, ' ',n_frame,' ', num_frames)
            if n_frame == 0:                   
                pred, probs = fa.get_landmarks(frame)
                if len(probs) > 1:
                    for prob in probs:
                        overlapped_list.append(prob)
                    min_index=overlapped_list.index(max(overlapped_list))
                    pred=[pred[min_index]]
                    overlapped_list=[]
                
                pred = np.squeeze(pred)
                x = pred[:,0]
                y = pred[:,1]
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

                
            (x, y, w, h) = [int(v) for v in box]        
            left_boundary=int((h+y)/2)-standard
            if left_boundary <0:
                left_boundary=0
            right_boundary=int((h+y)/2)+standard
            top_boundary=int((w+x)/2)-standard
            if top_boundary <0:
                top_boundary=0
            bottom_boundary=int((w+x)/2)+standard

            crop_img = frame[left_boundary:right_boundary,top_boundary:bottom_boundary]
            resized_crop_img=cv2.resize(crop_img, dsize=resize_face,interpolation=cv2.INTER_LINEAR)
            files.append(resized_crop_img)
            n_frame += 1

            face_box['Face_bounding_box']['xtl_ytl_xbr_ybr'].append([left_boundary,top_boundary,right_boundary,bottom_boundary])
        
        print("mpg vs mpg_crop: {} vs {}".format(n_frame,len(files)))

        if num_frames == len(files):
            print("Good crop: ", video)
            npy_out_path = face_npy_SAVE_ROOT +label_save_video_name+'.npy'
            np.save(npy_out_path,files)
            with open(label_out_path, 'w', encoding='utf-8') as make_file:
                json.dump(face_box, make_file, indent="\t")
                
            f_c = open(save_where+'Face_crop_list.txt','a')
            f_c.write(video)
            f_c.write('\n')
            f_c.close()
            out = cv2.VideoWriter(
                    check_out_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    resize_face,
                ) 
            print("now starting to save cropped video")
            for k in range(len(files)):
                out.write(files[k])
            out.release()
            print(video, " saved")

        else:
            print("No crop: ", video)
            f_e = open(save_where+'Face_no_crop_list.txt','a')
            f_e.write(video)
            f_e.write('\n')
            f_e.close()
        
        print("time: ",time.time()-start_time) 

    except:
        print("error!  ", video)
        f_e = open(save_where+'Face_no_crop_list.txt','a')
        f_e.write(video)
        f_e.write('\n')
        f_e.close()


def search(d_name,li,ext1):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == ext1 or ext=='.mp4':
                li.append(
                        os.path.join(
                            os.path.join(
                                os.path.abspath(d_name),paths
                                ), 
                            filename
                            )
                        )

READ_ROOT   =   '/home/nia-jungwook/210907~0909_31명/'
read_folder = READ_ROOT.split('/')[-2]
save_where = '/home/nas4/user/jungwook/ubuntu/'+read_folder +'/'
if not os.path.exists(save_where):
    os.makedirs(save_where)

videos_list = []
ext = '.MP4'
search(READ_ROOT, videos_list,ext)
videos_list = sorted(videos_list)
pdb.set_trace()

yes_crop_file = save_where+'Face_crop_list.txt'


if os.path.isfile(yes_crop_file):
    datalist = open(yes_crop_file,'r')
    datalist = list(csv.reader(datalist))
    datalist = [f[0] for f in datalist]

    for i in range(len(datalist)):
        try:
            videos_list.remove(datalist[i])
        except:
            pass


if __name__ == "__main__":
    pdb.set_trace()
    f = open(save_where+'Face_no_crop_list.txt','a')
    f.close()

    f = open(save_where+'Face_crop_list.txt','a')
    f.close()

    for i in range(len(videos_list)):
        process(i)