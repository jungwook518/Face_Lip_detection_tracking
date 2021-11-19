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
    # border_face = 10

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
            # border_lip = int(max(max(x) - min(x), max(y) - min(y))*0.5)
            # min_x = min(x)-border_lip
            # min_y = min(y)-border_lip
            # max_x = max(x)+border_lip
            # max_y = max(y)+border_lip
            min_x = min(x)
            min_y = min(y)
            max_x = max(x)
            max_y = max(y)
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

        border_face = int(max(right_boundary-left_boundary,bottom_boundary-top_boundary)*0.5)

        left_boundary -=border_face
        right_boundary+=border_face
        top_boundary-=border_face
        bottom_boundary+=border_face
        if left_boundary < 0 :
            left_boundary = 0
        if top_boundary < 0 :
            top_boundary = 0

        crop_img = frame[left_boundary:right_boundary,top_boundary:bottom_boundary]
        resized_crop_img=cv2.resize(crop_img, dsize=resize_face,interpolation=cv2.INTER_LINEAR)
        files.append(resized_crop_img)
        n_frame += 1

        face_box['Face_bounding_box']['xtl_ytl_xbr_ybr'].append([left_boundary,top_boundary,right_boundary,bottom_boundary])
    

    if num_frames == n_frame:
        print("Good crop: ", video)
        np.save(npy_out_path,files)
        with open(label_out_path, 'w', encoding='utf-8') as make_file:
            json.dump(face_box, make_file, indent="\t")
            
        
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

        f_c = open(save_where+'Face_crop_list.txt','a')
        f_c.write(video)
        f_c.write('\n')
        f_c.close()

    else:
        print("No crop: ", video)
        f_e = open(save_where+'Face_no_crop_list.txt','a')
        f_e.write(video)
        f_e.write('\n')
        f_e.close()
    
    print("time: ",time.time()-start_time) 

    # except:
    #     print("error!  ", video)
    #     f_e = open(save_where+'Face_no_crop_list.txt','a')
    #     f_e.write(video)
    #     f_e.write('\n')
    #     f_e.close()


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

READ_ROOT   =   '/home/nia-jungwook/210907~0909_31명/' ##### 읽을 파일 폴더 위치 #########
read_folder = READ_ROOT.split('/')[-2]
save_where = '/home/nas4/user/jungwook/ubuntu/'+read_folder +'/'  #####처리한 파일 저장 위치 #######
if not os.path.exists(save_where):
    os.makedirs(save_where)

videos_list = []
ext = '.MP4'
search(READ_ROOT, videos_list,ext)
videos_list = sorted(videos_list)


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