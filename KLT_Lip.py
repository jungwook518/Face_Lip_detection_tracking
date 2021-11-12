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
from applyGeometricTransformation import *
from estimateAllTranslation import *
from estimateFeatureTranslation import *
from getFeatures import *


def process(idx):
    try:
        video = videos_list[idx]     
        fa=face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu',flip_input=False, face_detector='sfd')
        fa_probs_threshold  = 0.95
        fps=30

        resize_lip=(88,88)
        # border_lip = 100

        lip_mp4_SAVE_ROOT   =   save_where+'Lip_MP4_save_output/'
        lip_label_SAVE_ROOT   =   save_where+'Lip_label_save_output/'
        lip_npy_SAVE_ROOT   =   save_where+'Lip_npy_save_output/'

        
        label_save_video_name = video.split('/')[-1][:-4]
        label_out_path = lip_label_SAVE_ROOT +label_save_video_name+'.json'

        # npy_lip_save_video_name = video.split('/')[-1][:-4]
        # npy_out_path = lip_npy_SAVE_ROOT +npy_lip_save_video_name+'.npy'

        check_save_video_name = video.split('/')[-1][:-4]
        check_out_path = lip_mp4_SAVE_ROOT +check_save_video_name+'.mp4'
        
        if not os.path.exists(lip_label_SAVE_ROOT):
            os.makedirs(lip_label_SAVE_ROOT)
        # if not os.path.exists(lip_npy_SAVE_ROOT):
        #     os.makedirs(lip_npy_SAVE_ROOT)
        if not os.path.exists(lip_mp4_SAVE_ROOT):
            os.makedirs(lip_mp4_SAVE_ROOT)

        start_time=time.time()
        reader = skvideo.io.FFmpegReader(video)
        video_shape = reader.getShape()
        (num_frames, h, w, c) = video_shape    

        n_frame=0
        files=[]
        overlapped_list=[]
        lip_box = dict()
        lip_box['Lip_bounding_box']={}
        lip_box['Lip_bounding_box']['xtl_ytl_xbr_ybr']=[]

        for frame in reader.nextFrame():    
            if n_frame % 1 ==0:
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
                border_lip = int(max(right_boundary-left_boundary,bottom_boundary-top_boundary)*0.5)

                left_boundary -=border_lip
                right_boundary+=border_lip
                top_boundary-=border_lip
                bottom_boundary+=border_lip
                if left_boundary < 0 :
                    left_boundary = 0
                if top_boundary < 0 :
                    top_boundary = 0
                
                bbox = np.array([[[top_boundary,left_boundary],[bottom_boundary,left_boundary],[top_boundary,right_boundary],[bottom_boundary,right_boundary]]]).astype(float)
                startXs,startYs = getFeatures(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),bbox,use_shi=False)
                
                crop_img = frame[left_boundary:right_boundary,top_boundary:bottom_boundary]
                resized_crop_img=cv2.resize(crop_img, dsize=resize_lip,interpolation=cv2.INTER_LINEAR)
                files.append(resized_crop_img)
                
                n_frame += 1
                prev_frame = frame
                previous_bbox = bbox
                lip_box['Lip_bounding_box']['xtl_ytl_xbr_ybr'].append([left_boundary,top_boundary,right_boundary,bottom_boundary])
            else:

                newXs, newYs = estimateAllTranslation(startXs, startYs, prev_frame, frame)
                Xs, Ys ,bbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, previous_bbox)
                startXs = Xs; startYs = Ys
                n_features_left = np.sum(Xs!=-1)
                if n_features_left < 15:
                    bbox = previous_bbox
                    startXs,startYs = getFeatures(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),bbox)

                top_boundary = int(bbox[0][0][0])
                bottom_boundary = int(bbox[0][1][0])
                left_boundary = int(bbox[0][0][1])
                right_boundary = int(bbox[0][2][1])

                crop_img = frame[left_boundary:right_boundary,top_boundary:bottom_boundary]
                resized_crop_img=cv2.resize(crop_img, dsize=resize_lip,interpolation=cv2.INTER_LINEAR)
                files.append(resized_crop_img)
                n_frame += 1
                prev_frame = frame
                previous_bbox = bbox
                previous_startXs = startXs
                previous_startYs = startYs
                lip_box['Lip_bounding_box']['xtl_ytl_xbr_ybr'].append([left_boundary,top_boundary,right_boundary,bottom_boundary])
            #pdb.set_trace()
        print("mpg vs mpg_crop: {} vs {}".format(n_frame,len(files)))

        if num_frames == len(files):
            print("Good crop: ", video)
            with open(label_out_path, 'w', encoding='utf-8') as make_file:
                json.dump(lip_box, make_file, indent="\t")
                
            out = cv2.VideoWriter(check_out_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,resize_lip) 
            print("now starting to save cropped video")
            for k in range(len(files)):out.write(files[k])
            out.release()
            print(video, " saved")

            f_c = open(save_where+'Lip_crop_list.txt','a')
            f_c.write(video)
            f_c.write('\n')
            f_c.close()

        else:
            print("No crop: ", video)
            f_e = open(save_where+'Lip_no_crop_list.txt','a')
            f_e.write(video)
            f_e.write('\n')
            f_e.close()
        
        print("time: ",time.time()-start_time) 
    except:
        print("error!  ", video)
        f_e = open(save_where+'Lip_no_crop_list.txt','a')
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

READ_ROOT   =   '/home/nia-jungwook/210907~0909_31명/'  ##### 읽을 파일 폴더 위치 #########
read_folder = READ_ROOT.split('/')[-2]
save_where = '/home/nas4/user/jungwook/ubuntu/'+read_folder +'/'   #####처리한 파일 저장 위치 #######
if not os.path.exists(save_where):
    os.makedirs(save_where)

videos_list = []
ext = '.MP4'
search(READ_ROOT, videos_list,ext)
videos_list = sorted(videos_list)
pdb.set_trace()

yes_crop_file = save_where+'Lip_crop_list.txt'

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
    # pdb.set_trace()
    f = open(save_where+'Lip_no_crop_list.txt','a')
    f.close()

    f = open(save_where+'Lip_crop_list.txt','a')
    f.close()

    for i in range(len(videos_list)):
        process(i)

