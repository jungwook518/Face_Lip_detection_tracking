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
        video = videos_list[idx]
        face_label = face_list[idx]
        lip_label = lip_list[idx]
        fps=30
        resize_face=(224,224)

        visual_mp4_SAVE_ROOT   =   save_where+'Visual_MP4_save_output/'
    
        check_save_video_name = video.split('/')[-1][:-4]
        check_out_path = visual_mp4_SAVE_ROOT +check_save_video_name+'.mp4'
        
        if not os.path.exists(visual_mp4_SAVE_ROOT):
            os.makedirs(visual_mp4_SAVE_ROOT)
        files=[]
        start_time=time.time()
        reader = skvideo.io.FFmpegReader(video)
        video_shape = reader.getShape()
        (num_frames, h, w, c) = video_shape    

        with open(face_label , 'r') as f:
            face_datas = json.load(f)
        with open(lip_label , 'r') as f:
            lip_datas = json.load(f)

        

        n_frame=0
        for frame in reader.nextFrame(): 
            print(n_frame)
            face_xtl = face_datas['Face_bounding_box']['xtl_ytl_xbr_ybr'][n_frame][0]
            face_xbr = face_datas['Face_bounding_box']['xtl_ytl_xbr_ybr'][n_frame][2]
            face_ytl = face_datas['Face_bounding_box']['xtl_ytl_xbr_ybr'][n_frame][1]
            face_ybr = face_datas['Face_bounding_box']['xtl_ytl_xbr_ybr'][n_frame][3]

            lip_xtl = lip_datas['Lip_bounding_box']['xtl_ytl_xbr_ybr'][n_frame][0]
            lip_xbr = lip_datas['Lip_bounding_box']['xtl_ytl_xbr_ybr'][n_frame][2]
            lip_ytl = lip_datas['Lip_bounding_box']['xtl_ytl_xbr_ybr'][n_frame][1]
            lip_ybr = lip_datas['Lip_bounding_box']['xtl_ytl_xbr_ybr'][n_frame][3]
            
            point1 = (face_ytl,face_xtl)
            point2 = (face_ybr,face_xbr)

            point3 = (lip_ytl,lip_xtl)
            point4 = (lip_ybr,lip_xbr)

            thickness = 15
            color_face = (0,0,255)
            color_lip = (255,0,0)

            face_rec = cv2.rectangle(frame, point1, point2, color_face, thickness)
            face_lip_rec = cv2.rectangle(face_rec, point3, point4, color_lip, thickness)
            # pdb.set_trace()
            resized_crop_img=cv2.resize(face_lip_rec, dsize=resize_face,interpolation=cv2.INTER_LINEAR)
            resized_crop_img = np.flip(resized_crop_img, axis = 2)

            files.append(resized_crop_img)
            n_frame += 1

        if num_frames == len(files):
            print("Good crop: ", video)
            out = cv2.VideoWriter(check_out_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,resize_face)
             
            print("now starting to save cropped video")
            for k in range(len(files)):
                out.write(files[k])
            out.release()
            print(video, " saved")

            f_c = open(save_where+'Visual_crop_list.txt','a')
            f_c.write(video)
            f_c.write('\n')
            f_c.close()
        else:
            print("No crop: ", video)
            f_e = open(save_where+'Visual_no_crop_list.txt','a')
            f_e.write(video)
            f_e.write('\n')
            f_e.close()
        
        print("time: ",time.time()-start_time) 

    except:
        pdb.set_trace()
        print("error!  ", video)
        f_e = open(save_where+'Visual_no_crop_list.txt','a')
        f_e.write(video)
        f_e.write('\n')
        f_e.close()


def search(d_name,li,ext1):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == ext1:
                li.append(
                        os.path.join(
                            os.path.join(
                                os.path.abspath(d_name),paths
                                ), 
                            filename
                            )
                        )

def search_mp4(d_name,li,ext1):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == ext1 or ext == '.mp4':
                li.append(
                        os.path.join(
                            os.path.join(
                                os.path.abspath(d_name),paths
                                ), 
                            filename
                            )
                        )



READ_ROOT_VIDEO   =   '/home/jungwook/Face_Lip_detection_tracking/temp_data/'
save_label = '/home/jungwook/Face_Lip_detection_tracking/temp/'
READ_ROOT_FACE   =   save_label+'Face_label_save_output/'
READ_ROOT_LIP   =   save_label+'Lip_label_save_output/'

save_where = '/home/jungwook/Face_Lip_detection_tracking/temp/'

videos_list =[]
face_list = []
lip_list = []
ext = '.json'
ext_video = '.MP4'
search_mp4(READ_ROOT_VIDEO, videos_list,ext_video)
search(READ_ROOT_FACE, face_list,ext)
search(READ_ROOT_LIP, lip_list,ext)
videos_list = sorted(videos_list)
face_list = sorted(face_list)
lip_list = sorted(lip_list)

if len(face_list) != len(lip_list) or len(face_list) != len(videos_list) or len(lip_list) != len(videos_list) :
    print("Not match")
    pdb.set_trace()
    sys.exit()
    
yes_crop_file = save_where+'Visual_crop_list.txt'

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
    f = open(save_where+'Visual_no_crop_list.txt','a')
    f.close()

    f = open(save_where+'Visual_crop_list.txt','a')
    f.close()

    for i in range(len(videos_list)):
        process(i)