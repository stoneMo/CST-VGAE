import os
import numpy as np
import face_alignment
from skimage import io
import argparse
import torch
import pickle

import cv2
import copy

from src import model
from src import util
from src.body import Body

body_estimation = Body('./model/body_pose_model.pth')  # openpose

def walk_dir(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            file_list.append((os.path.join(root, f)))
    return file_list

def get_args():
    parser = argparse.ArgumentParser(description="Create gazes pkl file")
    parser.add_argument("--dataset_path", type=str, default='./dataset/hpdb/', help="path to BIWI dataset")
    parser.add_argument("--output_path", type=str, default='./data/', help='path to output gazes pkl file')
    parser.add_argument("--max_angle", type=int, default='90', help='largest pose angle')
    args = parser.parse_args()
    return args

def find_pupil_points(candidate, pupil_set):
    
    data_numpy = np.zeros((2, 2))

    #pupil points
    for i in range(2):
        if pupil_set[14+i] != -1:
            for candid in candidate:
                if candid[-1] == pupil_set[14+i]:
                    data_numpy[0, i] = candid[0]
                    data_numpy[1, i] = candid[1]
        else:
            for candid in candidate:
                if candid[-1] == pupil_set[0]:
                    data_numpy[0, i] = candid[0]
            if i == 0:
                for candid in candidate:
                    if candid[-1] == pupil_set[15+i]:
                        data_numpy[1, i] = candid[1]
            if i == 1:
                for candid in candidate:
                    if candid[-1] == pupil_set[13+i]:
                        data_numpy[1, i] = candid[1]
                        
    #print(data_numpy)

    return data_numpy

def dump_pyr_xyc(pose_path, max_angle, name):

    file_list = []
    file_list = walk_dir(pose_path)
    size = 5
    sample_list = []
    label_list = []
    samples = {}

    for file in sorted(file_list):
        if file.endswith('.txt'):
            image_id = file.split('/')[-1].split('.')[0].split('_')[1]
            #print(image_id)
            # for removing no-face frames
            if int(name) == 6 and 193 <= int(image_id) <= 220 and int(image_id) == 389:  continue    #for instance 6 
            elif int(name) == 18 and 472 <= int(image_id) <= 478:  continue    #for instance 18
            elif int(name) == 21 and 218 <= int(image_id) <= 223 and 453 <= int(image_id) <= 458:  continue    #for instance 21
            elif int(name) == 22 and 224 <= int(image_id) <= 239:  continue    #for instance 22
            elif int(name) == 23 and 221 <= int(image_id) <= 234:  continue    #for instance 23
            elif int(name) == 24 and 141 <= int(image_id) <= 155:  continue    #for instance 24
            else:
                data = np.genfromtxt(file, skip_footer=0)[:-1]

                # convert_to_angle
                roll = -np.arctan2(data[1][0], data[0][0]) * 180 / np.pi
                yaw = -np.arctan2(-data[2][0], np.sqrt(data[2][1] ** 2 + data[2][2] ** 2)) * 180 / np.pi
                pitch = np.arctan2(data[2][1], data[2][2]) * 180 / np.pi

                #print(f'roll:', {roll}, 'yaw:', {yaw}, 'pitch:', {pitch})
                id = int(image_id) + int(name) * 1000

                if -max_angle <= pitch <= max_angle and -max_angle <= yaw <= max_angle and -max_angle <= roll <= max_angle:
                    
                    imagefile = ('/').join(file.split('/')[:-1]) + '/frame_' + image_id + '_rgb.png'
                    # detect pupil
                    oriImg = cv2.imread(imagefile)  # B,G,R order
                    #print(image_id)
                    candidate, subset = body_estimation(oriImg)
                    #print(f'candidate:{candidate}')
                    pupil_index = 0               
                    if len(subset) == 1: 
                        pupil_set = subset[0]
                    else:
                        for pupilset in subset:
                            if pupilset[0] > pupil_index:
                                pupil_index = pupilset[0]
                                pupil_set = pupilset

                    samples[id] = find_pupil_points(candidate, pupil_set)

                    if 0. in samples[id]:
#                         print(f'name: {name}, image_id_0: {image_id}')
#                         print(candidate)
#                         print(subset)
#                         print(samples[id])
                        del samples[id]
                        

    return samples

def main():
    args = get_args()
    dataset_path = args.dataset_path
    output_path = args.output_path
    max_angle = args.max_angle
    data = dict()

    for root, dirs, files in os.walk(dataset_path, topdown=True):
        for name in sorted(dirs):
            if int(name) > 0:
                print(name)
                dir_path = os.path.join(root, name)
                samples = dump_pyr_xyc(dir_path, max_angle, name)
                print(len(samples.keys()))
                data.update(samples)
            
    output = open(output_path+'gaze_all.pkl', 'wb')
    pickle.dump(data, output)
    output.close()
    print(f'save gaze file finished!!!')    

if __name__ == '__main__':
    main()