import os
import numpy as np
# import face_alignment
from skimage import io
import argparse
import torch

import cv2
import copy
import pickle


def walk_dir(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            file_list.append((os.path.join(root, f)))
    return file_list

def get_args():
    parser = argparse.ArgumentParser(description="Create head pose pkl file")
    parser.add_argument("--dataset_path", type=str, default='./dataset/Head_Pose_Database_UPNA', help="path to BIWI dataset")
    parser.add_argument("--output_path", type=str, default='./data/', help='path to output head pose pkl file')
    parser.add_argument("--max_angle", type=int, default='90', help='largest pose angle')
    args = parser.parse_args()
    return args

def dump_pyr_xyc(pose_path, max_angle, name):

    file_list = []
    file_list = walk_dir(pose_path)
    samples = {}
    max_ang = float('-inf')
    min_ang = float('inf')

    # for file in sorted(file_list):
    #     if file.endswith('.txt'):
    #         image_id = file.split('/')[-1].split('.')[0].split('_')[1]
    #         #print(image_id)
    #         # for removing no-face frames
    #         if int(name) == 6 and 193 <= int(image_id) <= 220 and int(image_id) == 389:  continue    #for instance 6 
    #         elif int(name) == 18 and 472 <= int(image_id) <= 478:  continue    #for instance 18
    #         elif int(name) == 21 and 218 <= int(image_id) <= 223 and 453 <= int(image_id) <= 458:  continue    #for instance 21
    #         elif int(name) == 22 and 224 <= int(image_id) <= 239:  continue    #for instance 22
    #         elif int(name) == 23 and 221 <= int(image_id) <= 234:  continue    #for instance 23
    #         elif int(name) == 24 and 141 <= int(image_id) <= 155:  continue    #for instance 24
    #         else:
    #             data = np.genfromtxt(file, skip_footer=0)[:-1]
    #             data = np.transpose(data)

    #             # convert_to_angle
    #             roll = -np.arctan2(data[1][0], data[0][0]) * 180 / np.pi
    #             yaw = -np.arctan2(-data[2][0], np.sqrt(data[2][1] ** 2 + data[2][2] ** 2)) * 180 / np.pi
    #             pitch = np.arctan2(data[2][1], data[2][2]) * 180 / np.pi
    for file in sorted(file_list):
        print(file)
        if file.endswith('.jpg'):

            image_name = file.split('/')[-1]

            user_index = int(image_name.split("_")[1])
            video_index = int(image_name.split("_")[3])
            frame_index = int(image_name.split("_")[-1].split(".")[0])

            pose_file_name = "_".join(image_name.split("_")[:-1])+"_groundtruth3D.txt"
            pose_file = os.path.join("/".join(file.split('/')[:-1]), pose_file_name)

            pose_data = np.genfromtxt(pose_file, skip_footer=0)
            print("pose_data:", pose_data.shape)

            roll = pose_data[frame_index-1][-3]
            yaw = pose_data[frame_index-1][-2]
            pitch = pose_data[frame_index-1][-1]
            print(f'roll:', {roll}, 'yaw:', {yaw}, 'pitch:', {pitch})

            id = frame_index + int(video_index-1) * 301 + int(user_index) * 10000
            print("id:", id)
            data = np.zeros((3,19))

            if max(pitch, yaw, roll) > max_ang:
                max_ang = max(pitch, yaw, roll)
            elif min(pitch, yaw, roll) < min_ang:
                min_ang = min(pitch, yaw, roll)

            if -max_angle <= pitch <= max_angle and -max_angle <= yaw <= max_angle and -max_angle <= roll <= max_angle:
                data[0,:] = yaw
                data[1,:] = pitch
                data[2,:] = roll 
                samples[id] = data
    
    return samples, max_ang, min_ang

def main():

    args = get_args()
    dataset_path = args.dataset_path
    output_path = args.output_path
    max_angle = args.max_angle
    data = dict()

    for root, dirs, files in os.walk(dataset_path, topdown=True):
        for name in sorted(dirs):
            folder_index = int(name.split("_")[-1])
            print("folder_index:", folder_index)
            # if int(folder_index) == 1:
            dir_path = os.path.join(root, name)
            samples, max_ang, min_ang = dump_pyr_xyc(dir_path, max_angle, name)
#                 print(len(samples.keys()))
#                 print(f'max_angle:{max_ang}, min_angle:{min_ang}')

            data.update(samples)
    
#     print(len(data.keys()))

    output = open(output_path+'UPNA_YPR_all.pkl', 'wb')
    pickle.dump(data, output)
    output.close()

    print('save ypr file finished!!!')

if __name__ == '__main__':
    main()