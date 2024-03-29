import os
import numpy as np
import face_alignment   # FAN
from skimage import io
import argparse
import torch
import pickle

import cv2
import copy


device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)

def walk_dir(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            file_list.append((os.path.join(root, f)))
    return file_list

def get_args():
    parser = argparse.ArgumentParser(description="Create landmarks pkl file")
    parser.add_argument("--dataset_path", type=str, default='./dataset/hpdb/', help="path to BIWI dataset")
    parser.add_argument("--output_path", type=str, default='./data/', help='path to output landmakrs pkl file')
    parser.add_argument("--max_angle", type=int, default='90', help='largest pose angle')
    args = parser.parse_args()
    return args

def find_19_points_st_gcn(key_points):
    graph_points = []
    graph_points_list = [1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36,
                         39, 42, 45, 48, 51, 54]

    for i in range(len(key_points)):
        if i in graph_points_list:
            x = key_points[i][0]
            y = key_points[i][1]
            graph_points.append(x)
            graph_points.append(y)

    #print(graph_points)

    #norm_graph_points = style_19_points(graph_points)
    #print(norm_graph_points)

    data_numpy = np.zeros((2, 19))
    data_numpy[0, :] = graph_points[0::2]
    data_numpy[1, :] = graph_points[1::2]

    #print(data_numpy)

    return data_numpy

def style_19_points(key_points):
    norm_graph_points = []
    x = [x for (x, y) in key_points]
    y = [y for (x, y) in key_points]
    x_center = x[10]
    y_center = y[10]
    x_max    = max(x)
    y_max    = max(y)

    # normalization
    norm_x = [(x - x_center)/(x_max*1.0) for (x, y) in key_points]
    norm_y = [(y - y_center)/(y_max*1.0) for (x, y) in key_points]

    for i in range(len(norm_x)):
        norm_graph_points.append(norm_x[i])
        norm_graph_points.append(norm_y[i])

    return norm_graph_points

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
                data = np.transpose(data)

                # convert_to_angle
                roll = -np.arctan2(data[1][0], data[0][0]) * 180 / np.pi
                yaw = -np.arctan2(-data[2][0], np.sqrt(data[2][1] ** 2 + data[2][2] ** 2)) * 180 / np.pi
                pitch = np.arctan2(data[2][1], data[2][2]) * 180 / np.pi

                #print(f'roll:', {roll}, 'yaw:', {yaw}, 'pitch:', {pitch})
                id = int(image_id) + int(name) * 1000

                if -max_angle <= pitch <= max_angle and -max_angle <= yaw <= max_angle and -max_angle <= roll <= max_angle:
                    #detect face 
                    imagefile = ('/').join(file.split('/')[:-1]) + '/frame_' + image_id + '_rgb.png'
                    input = io.imread(imagefile)
                    key_points = fa.get_landmarks(input)
                    if isinstance(key_points, list):
                        if len(key_points) > 1:
                            face_id = 0
                            face_point = 0
                            for i, points in enumerate(key_points):
                                point = points[0][0]
                                if point > face_point:
                                    face_point = point
                                    face_id = i
                            key_point = key_points[face_id]
                        else:
                            key_point = key_points[0]
                    
                        if len(key_points) > 0:
                            samples[id] = find_19_points_st_gcn(key_point)
                    else:
                        print(f'instance: {name}, image_id: {image_id}')

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
#                 print(name)
                dir_path = os.path.join(root, name)
                samples = dump_pyr_xyc(dir_path, max_angle, name)
#                 print(len(samples.keys()))
                data.update(samples)

    output = open(output_path+'landmarks_all.pkl', 'wb')
    pickle.dump(data, output)
    output.close()
    print(f'save landmark file finished!!!')

if __name__ == '__main__':
    main()