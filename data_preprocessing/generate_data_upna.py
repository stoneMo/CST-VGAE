import numpy as np
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="Create training dataset")
parser.add_argument("--predict_frames", type=str, default='1', help="forecasted frames")
parser.add_argument("--landmark_path", type=str, default='./data/UPNA_landmarks_all.pkl', help="path to landmarks file")
parser.add_argument("--gaze_path", type=str, default='./data/UPNA_gaze_all.pkl', help='path to gaze file')
parser.add_argument("--YawPR_path", type=str, default='./data/UPNA_YPR_all.pkl', help='path to YawPR file')
parser.add_argument("--Gaze", type=int, default=1, help='conditioned on Gaze prior')

args = parser.parse_args()

landmark_file = open(args.landmark_path, 'rb')
landmark_dict = pickle.load(landmark_file)
landmark_file.close()

gaze_file = open(args.gaze_path, 'rb')
gaze_dict = pickle.load(gaze_file)
gaze_file.close()

YawPR_file = open(args.YawPR_path, 'rb')
YawPR_dict = pickle.load(YawPR_file)
YawPR_file.close()

def normalize_points(key_points):
    # key_points (2, 19)  or (2, 21)
    
    x = key_points[0]
    y = key_points[1]
    x_center = x[10]
    y_center = y[10]
    x_max    = max(x)
    y_max    = max(y)

    # normalization
    norm_x = [(x - x_center)/(x_max*1.0) for x in x]
    norm_y = [(y - y_center)/(y_max*1.0) for y in y]

    norm_graph_points = [norm_x, norm_y]
    norm_graph_points = np.asarray(norm_graph_points)

    return norm_graph_points

train_data_list = []
train_label_list = []
valid_data_list = []
valid_label_list = []

data_frames = 5
predict_frames = int(args.predict_frames)

gaze_prior = True if args.Gaze == 1 else False

gaze_before = 5                               #gaze happens before the forecasted frame 
YawPitchRoll_after = 5                        #YawPitchRoll happens after the current frame 

image_id_list = []
gaze_id_list = []
YawPitchRoll_id_list = []
label_id_list = []


for i in range(1, 11):
    print('instance: ', i)
    for j in range(data_frames): 
        min_frame = i * 10000 + j
        max_frame = (i+1) * 10000 - 1
        delta = (max_frame - min_frame) % data_frames
        index_arr = np.arange(min_frame, max_frame-delta).reshape((-1,data_frames))

        for id in index_arr:
            print(f'id: {id}')
            image_list = [image_id for image_id in id if image_id in landmark_dict]
            print(f'image_list: {image_list}')
            label_id = [id[-1]+m for m in range(1, predict_frames+1) if id[-1]+m in YawPR_dict \
                                                                    and id[-1]+m in landmark_dict]
            print(f'label_id: {label_id}')
            if int(args.predict_frames) < 5:
                gaze_list = image_list
                gaze_predict_frames = 5
            else:
                gaze_list = [image_id-gaze_before for image_id in label_id if image_id-gaze_before in gaze_dict]
                gaze_predict_frames = predict_frames
            print(f'gaze_list: {gaze_list}')
            YawPitchRoll_list = [image_id for image_id in label_id if image_id in YawPR_dict]
            label_gaze_list = [image_id for image_id in label_id if image_id in gaze_dict]

            #for gaze_prior 
            if gaze_prior and len(image_list) == data_frames\
                and len(gaze_list) == gaze_predict_frames\
                and len(label_id) == predict_frames and len(label_gaze_list) == predict_frames:

                data_frame = [landmark_dict[id] for id in image_list]    #[(2, 19), (2, 19), (2, 19), (2, 19), (2, 19)]
                print("data_frame:", data_frame[0].shape)
                data_gaze = [gaze_dict[id] for id in gaze_list]      #[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
                print("data_gaze:", data_gaze[0].shape)
                data_YPR = [np.concatenate((YawPR_dict[id][:3], YawPR_dict[id][:3,:2]), axis=-1) for id in image_list]   #[(2, 21), (2, 21), (2, 21), (2, 19), (2, 19)]
                print("data_YPR:", data_YPR[0].shape)

                label_land = [landmark_dict[id] for id in label_id]          #[(2, 19), (2, 19), (2, 19), (2, 19), (2, 19)]          #[(2, 19), (2, 19), (2, 19), (2, 19), (2, 19)]
                label_gaze = [gaze_dict[id] for id in label_gaze_list]        #[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)] 
                # label_YPR = [YawPR_dict[id] for id in label_id]        #[(3, 19), (3, 19), (3, 19), (3, 19), (3, 19)]
                label_YPR = [np.concatenate((YawPR_dict[id][:3], YawPR_dict[id][:3,:2]), axis=-1) for id in label_id]  #[(3, 21), (3, 21), (3, 21), (3, 21), (3, 21)]
                # print(label_YPR[0].shape)

                data_21 = [np.concatenate((frame, gaze), axis=-1) for frame, gaze in zip(data_frame, data_gaze)]     #[(2, 21), (2, 21), (2, 21), (2, 21), (2, 21)]
                print("data_21:", data_21[0].shape)
                #normalize data_21
                data_21 = [normalize_points(points) for points in data_21]
                data_21 = data_21 * 6
                print("data_21:", data_21[0].shape)
                
                #cat (2, 21) for yaw pitch 
                data_YP = [np.concatenate((land_21, YawP), axis=0) for land_21, YawP in zip(data_21, data_YPR)]    #[(5, 21), (5, 21), (5, 21), (5, 21), (5, 21)]
                #print(data_YP[0])
                data = np.asarray(data_YP)                         #(T, 4, 21)

                label_points = [np.concatenate((target_land, target_gaze), axis=-1) for target_land, target_gaze in zip(label_land, label_gaze)]     #[(2, 21), (2, 21), (2, 21), (2, 21), (2, 21)]
                label_norm = [normalize_points(points) for points in label_points] 
                label_data = [np.concatenate((landmarks, YawPR), axis=0) for landmarks, YawPR in zip(label_norm, label_YPR)]       
                label_arr = np.asarray(label_data)     #(pred_T, 5, 21)
                
                if i < 24 * 0.7:         
                    train_data_list.append(data)      #(T, C, V)
                    train_label_list.append(label_arr)  
                else:                         #for F05 & M12
                    valid_data_list.append(data)      #(T, C, V)
                    valid_label_list.append(label_arr) 

                #for debug 
                image_id_list.append(image_list)
                gaze_id_list.append(gaze_list)
                label_id_list.append(label_id)

    if gaze_prior:
        print('image_id_list: ', image_id_list[0])
        print('gaze_id_list: ', gaze_id_list[0])
        print('label_id_list: ', label_id_list[0])

    if i < 24 * 0.7:         
        print('train_data length: ', len(train_data_list))
        print(train_data_list[0].shape)
        print(train_label_list[0].shape)
    else:
        print('valid_data length: ',len(valid_data_list))
        print(valid_data_list[0].shape)
        print(valid_label_list[0].shape)

if gaze_prior:
    if not os.path.exists('./data/data_GAZE_prior/data_GAZE_'+args.predict_frames):
        os.makedirs('./data/data_GAZE_prior/data_GAZE_'+args.predict_frames)
    np.save('./data/data_GAZE_prior/data_GAZE_'+args.predict_frames+'/UPNA_data_gaze_'+str(gaze_before)+'.npy', train_data_list)
    np.save('./data/data_GAZE_prior/data_GAZE_'+args.predict_frames+'/UPNA_label_gaze_'+str(gaze_before)+'.npy', train_label_list)
    # np.save('./data/data_GAZE_prior/data_GAZE_'+args.predict_frames+'/valid_data_gaze_'+str(gaze_before)+'.npy', valid_data_list)
    # np.save('./data/data_GAZE_prior/data_GAZE_'+args.predict_frames+'/valid_label_gaze_'+str(gaze_before)+'.npy', valid_label_list)




