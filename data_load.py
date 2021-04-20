import numpy as np
import torch

SEED = 2020
torch.manual_seed(SEED)

class Dataset(torch.utils.data.Dataset):
        """ Feeder for BIWI dataset in the long-term head pose forecasting tasks
        Arguments:
            data_path: the path to '.npy' data, the shape of data should be (N, T, C, V)
            label_path: the path to label
            debug: If true, only use the first 300 samples
            predict_frames: number of forecasted frames in the long-term head pose forecasting tasks
        """
        
        def __init__(self,
                    data_path,
                    label_path,
                    debug=False,
                    mmap=False,
                    predict_frames=3):
            self.debug = debug
            self.data_path = data_path
            self.label_path = label_path
            self.predict_frames = predict_frames
            self.load_data(mmap)

        def load_data(self, mmap):
            # data: N C V T 

            # load label
            self.label = np.load(self.label_path)
            # print('label.shape', self.label.shape)
            self.label = self.label[:,:self.predict_frames,:,:]               #(N, T, C, V)
            self.label = np.transpose(self.label, (0,1,3,2))   #(N, T, V, C)
            
            # load data
            if mmap:
                self.data = np.load(self.data_path, mmap_mode='r')
            else:
                self.data = np.load(self.data_path)

            self.data = np.transpose(self.data, (0,1,3,2))
            #print(self.data.shape)

            if self.debug:
                self.label = self.label[0:300]
                self.data = self.data[0:300]
                
            self.N, self.T, self.V, self.C = self.data.shape

        def __len__(self):
            return len(self.label)

        def __getitem__(self, index):
            # get data
            data_numpy = np.array(self.data[index])
            label = self.label[index]

            return data_numpy, label

def data_load(model_type, data_path, YawPR_frame_interval=5, gaze_frame_interval=2):

    if model_type == 'GAE':
        from model.model_GAE import Model
        input_channels = 2
        prior = 'YPR'  #'YPR' OR 'gaze'
        data_name = 'YPR_30'
        dataset = 'YawPR_' + str(YawPR_frame_interval)
        
    elif model_type == 'VGAE':
        from model.model_VGAE import Model
        input_channels = 2
        prior = 'YPR'  #'YPR' OR 'gaze'
        data_name = 'YPR_30'
        dataset = 'YawPR_' + str(YawPR_frame_interval)
    
    elif model_type == 'CST-VGAE': 
        #for gaze_prior
        from model.model_CST_VGAE import Model
        input_channels = 4
        prior = 'GAZE'  #'YPR' OR 'gaze'
        data_name = 'GAZE_30'
        dataset = 'gaze_' + str(gaze_frame_interval) 
    
    train_data = data_path+'/data_'+str(prior)+'_prior/data_'+str(data_name)+'/train_data_'+str(dataset)+'.npy'
    train_label = data_path+'/data_'+str(prior)+'_prior/data_'+str(data_name)+'/train_label_'+str(dataset)+'.npy'
    valid_data = data_path+'/data_'+str(prior)+'_prior/data_'+str(data_name)+'/valid_data_'+str(dataset)+'.npy'
    valid_label = data_path+'/data_'+str(prior)+'_prior/data_'+str(data_name)+'/valid_label_'+str(dataset)+'.npy'


    return train_data, train_label, valid_data, valid_label, Model, input_channels