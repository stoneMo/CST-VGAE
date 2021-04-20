import torch
import argparse
from datetime import timedelta

parser = argparse.ArgumentParser(description="main function")
parser.add_argument("--data_path", type=str, default='./data_preprocessing/data', help="the path to the data")
parser.add_argument("--model_type", type=str, default='CST-VGAE', help="model type: GAE or VGAE or CST-VGAE")
parser.add_argument("--pipeline_mode", type=str, default='train', help="train or test")
parser.add_argument("--pretrain", type=int, default=0, help="pretrain must be True during test time")
parser.add_argument("--predict_frames", type=int, default=30, help='forecasted frames in the long-term head pose forecasting tasks')
parser.add_argument("--embedd_dim", type=int, default=32, help='the dimensionality of embeddings')
parser.add_argument("--batch_size", type=int, default=1024, help='batch_size')
parser.add_argument("--epoch_num", type=int, default=300, help='epoch number')
parser.add_argument("--learn_rate", type=float, default='5e-4', help='learn_rate')
parser.add_argument("--bebug_mode", type=bool, default=False, help='bebug or not')
parser.add_argument("--optimizer", type=str, default='Adam', help='optimizer')
parser.add_argument("--gradient_clip", type=bool, default=True, help='gradient clip')
parser.add_argument("--dropout", type=int, default=0, help='dropout')
parser.add_argument("--YawPR", type=int, default=5, help='the frame interval between the forecasted frames and the given condition YawPitchRoll frame')
parser.add_argument("--Gaze", type=int, default=5, help='the frame interval between the forecasted frames and the given condition gaze frame')

args = parser.parse_args()


if torch.cuda.is_available():
    device = 'cuda:0'
else: 
    device = 'cpu'

# kl_loss penalty 
gamma = [1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4]

print("Note that current model type:", args.model_type)

if args.model_type == 'GAE':
    from pipeline_GAE import pipeline
    predict_frames = args.predict_frames # Number of forecasted frames in the long-term head pose forecasting tasks
    
elif args.model_type == 'VGAE':
    from pipeline_VGAE import pipeline
    predict_frames = args.predict_frames # Number of forecasted frames in the long-term head pose forecasting tasks

elif args.model_type == 'CST-VGAE':
    from pipeline_CST_VGAE import pipeline
    predict_frames = args.predict_frames # Number of forecasted frames in the long-term head pose forecasting tasks

best_loss_valid, best_train_loss, time_dif, best_loss_yaw, best_loss_pitch, \
    best_loss_roll, best_loss_mae, best_loss_kl = pipeline(data_path=args.data_path,\
        model_type=args.model_type, embedd_dim=args.embedd_dim, epoch_num=args.epoch_num, \
            debug=args.bebug_mode, batch_size=args.batch_size, predict_frames=predict_frames, \
                learn_rate=args.learn_rate, gradient_clip=args.gradient_clip, gamma=gamma, optim_type=args.optimizer, \
                    dropout=args.dropout, YawPR_frame_interval=args.YawPR, gaze_frame_interval=args.Gaze, \
                        device=device, pretrain=args.pretrain, pipeline_mode=args.pipeline_mode)