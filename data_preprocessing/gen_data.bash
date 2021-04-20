#!/bin/bash

###########################
# Necessary Configurations

SKIP_PKL=1  # 1 for skipping to generate landmark, gaze, ypr pkl files, 0 for implementing this step.
dataset_path='./dataset/hpdb/'   # path to dataset.
output_path='./data/'         # path to output files.

GAZE_PRIOR=1  # 1 for generating GAZE_PRIOR data.
FORECASTED_FRAMES=30 # number of forecasted frames in the long-term head pose forecasting tasks.

# NOTE: please make sure to generate landmarks, gazes, and YPR data according to Step 1.
# OR YOU CAN download data directly from the given link. 
# NOTE: MAKE SURE TO UPDATA THE PATH IF YOU CHANGE THESE DATA PATHS.
landmark_path='./data/landmarks_all.pkl'
gaze_path='./data/gaze_all.pkl'
YawPR_path='./data/YPR_all.pkl'
###########################

if [[ $SKIP_PKL -eq 0 ]]
then
	###########################
	# 1. Generating landmark, gaze, ypr pkl file
	#
	#   NOTE: MAKE SURE TO IMPLEMENT THIS STEP ON GPUS, AS IT WILL SAVE YOUR TIME A LOT.
	#
	python process_landmark.py \
		--dataset_path ${dataset_path} \
		--output_path ${output_path} \
		--max_angle 90 \

	python process_gaze.py \
		--dataset_path ${dataset_path} \
		--output_path ${output_path} \
		--max_angle 90 \
	
	python process_ypr.py \
		--dataset_path ${dataset_path} \
		--output_path ${output_path} \
		--max_angle 90 \

		#####################################################################################
		# NOTE: when adding the lines below, make sure to use the backslash ( \ ) correctly,
		#       such that the full command is correctly constructed and registered.
fi

if [[ $GAZE_PRIOR -eq 1 ]]
then
	###########################
	# 2. Generating GAZE_PRIOR data for CST-VGAE model
	#
	python generate_data.py \
		--predict_frames $FORECASTED_FRAMES \
		\
		--landmark_path ${landmark_path} \
		--gaze_path ${gaze_path} \
		--YawPR_path ${YawPR_path} \
		\
		--Gaze $GAZE_PRIOR \
		--YawPR 0 \

		#####################################################################################
		# NOTE: when adding the lines below, make sure to use the backslash ( \ ) correctly,
		#       such that the full command is correctly constructed and registered.
fi