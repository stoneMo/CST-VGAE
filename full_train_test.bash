#!/bin/bash

###########################
# Necessary Configurations

SKIP_TRAINING=0  # 1 for evaluation and 0 for training model

# NOTE: please make sure to process data according to read.me or download data from the given link. 
data_path='./data_preprocessing/data'

# Select model type
model_type='CST-VGAE'       # GAE or VGAE or CST-VGAE


if [[ $SKIP_TRAINING -eq 0 ]]
then
	############################
	# 1. training
	#
	#    The original setup used here was with:
	#    > Batch size:  1024
	#    > # of epochs: 300
	#    > # of GPUs:   1
	#    > GPU model:   NIVIDA GeForce RTX 2080 Ti
	python main.py \
		--data_path ${data_path} \
		--model_type ${model_type} \
		\
		--pipeline_mode train \
		--pretrain 0 \
		--predict_frames 30 \
		\
		--embedd_dim 32 \
		--batch_size 1024 \
		--epoch_num 300 \

		#####################################################################################
		# NOTE: when adding the lines below, make sure to use the backslash ( \ ) correctly,
		#       such that the full command is correctly constructed and registered.

		# Use (append to above) the line below if wanting to use pre-trained weights to fine-tune the model
		# DO NOT JUST UNCOMMENT IT, IT WILL HAVE NO EFFECT DUE TO BASH PARSING
		# --pretrain 1\
fi

if [[ $SKIP_TRAINING -eq 1 ]]
then

	###########################
	# 2. Evaluating
	#
	#    NOTE: DO NOT SET 'PRETRAIN' TO 0, OTHERWISE YOU WILL NOT LOAD PRETRAIN MODEL WEIGHTS.
	#           

	python main.py \
		--data_path ${data_path} \
		--model_type ${model_type} \
		\
		--pipeline_mode test \
		--pretrain 1 \
		--predict_frames 30 \
		\
		--embedd_dim 32 \
		--batch_size 1024 \

fi
