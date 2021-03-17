#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='./input'

# You need to modify this path to your workspace to store features and models
WORKSPACE='./work'

# Hyper-parameters
GPU_ID=1
SEGMENT_SECONDS=5
HOP_SECONDS=2
PAD_TYPE='repeat'  # 'constant' | 'repeat'
HOLDOUT_FOLD=1   # split into 4 folds for validation
MODEL_TYPE='Cnn_9layers_AvgPooling'
BATCH_SIZE=32
MAX_ITERS=20000

# Create cross validation files
python utils/create_cross_validation_files.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_curated'
python utils/create_cross_validation_files.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_noisy'

# Calculate feature
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_curated'
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_noisy'
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='test'

# Calculate scalar
python utils/features.py calculate_scalar --data_type='train_noisy' --workspace=$WORKSPACE

############ Train and validate on development dataset ############
# Train
CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --train_source='curated_and_noisy' --segment_seconds=$SEGMENT_SECONDS --hop_seconds=$HOP_SECONDS --pad_type=$PAD_TYPE --holdout_fold=$HOLDOUT_FOLD --model_type='Cnn_9layers_AvgPooling' --batch_size=$BATCH_SIZE --max_iters=$MAX_ITERS --cuda

# Inference on validation data
CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --train_source='curated_and_noisy' --segment_seconds=$SEGMENT_SECONDS --hop_seconds=$HOP_SECONDS --pad_type=$PAD_TYPE --holdout_fold=$HOLDOUT_FOLD --model_type='Cnn_9layers_AvgPooling' --iteration=20000 --batch_size=$BATCH_SIZE --cuda

# Plot statistics
python utils/plot_results.py --workspace=$WORKSPACE --train_source=curated_and_noisy --segment_seconds=$SEGMENT_SECONDS --hop_seconds=$HOP_SECONDS --pad_type=$PAD_TYPE


############ Train on full development dataset ############
# Train
# CUDA_VISIBLE_DEVICES=3 python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --train_source='curated_and_noisy' --segment_seconds=$SEGMENT_SECONDS --hop_seconds=$HOP_SECONDS --pad_type=$PAD_TYPE --holdout_fold='none' --model_type='Cnn_9layers_AvgPooling' --batch_size=$BATCH_SIZE --cuda

# Inference on test data and write out submission
# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_test --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --train_source='curated_and_noisy' --segment_seconds=$SEGMENT_SECONDS --hop_seconds=$HOP_SECONDS --pad_type=$PAD_TYPE --model_type='Cnn_9layers_AvgPooling' --iteration=20000 --batch_size=$BATCH_SIZE --cuda

############ END ############
