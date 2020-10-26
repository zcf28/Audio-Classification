CUDA_VISIBLE_DEVIRCES=0 
python -u main.py \
 --dataset esc50 \
 --netType EnvNet4  \
 --nEpochs 1600 \
 --batchSize 16 \
 --optimizer Adam \
 --LR 1e-3 \
 --save_model 1000 1500 \
 --milestones 1000 1400 \
 --data /home/yons/chengfei/bc_learning_sound_pytorch-master/datasets/orign-data \
 --save_dir /home/yons/chengfei/bc_learning_sound_pytorch-master/result/esc50/envnet4_3 \
 --BC \
 --strongAugment
