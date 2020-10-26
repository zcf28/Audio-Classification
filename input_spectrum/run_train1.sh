CUDA_VISIBLE_DEVIRCES=0 
python -u main.py \
 --dataset esc50 \
 --netType EnvNet5  \
 --nEpochs 1600 \
 --batchSize 16 \
 --optimizer Adam \
 --LR 1e-3 \
 --save_model 1000 1500 \
 --milestones 1000 1400 \
 --data /home/yons/chengfei/spectrum_learning/datasets/orign-data\
 --save_dir /home/yons/chengfei/spectrum_learning/result/esc50/envnet5_3\
 --BC \
 --strongAugment
