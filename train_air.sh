CUDA_VISIBLE_DEVICES=4,5 python main.py --bs 30 --net 'resnet50' --data aircraft --epochs 100 --drop_rate 0.15 --gpu 0,1