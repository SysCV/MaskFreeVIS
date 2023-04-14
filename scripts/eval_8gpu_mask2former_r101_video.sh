export PYTHONPATH=$PYTHONPATH:`pwd`

ID=159


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_net_video.py --num-gpus 8 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8ep.yaml\
        --eval-only MODEL.WEIGHTS ./mfvis_models/model_final_r101_0491.pth
