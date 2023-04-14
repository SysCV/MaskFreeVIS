export PYTHONPATH=$PYTHONPATH:`pwd`

CUDA_VISIBLE_DEVICES=0 python3 demo_video/demo.py --config-file configs/youtubevis_2019/video_maskformer2_R101_bs16_8ep.yaml --save-frames True \
  --input './datasets/ytvis_2019/valid/JPEGImages/' \
  --output 'r101_vis/' \
  --opts MODEL.WEIGHTS ./mfvis_models/model_final_r101_0491.pth
