#conda activate pythia
export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_BASE_URL=https://api.wandb-cn.top
export WANDB_API_KEY=7a2d895463d865263cf5d187687a66e3a64534ec
export SWANLAB_API_KEY=4ssKiG3Zo7KqbNo3kENIX
../nsight-systems/2025.3.1/bin/nsys launch --session-new=my_session1 --trace=nvtx,cuda,cublas,cudnn,osrt --cuda-graph-trace=node --wait=all --cuda-memory-usage=true --trace-fork-before-exec=true \
python3 deepy.py train.py /apdcephfs_sh7/share_300819555/hunyuan_infer/zhiyugui/gpt-neox/configs/pythia/14M.yml
