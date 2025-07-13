conda activate pythia
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_BASE_URL=https://api.wandb-cn.top
export WANDB_API_KEY=7a2d895463d865263cf5d187687a66e3a64534ec
python3 deepy.py train.py /NAS/wujunkang/guizhiyu/gpt-neox/configs/pythia/31M.yml