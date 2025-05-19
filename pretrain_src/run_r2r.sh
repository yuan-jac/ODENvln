NODE_RANK=0
NUM_GPUS=1
outdir=../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new2

# train (使用 torchrun)
CUDA_VISIBLE_DEVICES='6' torchrun --master_port 29504 \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/r2r_model_config.json \
    --config config/r2r_pretrain.json \
    --output_dir $outdir \
    --checkpoint /home/files/A/zhanghuaxiang3/GridMM/datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new2/ckpts/model_step_75000.pt