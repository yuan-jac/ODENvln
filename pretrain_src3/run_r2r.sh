NODE_RANK=0
NUM_GPUS=2
outdir=../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new-ckp-3-2

# train (使用 torchrun)
CUDA_VISIBLE_DEVICES='3,4' torchrun --master_port 29513 \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/r2r_model_config.json \
    --config config/r2r_pretrain.json \
    --output_dir $outdir \
    --checkpoint /home/files/A/zhanghuaxiang3/GridMM/datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new-ckp-3/ckpts/model_step_54000.pt