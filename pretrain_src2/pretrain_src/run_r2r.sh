NODE_RANK=0
NUM_GPUS=1
outdir=../../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-mrcplus3

# train (使用 torchrun)
CUDA_VISIBLE_DEVICES='1' torchrun --master_port 29501 \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --nnodes=1 \
    train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/r2r_model_config.json \
    --config config/r2r_pretrain.json \
    --output_dir $outdir \