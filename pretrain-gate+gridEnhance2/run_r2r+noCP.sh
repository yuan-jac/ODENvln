NODE_RANK=0
NUM_GPUS=2
outdir=../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.sap-init.lxmert-aug.speaker-new-nock-allmodule

# train (使用 torchrun)
CUDA_VISIBLE_DEVICES='2,3' torchrun --master_port 29503 \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/r2r_model_config.json \
    --config config/r2r_pretrain.json \
    --output_dir $outdir \
