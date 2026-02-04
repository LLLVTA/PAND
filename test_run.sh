#!/bin/bash
# Quick test run with single dataset and fewer epochs
# Using DTD dataset as it's smaller

# python src/train.py \
#     data/attributes=3_DTD \
#     trainer=gpu \
#     trainer.max_epochs=2 \
#     data.batch_size=32 \
#     model.use_teacher=false \
#     logger=csv \
#     tags=["test_run"]

nohup python src/train.py \
    data/attributes=0_CUB_200_2011 \
    trainer=ddp \
    trainer.devices=4 \
    trainer.max_epochs=300 \
    data.batch_size=128 \
    model.use_teacher=true \
    logger=csv \
    'tags=["cub","prompt_ensemble","8templates","vl2lite"]' \
    > experiments/logs/train_cub_prompt_ensemble_$(date +"%Y%m%d_%H%M%S").log 2>&1 &
