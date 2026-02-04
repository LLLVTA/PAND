#!/bin/bash

# Simpler version: analyze specific error images directly

CHECKPOINT_PATH="/data/lvta/logs/vl2lite/train/runs/cub_vl2lite_2025-11-15_17-30-50/checkpoints/epoch_179.ckpt"
ERROR_DIR="/data/lvta/fault_analysis/0_CUB_200_2011/distilled/error_images"
OUTPUT_DIR="/data/lvta/attention_analysis/0_CUB_200_2011/distilled"

echo "Analyzing attention for black bird errors..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_DIR"

python experiments/analyze_attention_simple.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --error_dir "$ERROR_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_classes Fish_Crow American_Crow Common_Raven Brewer_Blackbird Shiny_Cowbird Boat_tailed_Grackle \
    --num_samples 50

echo "Done! Check results at: $OUTPUT_DIR"
