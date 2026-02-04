#!/bin/bash

# Analyze attention for black bird confusion cases
# This script uses Grad-CAM to visualize what the model is looking at

CHECKPOINT_PATH="/data/lvta/logs/vl2lite/train/runs/cub_vl2lite_2025-11-15_17-30-50/checkpoints/epoch_179.ckpt"
ERROR_CSV="/data/lvta/fault_analysis/0_CUB_200_2011/distilled/error_cases_CUB_200_2011_distilled.csv"
OUTPUT_DIR="/data/lvta/attention_analysis/0_CUB_200_2011/distilled"

# Target classes: black birds that are frequently confused
TARGET_CLASSES=(
    "Fish_Crow"
    "American_Crow"
    "Common_Raven"
    "Brewer_Blackbird"
    "Shiny_Cowbird"
    "Boat_tailed_Grackle"
)

echo "Running Grad-CAM attention analysis..."
echo "Error CSV: $ERROR_CSV"
echo "Target classes: ${TARGET_CLASSES[@]}"
echo "Output: $OUTPUT_DIR"

python experiments/analyze_attention.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --error_csv "$ERROR_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 100 \
    --target_classes "${TARGET_CLASSES[@]}"

echo ""
echo "Analysis complete! Check results at:"
echo "  Visualizations: $OUTPUT_DIR/*.jpg"
echo "  Metrics: $OUTPUT_DIR/attention_metrics.csv"
echo ""
echo "Key metrics to check:"
echo "  - edge_attention > 0.3: Model focusing on edges/background"
echo "  - dispersion > 0.3: Scattered attention, not focused on bird"
echo "  - center near (0.5, 0.5): Centered on object"
