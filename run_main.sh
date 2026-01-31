#!/bin/bash

# Script to run NeOTF, MORE, and HIO+ER algorithms
# Usage: ./run_main.sh [options]

set -e

CONFIG_FILE="config.yml"
USE_GPU=""
OUTPUT_DIR="./outputs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --use_gpu)
            USE_GPU="--use_gpu"
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_main.sh [--config CONFIG_FILE] [--use_gpu] [--output_dir OUTPUT_DIR]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "NeOTF Algorithm Suite - Benchmark Script"
echo "=========================================="
echo ""
echo "Configuration: $CONFIG_FILE"
echo "Output Directory: $OUTPUT_DIR"
if [ -n "$USE_GPU" ]; then
    echo "GPU: Enabled"
else
    echo "GPU: Disabled"
fi
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run HIO+ER
echo "=========================================="
echo "Running HIO+ER Algorithm..."
echo "=========================================="
python HIOER.py --config "$CONFIG_FILE" $USE_GPU
echo ""

# Run MORE
echo "=========================================="
echo "Running MORE Algorithm..."
echo "=========================================="
python MORE.py --config "$CONFIG_FILE" $USE_GPU
echo ""

# Run NeOTF
echo "=========================================="
echo "Running NeOTF Algorithm..."
echo "=========================================="
python NeOTF.py --config "$CONFIG_FILE"
echo ""

echo "=========================================="
echo "All algorithms completed!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated outputs:"
echo "  - HIO+ER results: $OUTPUT_DIR/HIO_letter_mask_white_50ms/"
echo "  - MORE results: $OUTPUT_DIR/MORE_letter_mask_white_50ms/"
echo "  - NeOTF results: $OUTPUT_DIR/exp_data_num_mask_*/"
