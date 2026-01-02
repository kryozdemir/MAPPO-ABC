#!/bin/bash

# Train MAPPO-ABC on all 5 maps used in the paper
# This script reproduces the experimental results

EPISODES=5000
SEED=42

echo "Starting training on all maps..."
echo "Episodes per map: $EPISODES"
echo "Random seed: $SEED"
echo ""

# Create results directory
mkdir -p results
mkdir -p models

# Array of maps
MAPS=("protoss_extended" "protoss_sharp" "terran_base" "terran_advanced" "zerg_balanced")

for MAP in "${MAPS[@]}"
do
    echo "========================================"
    echo "Training on map: $MAP"
    echo "========================================"
    
    python scripts/train/train_mappo_abc.py \
        --map $MAP \
        --episodes $EPISODES \
        --seed $SEED \
        --save-dir models/$MAP \
        --log-dir results/$MAP \
        --save-interval 500 \
        --eval-interval 100
    
    if [ $? -eq 0 ]; then
        echo "✓ Training completed successfully for $MAP"
    else
        echo "✗ Training failed for $MAP"
    fi
    echo ""
done

echo "========================================"
echo "All training jobs completed!"
echo "========================================"
echo "Models saved to: models/"
echo "Results saved to: results/"
echo ""
echo "To visualize results, run:"
echo "python scripts/visualize.py --log-dirs results/* --labels ${MAPS[@]}"
