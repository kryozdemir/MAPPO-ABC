#!/bin/bash

# Train all baseline algorithms on all maps
# This script reproduces baseline results for comparison

EPISODES=5000
SEED=42

echo "Training all baseline algorithms..."
echo "Episodes per map: $EPISODES"
echo "Random seed: $SEED"
echo ""

# Algorithms to train
ALGORITHMS=("ippo" "mappo" "qmix" "vdn")

# Maps to train on
MAPS=("protoss_extended" "protoss_sharp" "terran_base" "terran_advanced" "zerg_balanced")

for ALGO in "${ALGORITHMS[@]}"
do
    echo "========================================"
    echo "Training algorithm: ${ALGO^^}"
    echo "========================================"
    
    for MAP in "${MAPS[@]}"
    do
        echo "Training on map: $MAP"
        
        python scripts/train/train_baselines.py \
            --algorithm $ALGO \
            --map $MAP \
            --episodes $EPISODES \
            --seed $SEED \
            --save-dir models/baselines/$ALGO \
            --log-dir results/baselines/$ALGO \
            --save-interval 500 \
            --eval-interval 100
        
        if [ $? -eq 0 ]; then
            echo "✓ Training completed successfully for $ALGO on $MAP"
        else
            echo "✗ Training failed for $ALGO on $MAP"
        fi
        echo ""
    done
done

echo "========================================"
echo "All baseline training jobs completed!"
echo "========================================"
echo "Models saved to: models/baselines/"
echo "Results saved to: results/baselines/"
