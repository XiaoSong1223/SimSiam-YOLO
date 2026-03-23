#!/bin/bash
# Script to check training progress

CHECKPOINT_DIR="./checkpoints"
echo "Checking training progress..."
echo "================================"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoints found:"
    ls -lh "$CHECKPOINT_DIR"/*.pth.tar 2>/dev/null | tail -5
    echo ""
    echo "Latest checkpoint:"
    ls -t "$CHECKPOINT_DIR"/*.pth.tar 2>/dev/null | head -1
else
    echo "Checkpoints directory not found yet"
fi

echo ""
echo "Training process:"
ps aux | grep train_cspdarknet | grep -v grep || echo "No training process found"

