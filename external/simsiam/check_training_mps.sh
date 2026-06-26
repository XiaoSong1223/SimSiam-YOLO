#!/bin/bash
# Script to check MPS training progress

CHECKPOINT_DIR="./checkpoints"
TRAINING_LOG="training.log"

echo "=========================================="
echo "SimSiam CSPDarknet 训练状态检查 (MPS)"
echo "=========================================="
echo ""

# Check training process
echo "训练进程状态:"
if ps aux | grep train_cspdarknet | grep -v grep > /dev/null; then
    echo "✓ 训练进程正在运行"
    ps aux | grep train_cspdarknet | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "✗ 未找到训练进程"
fi
echo ""

# Check checkpoints
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoints目录:"
    CHECKPOINT_COUNT=$(ls -1 "$CHECKPOINT_DIR"/*.pth.tar 2>/dev/null | wc -l)
    if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
        echo "✓ 找到 $CHECKPOINT_COUNT 个checkpoint文件"
        echo ""
        echo "最新的checkpoints:"
        ls -lht "$CHECKPOINT_DIR"/*.pth.tar 2>/dev/null | head -5 | awk '{print "  " $9, "(" $5 ")"}'
        echo ""
        echo "最新checkpoint:"
        LATEST=$(ls -t "$CHECKPOINT_DIR"/*.pth.tar 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "  $LATEST"
            ls -lh "$LATEST" | awk '{print "  大小:", $5, "修改时间:", $6, $7, $8}'
        fi
    else
        echo "⚠  Checkpoints目录存在但还没有checkpoint文件"
    fi
else
    echo "⚠  Checkpoints目录不存在"
fi
echo ""

# Check training log
if [ -f "$TRAINING_LOG" ]; then
    echo "训练日志 (最后20行):"
    echo "----------------------------------------"
    tail -20 "$TRAINING_LOG"
    echo "----------------------------------------"
else
    echo "⚠  训练日志文件不存在"
fi
echo ""

# Check MPS availability
echo "MPS设备状态:"
python3 -c "import torch; print('  MPS可用:', torch.backends.mps.is_available()); print('  MPS已构建:', torch.backends.mps.is_built())" 2>/dev/null || echo "  无法检查MPS状态"

