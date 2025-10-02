#!/bin/bash
# train_with_args.sh - 支持命令行参数覆盖的训练启动脚本
# 使用方法：
#   ./train_with_args.sh --config go1/configs/go1_air_sft_lxy.py --runname my_exp --resume experiment/prev/checkpoint-20000

set -e

# 解析命令行参数
BASE_CONFIG=""
RUNNAME=""
RESUME_CKPT=""
OVERWRITE_DIR="true"
BATCH_SIZE=""
LEARNING_RATE=""

show_help() {
    cat << EOF
Usage: $0 --config <path> --runname <name> [options]

必需参数:
  --config <path>                基础配置文件路径

  --runname <name>               实验名称（输出目录）

可选参数:
  --resume <checkpoint_path>     从指定 checkpoint 恢复（自动设置 overwrite=false）
  
  --overwrite <true|false>       是否覆盖现有输出目录（默认: true）
  
  --batch-size <int>             每卡 batch size
  
  --lr <float>                   学习率

示例:
  # 从 checkpoint 恢复
  $0 --config go1/configs/go1_air_sft_lxy.py \\
     --runname lxy_test_002 \\
     --resume experiment/lxy_test_001/checkpoint-23000

  # 调整超参数
  $0 --config go1/configs/go1_air_sft_lxy.py \\
     --runname hyperp_test \\
     --batch-size 8 \\
     --lr 1e-5

EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --config)
            BASE_CONFIG="$2"
            shift 2
            ;;
        --runname)
            RUNNAME="$2"
            shift 2
            ;;
        --resume)
            RESUME_CKPT="$2"
            OVERWRITE_DIR="false"
            shift 2
            ;;
        --overwrite)
            OVERWRITE_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 验证必需参数
if [ -z "$BASE_CONFIG" ] || [ -z "$RUNNAME" ]; then
    echo "Error: Missing required arguments"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Config file not found: $BASE_CONFIG"
    exit 1
fi

# 创建临时配置文件
TEMP_CONFIG="experiment/${RUNNAME}_temp_config.py"
mkdir -p "experiment"

echo "Creating temporary config: $TEMP_CONFIG"
echo "Base config: $BASE_CONFIG"

# 复制基础配置
cp "$BASE_CONFIG" "$TEMP_CONFIG"

# 使用 sed 覆盖配置值
if [ -n "$RESUME_CKPT" ]; then
    echo "Setting resume_from_checkpoint: $RESUME_CKPT"
    
    # 检查 checkpoint 是否存在
    if [ ! -d "$RESUME_CKPT" ]; then
        echo "Warning: Checkpoint directory not found: $RESUME_CKPT"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # 添加或修改 resume_from_checkpoint
    if grep -q "resume_from_checkpoint" "$TEMP_CONFIG"; then
        sed -i "s|resume_from_checkpoint:.*|resume_from_checkpoint: str = field(default=\"$RESUME_CKPT\")|" "$TEMP_CONFIG"
    else
        # 在 GOTrainingArguments 类定义后添加
        sed -i "/class GOTrainingArguments/,/^class\|^@dataclass\|^$/{ 
            /output_dir.*field/a\\    resume_from_checkpoint: str = field(default=\"$RESUME_CKPT\")
        }" "$TEMP_CONFIG"
    fi
fi

if [ -n "$BATCH_SIZE" ]; then
    echo "Setting per_device_train_batch_size: $BATCH_SIZE"
    sed -i "s/per_device_train_batch_size: int = field(default=[0-9]*[^)]*)/per_device_train_batch_size: int = field(default=$BATCH_SIZE)/" "$TEMP_CONFIG"
fi

if [ -n "$LEARNING_RATE" ]; then
    echo "Setting learning_rate: $LEARNING_RATE"
    sed -i "s/learning_rate: float = field(default=[0-9.e-]*)/learning_rate: float = field(default=$LEARNING_RATE)/" "$TEMP_CONFIG"
fi

# 覆盖 overwrite_output_dir
echo "Setting overwrite_output_dir: $OVERWRITE_DIR"
sed -i "s/overwrite_output_dir: bool = field(default=True)/overwrite_output_dir: bool = field(default=$OVERWRITE_DIR)/" "$TEMP_CONFIG"
sed -i "s/overwrite_output_dir: bool = field(default=False)/overwrite_output_dir: bool = field(default=$OVERWRITE_DIR)/" "$TEMP_CONFIG"

# 导出环境变量
export RUNNAME="$RUNNAME"

echo "=========================================="
echo "Starting training with:"
echo "  RUNNAME: $RUNNAME"
echo "  Config: $TEMP_CONFIG"
[ -n "$RESUME_CKPT" ] && echo "  Resume from: $RESUME_CKPT"
[ -n "$BATCH_SIZE" ] && echo "  Batch size: $BATCH_SIZE"
[ -n "$LEARNING_RATE" ] && echo "  Learning rate: $LEARNING_RATE"
echo "  Overwrite output: $OVERWRITE_DIR"
echo "=========================================="

# 启动训练
bash go1/shell/train.sh "$TEMP_CONFIG"

# 训练完成后保留临时配置（方便复现）
echo "Temporary config saved at: $TEMP_CONFIG"
echo "You can delete it manually if not needed."
