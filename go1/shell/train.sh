set -xe
set -o pipefail

export TZ='Asia/Shanghai'

CFG=$1
WORLD_SIZE=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# Auto-select available port if MASTER_PORT not set
if [ -z "${MASTER_PORT+x}" ]; then
    # Try default port first, then find random available port
    MASTER_PORT=12345
    if ss -tuln | grep -q ":${MASTER_PORT} "; then
        echo "[WARNING] Port ${MASTER_PORT} is already in use, selecting random available port..."
        MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
        echo "[INFO] Using port: ${MASTER_PORT}"
    fi
else
    echo "[INFO] Using specified port: ${MASTER_PORT}"
fi

if [ -z "${RUNNAME+x}" ]; then  
    echo "[ERROR] RUNNAME is not set, please inject RUNNAME for experiment output directory" 
    exit 1
else  
    echo "RUNNAME is set to: $RUNNAME"  
fi

OUTPUT_DIR="./experiment"/${RUNNAME}
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
LOG_OUTPUT_DIR="experiment"/${RUNNAME}/"log"
if [ ! -d "$LOG_OUTPUT_DIR" ]; then
  mkdir -p "$LOG_OUTPUT_DIR"
fi

export WANDB_PROJECT="go1"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LAUNCHER=pytorch
export NCCL_P2P_LEVEL=NVL

torchrun \
  --nnodes=${WORLD_SIZE} \
  --node-rank=${RANK} \
  --master-addr=${MASTER_ADDR} \
  --nproc-per-node=${NPROC_PER_NODE} \
  --master-port=${MASTER_PORT} \
  go1/internvl/train/go1_train.py \
  --cfg ${CFG} \
  2>&1 | tee -a "${LOG_OUTPUT_DIR}/training_log_nodeIdx$(printf "%03d" ${RANK})_$(date +"%Y%m%d_%H%M").txt"
