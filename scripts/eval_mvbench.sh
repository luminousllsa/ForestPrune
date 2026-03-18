#!/bin/bash
set -eo pipefail

ROOT_DIR="./ForestPrune"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

if [ ! -e "$ROOT_DIR" ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd "$ROOT_DIR"

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT=./.model_weights/LLaVA-Video-7B-Qwen2
CONV_MODE=qwen_1_5

# CKPT=./.model_weights/llava-onevision-qwen2-7b-ov
# CONV_MODE=qwen_1_5

TASK=mvbench
echo "TASK: $TASK"

MV_JSON_DIR=./datasets/MVBench/json
DATA_PREFIX=./datasets/MVBench/video

RES=224
FRAMES=64

QUESTION_TYPE=multi_choice
EVAL_ONLY=False
CHUNKS=8

METHOD=forestprune
KEEP_RATIO=0.1
TAU=0.9
TAU_SPATIAL=0.8
GPRUNE_RATIO=0.5

if [ "$METHOD" == "baseline" ]; then
    SAVE_DIR="$(basename "$CKPT")_${CONV_MODE}_frames_${FRAMES}_${QUESTION_TYPE}_${METHOD}"
else
    SAVE_DIR="$(basename "$CKPT")_${CONV_MODE}_frames_${FRAMES}_${QUESTION_TYPE}_${METHOD}_${KEEP_RATIO}_${TAU}_${TAU_SPATIAL}_${GPRUNE_RATIO}"
fi
echo "SAVE_DIR: $SAVE_DIR"

OUTPUT_DIR="./outputs/$TASK/$SAVE_DIR"
mkdir -p "$OUTPUT_DIR"

GPULIST=(0 1 2 3 4 5 6 7)

NUM_GPUS=${#GPULIST[@]}
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "No GPUs configured in GPULIST"
  exit 1
fi

GPUS_PER_CHUNK=$(( NUM_GPUS / CHUNKS ))
if [ "$GPUS_PER_CHUNK" -lt 1 ]; then
  GPUS_PER_CHUNK=1
fi

if [ "$EVAL_ONLY" == "False" ]; then
  for IDX in $(seq 1 $CHUNKS); do
      START=$(((IDX-1) * GPUS_PER_CHUNK))
      LENGTH=$GPUS_PER_CHUNK
      if [ $IDX -eq $CHUNKS ]; then
          LENGTH=$(( NUM_GPUS - START ))
          [ $LENGTH -le 0 ] && LENGTH=1
      fi

      CHUNK_GPUS=("${GPULIST[@]:$START:$LENGTH}")
      CHUNK_GPUS_STR=$(IFS=,; echo "${CHUNK_GPUS[*]}")

      echo "Chunk $IDX/$CHUNKS -> CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR"

      echo "CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR"
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 METHOD=$METHOD KEEP_RATIO=$KEEP_RATIO TAU=$TAU TAU_SPATIAL=$TAU_SPATIAL GPRUNE_RATIO=$GPRUNE_RATIO \
      python3 -W ignore llava/eval/model_${TASK}.py \
          --model-path $CKPT \
          --mvbench_json_dir $MV_JSON_DIR \
          --data_prefix $DATA_PREFIX \
          --conv-mode $CONV_MODE \
          --topk $FRAMES \
          --resolution $RES \
          --output-dir $OUTPUT_DIR \
          --output-name pred \
          --num-chunks $CHUNKS \
          --chunk-idx $((IDX - 1)) &
  done

  wait
fi

python3 ./llava/eval/eval_${TASK}.py \
    --output_dir "$OUTPUT_DIR" \
    --eval_type "$QUESTION_TYPE" \
    --num-chunks "$CHUNKS"