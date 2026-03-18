
#!/bin/bash
ROOT_DIR="./ForestPrune"
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT=./.model_weights/LLaVA-Video-7B-Qwen2
CONV_MODE=qwen_1_5

# CKPT=./.model_weights/llava-onevision-qwen2-7b-ov
# CONV_MODE=qwen_1_5

TASK=videomme
echo $TASK

QUESTION_TYPE=multi_choice
EVAL_ONLY=False
CHUNKS=8
FRAMES=64
METHOD=forestprune
KEEP_RATIO=0.1
TAU=0.8
TAU_SPATIAL=0.5
GPRUNE_RATIO=0.5
# SPATIAL_TOKENS=9
if [ "$METHOD" == "baseline" ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_${QUESTION_TYPE}_${METHOD}
else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_${QUESTION_TYPE}_${METHOD}_${KEEP_RATIO}_${TAU}_${TAU_SPATIAL}_${GPRUNE_RATIO}_new
fi
echo $SAVE_DIR

# Assuming GPULIST is a bash array containing your GPUs
GPULIST=()
for (( i=0; i<CHUNKS; i++ )); do
    GPULIST+=($i)
done

# Get the number of GPUs
NUM_GPUS=${#GPULIST[@]}

# Calculate GPUs per chunk
GPUS_PER_CHUNK=$((NUM_GPUS / CHUNKS))

if [ "$EVAL_ONLY" == False ]; then
    for IDX in $(seq 1 $CHUNKS); do
        START=$(((IDX-1) * GPUS_PER_CHUNK))
        LENGTH=$GPUS_PER_CHUNK # Length for slicing, not the end index
        
        CHUNK_GPUS=(${GPULIST[@]:$START:$LENGTH})
        
        # Convert the chunk GPUs array to a comma-separated string
        CHUNK_GPUS_STR=$(IFS=,; echo "${CHUNK_GPUS[*]}")

        echo "CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR"
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 METHOD=$METHOD KEEP_RATIO=$KEEP_RATIO TAU=$TAU TAU_SPATIAL=$TAU_SPATIAL GPRUNE_RATIO=$GPRUNE_RATIO CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR \
        python3 -W ignore llava/eval/model_${TASK}.py \
            --model-path $CKPT \
            --video-dir ./datasets/Video-MME/data \
            --gt-file ./datasets/Video-MME/videomme/test_anns.csv \
            --output-dir ./outputs/$TASK/$SAVE_DIR \
            --output-name pred \
            --num-chunks $CHUNKS \
            --chunk-idx $(($IDX - 1)) \
            --for_get_frames_num $FRAMES \
            --conv-mode $CONV_MODE &
    done

    wait
fi

python3 ./llava/eval/eval_$TASK.py \
    --output-dir ./outputs/$TASK/$SAVE_DIR \
    --eval-type $QUESTION_TYPE \
    --num-chunks $CHUNKS

