
#!/bin/bash
ROOT_DIR="/data/jsb/ForestPrune"
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT=/data/jsb/.model_weights/LLaVA-Video-7B-Qwen2
CONV_MODE=qwen_1_5

# CKPT=./.model_weights/llava-onevision-qwen2-7b-ov
# CONV_MODE=qwen_1_5

TASK=mlvu
echo $TASK

QUESTION_TYPE=multi_choice
EVAL_ONLY=False
CHUNKS=4
FRAMES=32
METHOD=pixel_diff # METHOD=("baseline" "forestprune" "region_cycle" "pixel_diff")
 
KEEP_RATIO=0.1

TAU=0.9
TAU_SPATIAL=0.8
GPRUNE_RATIO=0.5

NUM_REGIONS=10
KEEP_REGIONS_PER_FRAME=1
REGION_MODE=interleave
PRUNE_STATS_PATH=./collect_prune_stats.json

FIRST_KEEP_RATIO=0.2
DIFF_THRESH_Q=0.7

if [ "$METHOD" == "baseline" ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_${QUESTION_TYPE}_${METHOD}
elif [ "$METHOD" == "forestprune" ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_${QUESTION_TYPE}_${METHOD}_${KEEP_RATIO}_${TAU}_${TAU_SPATIAL}_${GPRUNE_RATIO}
elif [ "$METHOD" == "region_cycle" ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_${QUESTION_TYPE}_${METHOD}_${NUM_REGIONS}_${KEEP_REGIONS_PER_FRAME}_${REGION_MODE}
elif [ "$METHOD" == "pixel_diff" ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_${QUESTION_TYPE}_${METHOD}_${KEEP_RATIO}_${FIRST_KEEP_RATIO}_${DIFF_THRESH_Q}
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
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR \
        METHOD=$METHOD KEEP_RATIO=$KEEP_RATIO TAU=$TAU TAU_SPATIAL=$TAU_SPATIAL GPRUNE_RATIO=$GPRUNE_RATIO \
        PRUNE_STATS_PATH=$PRUNE_STATS_PATH NUM_REGIONS=$NUM_REGIONS KEEP_REGIONS_PER_FRAME=$KEEP_REGIONS_PER_FRAME REGION_MODE=$REGION_MODE \
        FIRST_KEEP_RATIO=$FIRST_KEEP_RATIO DIFF_THRESH_Q=$DIFF_THRESH_Q \
        python3 -W ignore llava/eval/model_${TASK}.py \
            --model-path $CKPT \
            --video-dir /data/jsb/datasets/MLVU/video \
            --gt-file /data/jsb/datasets/MLVU/annotations/multiple_choice.json \
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