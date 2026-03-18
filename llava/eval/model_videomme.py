import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from glob import glob
from PIL import Image
import math
import pandas as pd
from decord import VideoReader, cpu
import numpy as np
import random


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def load_video(video_path, args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    # frame_idx = [i for i in range(0, len(vr), fps)]
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    frame_time = [i/fps for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    video_time = total_frame_num / fps
    return spare_frames, frame_time, video_time


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    if model_name == "LLaVA-Video-7B-Qwen2":
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = 2
        overwrite_config["mm_spatial_pool_mode"] = "average"
        overwrite_config["mm_pooling_position"] = "before"
        overwrite_config["mm_newline_position"] = "grid"
        overwrite_config["delay_load"] = False
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, torch_dtype="bfloat16",attn_implementation="sdpa", overwrite_config=overwrite_config)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, torch_dtype="bfloat16", attn_implementation="sdpa")

    # Data
    df = pd.read_csv(args.gt_file)
    gt_questions = []

    for index, row in df.iterrows():
        # You can access data for each column by column name
        # import pdb; pdb.set_trace()

        # option = eval(row['options'])
        option_str = str(row['options'])[1:-1]
        option_dict = re.findall(r'([A-D])\.\s([^\.?]+[\.?])', option_str)
        option = [f'{idx}. {answer}' for idx, answer in option_dict]
        answer_id = row['answer']
        answer = option_dict[ord(answer_id) - ord('A')][1]
        

        index2ans = {}
        for idx, ans in option_dict:
            index2ans[idx] = ans
        

        gt_questions.append({
            'qid': row["question_id"],
            'question': row["question"],
            'video_id': row["videoID"],
            'duration_group': row['duration'],
            'option': option, 
            'answer_id': answer_id, 
            'answer': answer,
            'index2ans': index2ans,
        })
        
    questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir, exist_ok=True)
    # if args.num_chunks > 1:
    #     output_name = f"{args.num_chunks}_{args.chunk_idx}"
    # else:
    #     output_name = args.output_name
    # answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    # ans_file = open(answers_file, "a")

    # video_formats = [".mp4", ".avi", ".mov", ".mkv"]
    # if args.num_chunks > 1:
    #     output_name = f"{args.num_chunks}_{args.chunk_idx}"
    # else:
    #     output_name = args.output_name
    # answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    # ans_file = open(answers_file, "a")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")  # 你当前用的是 .json（JSONL 格式）

    # —— 读取已存在的 id —— 
    existing_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        existing_ids.add(rec["id"])
                except json.JSONDecodeError:
                    # 如果存在坏行，忽略以保证不中断
                    continue
    # —— 以追加模式打开输出文件 —— 
    ans_file = open(answers_file, "a", encoding="utf-8")
    miss_qid = []
    
    for line in tqdm(questions):
        qid = line["qid"]
        if qid in existing_ids:
            continue
        qid = line["qid"]
        answer = line["answer"]
        video_id = line["video_id"]
        answer_id = line["answer_id"]
        option = line["option"]
        index2ans = line["index2ans"]
        duration_group = line["duration_group"]

        question = [line['question']] + option
        question = '\n'.join(question)
        question = f'{question}\nPlease answer directly with only the letter of the correct option and nothing else.'

        sample_set = {
            "id": qid, 
            "video_id": video_id,
            "question": question, 
            "answer_id": answer_id, 
            "duration_group": duration_group,
            'answer': answer,
        }

        video_path = os.path.join(args.video_dir, f'{video_id}.mp4')
        # print(video_path)
        # exit(0)
        # Check if the video exists
        # print(video_path)
        if os.path.exists(video_path):
            # print(video_path)
            video, _, _ = load_video(video_path, args) # [T,C,W,H]
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            video = [video]
            # exit(0)
        else:
            continue
        qs = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                modalities= ["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print(outputs)

        parsed_pred = parse_multi_choice_response(outputs, ["A", "B", "C", "D"], index2ans)
        sample_set['acc'] = str(parsed_pred == answer_id)   
        sample_set["pred"] = outputs

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps(sample_set)+ "\n")
        ans_file.flush()
        del output_ids
        del input_ids
        del stopping_criteria
        del video 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        import gc; gc.collect()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--video-dir", type=str, default="")
    parser.add_argument("--gt-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--output-name", type=str, default="default")
    parser.add_argument("--question-type", type=str, default="multi_choice")
    parser.add_argument("--for_get_frames_num", type=int, default=16)

    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)