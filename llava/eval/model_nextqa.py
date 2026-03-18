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

def load_video(video_path, args, start=None, end=None, sample_frames=None, is_topk=False, topk_idxs=None):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    video_time = total_frame_num / fps
    if is_topk == False:
        # uniform_sampled_frames = np.linspace(start*fps, min(end*fps, total_frame_num-1), sample_frames, dtype=int)
        uniform_sampled_frames = np.linspace(0, total_frame_num-1, args.for_get_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    else:
        frame_idx = topk_idxs
        # print(frame_idx)
    frame_time = [i/fps for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # return spare_frames, frame_time, video_time
    return spare_frames

def eval_model(args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    overwrite_config = {}
    if model_name == "LLaVA-Video-7B-Qwen2":
        overwrite_config["mm_spatial_pool_stride"] = 2
        overwrite_config["mm_spatial_pool_mode"] = "average"
        overwrite_config["mm_pooling_position"] = "before"
        overwrite_config["mm_newline_position"] = "grid"
        overwrite_config["delay_load"] = False
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, attn_implementation="sdpa", overwrite_config=overwrite_config)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, attn_implementation="sdpa")

    # Data
    df = pd.read_csv(args.gt_file)
    with open("/data/jsb/datasets/nextqa/annotations/multi_choice/map_vid_vidorID.json") as file:
        vid_map = json.load(file)
    gt_questions = []
    for index, row in df.iterrows():
        option = []
        for i in range(5):
            option.append(row['a{}'.format(str(i))].strip())
        
        gt_questions.append({"qid": row["qid"], "video_name": vid_map[str(row["video"])], "video_id":str(row["video"]), "question": row["question"], "answer_id": row["answer"], 'option': option, 'answer': row[f'a{row["answer"]}'], 'type': row['type']})
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")
    
    total_len = len(gt_questions)
    correct = 0
    pred_wups = []
    for sample in tqdm(gt_questions):
        video_name = sample["video_name"] + ".mp4"
        qid = sample["qid"]
        answer = sample["answer"]
        video_id = sample["video_id"]
        answer_id = sample["answer_id"]
        option = sample["option"]

        if args.question_type == 'open_ended':
            question = f'{sample["question"]}?\nAnswer the question using several words or phrase.'
        elif args.question_type == 'multi_choice':
            question = [sample['question']]
            OPTIONS = ["A", "B", "C", "D", "E"]
            index2ans = {}
            for i in range(5):
                question.append(f'{OPTIONS[i]}. {sample["option"][i]}')
                index2ans[OPTIONS[i]] = sample['option'][i]
            question = '\n'.join(question)
            question = f'{question}\nPlease answer directly with only the letter of the correct option and nothing else.'
        sample_set = {"id": qid, "video_id": video_id,"question": question, "answer": answer, "answer_id": answer_id, 'type': sample['type']}
        video_path = os.path.join(args.video_dir, video_name)
        print(video_path)

        # Check if the video exists
        if os.path.exists(video_path):
            video = load_video(video_path, args) # [T,C,W,H]
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
            video = [video]
        
        qs = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        args.conv_mode = "qwen_1_5"
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
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        if args.question_type == 'multi_choice':
            print(f"Answer: {OPTIONS[answer_id]}")
            parsed_pred = parse_multi_choice_response(outputs, OPTIONS, index2ans)
            sample_set['acc'] = str(parsed_pred == OPTIONS[answer_id])
        sample_set["pred"] = outputs


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps(sample_set)+ "\n")
        ans_file.flush()
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