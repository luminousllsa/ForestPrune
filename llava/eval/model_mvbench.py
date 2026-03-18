import argparse
import os
import json
import math
from os.path import exists
import re
import random
import warnings
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
import shortuuid

from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset

from decord import VideoReader, cpu
from PIL import Image
import imageio
import cv2

warnings.filterwarnings("ignore")


class MVBench_dataset(Dataset):
    
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

        self.num_segments = num_segments

    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])

        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()

    def __len__(self):
        return len(self.data_list)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000

        start_idx = max(first_idx, int(round(start * fps)))
        end_idx = min(int(round(end * fps)), max_frame)
        if end_idx <= start_idx:
            end_idx = min(start_idx + max(1, self.num_segments), max_frame)

        seg_size = float(end_idx - start_idx + 1) / self.num_segments
        frame_indices = np.array([
            int(min(max_frame, max(first_idx, start_idx + int(seg_size * idx + seg_size / 2))))
            for idx in range(self.num_segments)
        ], dtype=np.int64)
        return frame_indices

    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 1e-6 else 25.0

        idxs = self.get_index(bound, fps, max_frame, first_idx=0)
        frames = vr.get_batch(idxs).asnumpy()
        return frames  # (T, H, W, 3)

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        idxs = set(self.get_index(bound, fps, max_frame, first_idx=0))
        images = []
        for i, frame in enumerate(gif):
            if i in idxs:
                # RGBA -> RGB
                if frame.shape[-1] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                images.append(frame)
        if len(images) == 0:
            gif.set_image_index(0)
            images = [gif.get_next_data()]
        frames = np.stack(images, axis=0).astype(np.uint8)
        return frames  # (T, H, W, 3)

    def read_frame(self, video_path, bound=None, fps=3):
        total = len(os.listdir(video_path))
        max_frame = total
        idxs = self.get_index(bound, fps, max_frame, first_idx=1)
        images = []
        for fid in idxs:
            fp = os.path.join(video_path, f"{fid:05d}.jpg")
            img = Image.open(fp).convert("RGB")
            images.append(np.array(img, dtype=np.uint8))
        frames = np.stack(images, axis=0)
        return frames  # (T, H, W, 3)

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])

        if not os.path.exists(video_path):
            print(f"[Warning] File not found: {video_path}. Skipping this sample.")
            new_idx = (idx + 1) % len(self.data_list)
            return self.__getitem__(new_idx)

        try:
            frames = decord_method(video_path, bound)  # (T,H,W,3) uint8
        except Exception as e:
            print(f"[Error] Failed to read {video_path}: {e}")
            new_idx = (idx + 1) % len(self.data_list)
            return self.__getitem__(new_idx)

        question, answer = self.qa_template(self.data_list[idx]['data'])

        return {
            'video': frames,          # np.uint8, (T,H,W,3)
            'question': question,
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_multi_choice_response(response, all_choices, index2ans):
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        return ''
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]
    return pred_index


def parse_choices_from_question(question_text: str):
    lines = question_text.splitlines()
    choices, index2ans = [], {}
    for ln in lines:
        m = re.match(r"\(([A-Z])\)\s*(.*)", ln.strip())
        if m:
            key = m.group(1)
            val = m.group(2).strip()
            choices.append(key)
            index2ans[key] = val
    return choices, index2ans


def letter_from_answer_string(answer_text: str):
    m = re.match(r"\(([A-Z])\)", answer_text.strip())
    return m.group(1) if m else ''


def build_data_list(root_prefix: str) -> dict:
    return {
        "Action Sequence": ("action_sequence.json", os.path.join(root_prefix, "star/Charades_v1_480/"), "video", True),
        "Action Prediction": ("action_prediction.json", os.path.join(root_prefix, "star/Charades_v1_480/"), "video", True),
        "Action Antonym": ("action_antonym.json", os.path.join(root_prefix, "ssv2_video/"), "video", False),
        "Fine-grained Action": ("fine_grained_action.json", os.path.join(root_prefix, "Moments_in_Time_Raw/videos/"), "video", False),
        "Unexpected Action": ("unexpected_action.json", os.path.join(root_prefix, "FunQA_test/test/"), "video", False),
        "Object Existence": ("object_existence.json", os.path.join(root_prefix, "clevrer/video_validation/"), "video", False),
        "Object Interaction": ("object_interaction.json", os.path.join(root_prefix, "star/Charades_v1_480/"), "video", True),
        "Object Shuffle": ("object_shuffle.json", os.path.join(root_prefix, "perception/videos/"), "video", False),
        "Moving Direction": ("moving_direction.json", os.path.join(root_prefix, "clevrer/video_validation/"), "video", False),
        "Action Localization": ("action_localization.json", os.path.join(root_prefix, "sta/sta_video/"), "video", True),
        "Scene Transition": ("scene_transition.json", os.path.join(root_prefix, "scene_qa/video/"), "video", False),
        "Action Count": ("action_count.json", os.path.join(root_prefix, "perception/videos/"), "video", False),
        "Moving Count": ("moving_count.json", os.path.join(root_prefix, "clevrer/video_validation/"), "video", False),
        "Moving Attribute": ("moving_attribute.json", os.path.join(root_prefix, "clevrer/video_validation/"), "video", False),
        "State Change": ("state_change.json", os.path.join(root_prefix, "perception/videos/"), "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", os.path.join(root_prefix, "nturgbd/"), "video", False),
        "Character Order": ("character_order.json", os.path.join(root_prefix, "perception/videos/"), "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", os.path.join(root_prefix, "vlnqa/"), "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", os.path.join(root_prefix, "tvqa/frames_fps3_hq/"), "frame", True),
        "Counterfactual Inference": ("counterfactual_inference.json", os.path.join(root_prefix, "clevrer/video_validation/"), "video", False),
    }

def load_existing_ids(path: str):
    
    done = set()
    lines = 0
    if not os.path.exists(path):
        return done, lines
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if "id" in obj:
                    done.add(obj["id"])
                    lines += 1
            except Exception:
                continue
    return done, lines


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    overwrite_config = {}
    if model_name == "LLaVA-Video-7B-Qwen2":
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = 2
        overwrite_config["mm_spatial_pool_mode"] = "average"
        overwrite_config["mm_pooling_position"] = "before"
        overwrite_config["mm_newline_position"] = "grid"
        overwrite_config["delay_load"] = False
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, attn_implementation="sdpa", overwrite_config=overwrite_config)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, attn_implementation="sdpa", overwrite_config=overwrite_config)
    data_dir = args.mvbench_json_dir
    data_list = build_data_list(args.data_prefix)
    dataset = MVBench_dataset(
        data_dir=data_dir,
        data_list=data_list,
        num_segments=args.topk,
        resolution=args.resolution,
    )

    indices = list(range(len(dataset)))
    indices = get_chunk(indices, args.num_chunks, args.chunk_idx)
    print(f"Total samples: {len(dataset)} | Running chunk {args.chunk_idx}/{args.num_chunks} => {len(indices)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"{args.num_chunks}_{args.chunk_idx}" if args.num_chunks > 1 else args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")

    existing_ids, existing_lines = load_existing_ids(answers_file)
    if existing_lines > 0:
        print(f"[Resume] Found existing results: {answers_file} with {existing_lines} lines. Will skip duplicates and append.")
    else:
        print(f"[Start Fresh] No existing results found at {answers_file} (will create).")

    ans_file = open(answers_file, "a", encoding="utf-8")

    skipped = 0
    written = 0

    for idx in tqdm(indices):
        task_type = dataset.data_list[idx]['task_type']
        rec_id = f"{task_type}_{idx}"
        if rec_id in existing_ids:
            skipped += 1
            continue

        sample = dataset[idx]
        question_text = sample["question"]
        choices, index2ans = parse_choices_from_question(question_text)
        if len(choices) == 0:
            choices = ["A", "B", "C", "D", "E"]
        answer_letter = letter_from_answer_string(sample["answer"])
        answer_id = answer_letter if answer_letter in choices else ""

        qs = question_text + "\nPlease answer directly with only the letter of the correct option and nothing else."
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        raw_frames = sample["video"]  # (T,H,W,3) uint8
        video = image_processor.preprocess(raw_frames, return_tensors='pt')['pixel_values'].half().cuda()
        video = [video]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        pred = parse_multi_choice_response(outputs, choices, index2ans)
        acc = str(pred == answer_id) if answer_id else "False"

        record = {
            "id": rec_id,
            "task_type": sample["task_type"],
            "question": question_text,
            "answer": sample["answer"],
            "answer_id": answer_id,
            "pred": outputs,
            "parsed_pred": pred,
            "acc": acc,
            "num_segments": args.topk,
        }
        ans_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        ans_file.flush()
        written += 1
        del output_ids
        del input_ids
        del stopping_criteria
        del video 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        import gc; gc.collect()

    ans_file.close()
    print(f"Done. Results saved to: {answers_file}")
    print(f"[Summary] skipped(existing)={skipped}, newly_written={written}, total_in_chunk={len(indices)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True,
                    help="Directory to save results.")
    parser.add_argument("--output-name", type=str, default="default")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--mvbench_json_dir", type=str, required=True, help="Path to MVBench json directory (e.g., .../json)")
    parser.add_argument("--data_prefix", type=str, required=True, help="Common prefix to replace your_data_path (e.g., /datasets)")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--topk", type=int, default=16, help="= MVBench_dataset.num_segments")

    args = parser.parse_args()
    eval_model(args)
