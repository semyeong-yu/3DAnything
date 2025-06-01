import os
import torch
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import pandas as pd

from tqdm import tqdm

from dataclasses import dataclass

import numpy as np

import ray

import argparse

from typing import List

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

# image loader for llm captioning


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class LLMPack:
    def __init__(self, model_local_path):
        self.model_local_path = model_local_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_local_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            self.model_local_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval()
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)


@dataclass
class CaptioningConfig:
    dataset_root: str
    original_df_path: str
    model_name: str
    model_batchify_size: int
    shard_factor: int
    caption_csv_path: str = "./captions.csv"
    rgb_column_name: str = "image_render"
    question: str = '<image>\nPlease describe the image shortly.'


def elem_caption_function(target_df: pd.DataFrame, caption_cfg: CaptioningConfig, debug: bool = False):
    llm_pack = LLMPack(caption_cfg.model_name)

    batch_size = caption_cfg.model_batchify_size

    for start_idx in tqdm(range(0, len(target_df), batch_size)):

        if debug and start_idx > 1:
            break

        end_idx = min(start_idx + batch_size, len(target_df))
        batch_df = target_df.iloc[start_idx:end_idx]

        # 배치 내 이미지 경로 생성
        image_paths = [
            os.path.join(caption_cfg.dataset_root,
                         row['meta_class'], row['class'], row['dataset_type'], row['base_name'])
            for _, row in batch_df.iterrows()
        ]

        try:
            # 이미지 로드 및 픽셀 값 준비
            pixel_values_list = [load_image(path, max_num=12).to(
                torch.bfloat16).cuda() for path in image_paths]
            num_patches_list = [pv.size(0)
                                for pv in pixel_values_list]  # 각 이미지의 패치 수
            pixel_values = torch.cat(pixel_values_list, dim=0)  # 배치 텐서로 결합

            # LLM에 보낼 질문 준비
            questions = [caption_cfg.question] * len(image_paths)

            # 캡션 생성
            responses = llm_pack.model.batch_chat(
                llm_pack.tokenizer,
                pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=llm_pack.generation_config
            )

            # DataFrame에 캡션 업데이트
            target_df.loc[batch_df.index, 'caption'] = responses

        except Exception as e:
            print(f"배치 {start_idx // batch_size} 처리 중 오류 발생: {e}")
            # 오류 발생 시 캡션을 'Error'로 설정 (선택 사항)
            target_df.loc[batch_df.index, 'caption'] = 'Error'

    return target_df


def get_proc_df(caption_cfg: CaptioningConfig) -> List[pd.DataFrame]:
    df = pd.read_csv(caption_cfg.original_df_path)

    proc_df = df[df["dataset_type"].isin([caption_cfg.rgb_column_name])]
    proc_df.loc[:, 'caption'] = ""
    sharded_proc_df = np.array_split(proc_df, caption_cfg.shard_factor)
    
    return sharded_proc_df


def gather_save_df(sharded_proc_df: List[pd.DataFrame], caption_cfg: CaptioningConfig):
    # 각 프로세스에서 처리된 DataFrame을 모아서 하나의 DataFrame으로 결합
    combined_df = pd.concat(sharded_proc_df, ignore_index=True)

    # 결과를 CSV 파일로 저장
    os.makedirs(os.path.dirname(caption_cfg.caption_csv_path), exist_ok=True)
    output_path = os.path.join(caption_cfg.caption_csv_path)
    combined_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--dataset_root', type=str, default="/workspace/data/3ddst/train",
                      help='Root directory of the dataset')
    args.add_argument('--original_df_path', type=str, default="/workspace/code/3DAnything/debug/data/3ddst.csv",
                      help='Path to the original DataFrame CSV file')
    args.add_argument("--caption_csv_path", type=str, default="/workspace/code/3DAnything/debug/data/captions.csv",)
    args.add_argument('--loca_model_name', type=str, default="/workspace/weight/InternVL3-14B",
                      help='Path to the LLM model directory')
    args.add_argument('--model_batchify_size', type=int, default=32,
                      help='Batch size for processing images with the LLM model')
    args.add_argument(
        "--num_gpus", type=int, default=8,
    )
    args.add_argument("--debug", action="store_true",)
    args = args.parse_args()

    caption_cfg = CaptioningConfig(
        dataset_root=args.dataset_root,
        original_df_path=args.original_df_path,
        model_name=args.loca_model_name,
        caption_csv_path=args.caption_csv_path,
        shard_factor=args.num_gpus,
        model_batchify_size=args.model_batchify_size
    )

    shareded_proc_df = get_proc_df(caption_cfg)

    ray.init()
    elem_remote_caption_function = ray.remote(
        num_cpus=(os.cpu_count() // args.num_gpus), num_gpus=1)(elem_caption_function)

    features = []
    for i, proc_df in enumerate(shareded_proc_df):
        features.append(
            elem_remote_caption_function.remote(proc_df, caption_cfg, debug=args.debug))

    shareded_proc_df = ray.get(features)
    gather_save_df(shareded_proc_df, caption_cfg)
