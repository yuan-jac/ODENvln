#!/usr/bin/env python3

import argparse
import math
import os

import MatterSim
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model, AutoModelForImageClassification

from utils import load_viewpoint_ids

torch.set_float32_matmul_precision('high')  # 推荐 A800 设置

# 图像采样参数
VIEWPOINT_SIZE = 36
WIDTH = 640
HEIGHT = 480
VFOV = 60


def BGR_to_RGB(image):
    return image[:, :, ::-1].copy()


def build_feature_extractors():
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    backbone_model = Dinov2Model.from_pretrained("../dinov2-base").to(device).eval()
    classifier_model = AutoModelForImageClassification.from_pretrained(
        "../dinov2-with-registers-base-imagenet1k-1-layer").to(device).eval()
    processor = AutoImageProcessor.from_pretrained("../dinov2-base")

    # ✅ PyTorch 2.0+ 编译优化
    if torch.__version__ >= "2.0.0":
        backbone_model = torch.compile(backbone_model)
        classifier_model = torch.compile(classifier_model)

    return backbone_model, classifier_model, processor, device


def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


def extract_features(scanvp_list, args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 构建模型和仿真器
    sim = build_simulator(args.connectivity_dir, args.scan_dir)
    backbone_model, classifier_model, processor, device = build_feature_extractors()

    with h5py.File(args.output_file, 'w') as outf:
        for scan_id, viewpoint_id in tqdm(scanvp_list, desc="提取特征中"):
            images = []
            for ix in range(VIEWPOINT_SIZE):
                if ix == 0:
                    sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    sim.makeAction([0], [1.0], [1.0])
                else:
                    sim.makeAction([0], [1.0], [0])
                state = sim.getState()[0]
                image = BGR_to_RGB(np.array(state.rgb, copy=True))
                image = Image.fromarray(image)
                images.append(image)

            # 分 batch 处理
            all_features = []
            for i in range(0, len(images), args.batch_size):
                batch_images = images[i:i + args.batch_size]
                inputs = processor(images=batch_images, return_tensors="pt").to(device)

                # ✅ 使用 channels_last 可选优化
                inputs = {
                    k: v.to(memory_format=torch.channels_last) if v.ndim == 4 else v
                    for k, v in inputs.items()
                }

                with torch.no_grad():
                    feat_768 = backbone_model(**inputs).last_hidden_state[:, 0]
                    feat_1000 = classifier_model(**inputs).logits
                    feat_concat = torch.cat([feat_768, feat_1000], dim=-1)  # (batch, 1768)

                all_features.append(feat_concat)

            all_features = torch.cat(all_features, dim=0).cpu().numpy()  # (36, 1768)

            key = f"{scan_id}_{viewpoint_id}"
            outf.create_dataset(key, data=all_features, compression='gzip')
            outf[key].attrs['scanId'] = scan_id
            outf[key].attrs['viewpointId'] = viewpoint_id
            outf[key].attrs['image_w'] = WIDTH
            outf[key].attrs['image_h'] = HEIGHT
            outf[key].attrs['vfov'] = VFOV


if __name__ == '__main__':
    # ✅ 启用 cudnn 加速优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', default='../datasets/R2R/connectivity')
    parser.add_argument('--scan_dir', default='../../Matterport3DSimulator/data/v1/scans')
    parser.add_argument('--output_file', default='../datasets/R2R/features/dino_features_36_1768dim.hdf5')
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)
    extract_features(scanvp_list, args)
