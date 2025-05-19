#!/usr/bin/env python3

import argparse
import math
import multiprocessing
import multiprocessing as mp
import os

import MatterSim
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm  # 替换掉 progressbar
from transformers import AutoImageProcessor, Dinov2Model

from utils import load_viewpoint_ids

# Set the start method to 'spawn' for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)

# 设置环境变量
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

# 视角 & 图片参数
VIEWPOINT_SIZE = 36  # 每个 viewpoint 采样 36 个视角
WIDTH = 640
HEIGHT = 480
VFOV = 60


# 颜色转换 (BGR -> RGB)
def BGR_to_RGB(image):
    image = image[:, :, ::-1]  # 交换通道顺序
    return image.copy()


# 初始化 DinoV2 模型
def build_feature_extractor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dinov2Model.from_pretrained("../dinov2-base").to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained("../dinov2-base", use_fast=True)
    return model, image_processor, device


# 初始化 MatterSim
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


# 存储特征文件
# 处理单个 viewpoint 的 36 张图像
def process_features(proc_id, out_queue, scanvp_list, args):
    print(f"进程 {proc_id} 开始...")

    # 手动分配显卡，显式指定使用的GPU设备
    gpu_list = [1, 2]  # 指定使用的GPU列表
    if proc_id < len(gpu_list):
        local_rank = gpu_list[proc_id]
    else:
        local_rank = gpu_list[0]  # 如果进程数超过显卡数，循环使用GPU
    print(f"进程 {proc_id} 使用的 GPU: {local_rank}")  # 打印分配的GPU

    # 设置CUDA_VISIBLE_DEVICES 环境变量确保进程只使用一个GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    torch.cuda.set_device(local_rank)

    # 初始化模型 & 模拟器
    sim = build_simulator(args.connectivity_dir, args.scan_dir)
    model, image_processor, device = build_feature_extractor()

    for scan_id, viewpoint_id in scanvp_list:
        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            # 读取图像 & 预处理
            image = np.array(state.rgb, copy=True)  # BGR
            image = BGR_to_RGB(image)  # 转换为 RGB
            image = Image.fromarray(image)
            inputs = image_processor(image, return_tensors="pt").to(device)
            images.append(inputs["pixel_values"])

        images = torch.cat(images, dim=0)  # 拼接 batch

        # 提取特征
        with torch.no_grad():
            fts = model(images).last_hidden_state  # 获取 36 个特征

            # 将特征从 GPU 转移到 CPU，并转换为 numpy 数组
            fts = fts.cpu().numpy()

        out_queue.put((scan_id, viewpoint_id, fts))

    out_queue.put(None)  # 结束信号


# 存储特征文件
def build_feature_file(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)

    num_finished_workers = 0
    num_finished_vps = 0
    progress_bar = tqdm(total=len(scanvp_list))  # 使用 tqdm
    progress_bar.set_description("Processing Viewpoints")  # 设置进度条描述

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts = res
                key = f"{scan_id}_{viewpoint_id}"

                outf.create_dataset(key, fts.shape, dtype='float', compression='gzip')
                outf[key][...] = fts
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                progress_bar.update(1)  # 更新进度条

    progress_bar.close()  # 关闭进度条
    for process in processes:
        process.join()


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', default='../datasets/R2R/connectivity')
    parser.add_argument('--scan_dir', default='../../Matterport3DSimulator/data/v1/scans')
    parser.add_argument('--output_file', default='../datasets/R2R/features/dino_features_36_vit_b14.hdf5')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    build_feature_file(args)
