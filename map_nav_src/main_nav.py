import json
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

from map_nav_src.utils.data import ImageFeaturesDB
from map_nav_src.utils.distributed import all_gather, merge_dist_results
from map_nav_src.utils.distributed import is_default_gpu
from map_nav_src.utils.logger import write_to_record_file, print_progress, timeSince
from map_nav_src.utils.misc import set_random_seed
from models.vlnbert_init import get_tokenizer
from r2r.agent import GMapNavAgent
from r2r.data_utils import construct_instrs
from r2r.env import R2RNavBatch
from r2r.parser import parse_args


def build_dataset(args, rank=0, is_test=False):
    """
    构建训练和验证数据集。

    参数:
        args: 参数对象，包含配置信息。
        rank: 当前进程的排名（分布式训练中使用）。
        is_test: 是否为测试模式。

    返回:
        train_env: 训练环境。
        val_envs: 验证环境字典。
        aug_env: 数据增强环境（可选）。
    """
    # 获取分词器
    tok = get_tokenizer(args)  # 获取分词器，用于处理指令文本

    # 判断是否为训练模式
    is_train = not is_test
    # 加载图像特征数据库
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, is_train)  # 加载图像特征数据库

    # 数据集类
    dataset_class = R2RNavBatch  # 数据集类，用于处理导航任务的数据

    # 数据增强环境
    if args.aug is not None:
        # 构建数据增强指令数据
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug],  # 数据增强的数据集名称
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,  # 指令的最大长度
            is_test=is_test
        )
        # 创建数据增强环境
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir,  # 连接性信息目录
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size,  # 批量大小和角度特征大小
            seed=args.seed + rank, sel_data_idxs=None, name='aug',  # 随机种子和数据索引
        )
    else:
        aug_env = None  # 如果没有数据增强，则设置为 None

    # 训练数据集
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'],  # 训练数据集名称
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,  # 指令的最大长度
        is_test=is_test
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir,  # 连接性信息目录
        batch_size=args.batch_size,  # 批量大小
        angle_feat_size=args.angle_feat_size, seed=args.seed + rank,  # 角度特征大小和随机种子
        sel_data_idxs=None, name='train',  # 数据索引和环境名称
    )

    # 验证数据集
    val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']  # 验证数据集名称
    if args.submit:
        val_env_names.append('test')  # 如果提交测试结果，则添加测试数据集

    val_envs = {}
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, is_train=False)  # 加载验证集的图像特征数据库
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split],  # 验证数据集名称
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,  # 指令的最大长度
            is_test=is_test
        )

        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir,  # 连接性信息目录
            batch_size=args.batch_size,  # 批量大小
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,  # 角度特征大小和随机种子
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size),  # 数据索引
            name=split,  # 环境名称
        )  # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    """
    训练模型。

    参数:
        args: 参数对象，包含配置信息。
        train_env: 训练环境。
        val_envs: 验证环境字典。
        aug_env: 数据增强环境（可选）。
        rank: 当前进程的排名（分布式训练中使用）。
    """
    # 检查是否为默认 GPU
    default_gpu = is_default_gpu(args)

    # 如果是默认 GPU，初始化日志文件和 TensorBoard
    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)  # 保存训练参数到文件
        writer = SummaryWriter(log_dir=args.log_dir)  # 初始化 TensorBoard
        record_file = os.path.join(args.log_dir, 'train.txt')  # 训练记录文件
        write_to_record_file(str(args) + '\n\n', record_file)  # 写入训练参数

    # 初始化代理类
    agent_class = GMapNavAgent
    listner = agent_class(args, train_env, rank=rank)  # 创建代理对象

    # 如果有恢复文件，加载模型
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))  # 加载模型
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter),
                record_file
            )  # 写入恢复信息

    # 首次验证
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            listner.test(use_dropout=False, feedback='argmax', iters=None)  # 测试模式
            preds = listner.get_results()  # 获取预测结果
            preds = merge_dist_results(all_gather(preds))  # 合并分布式结果
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)  # 计算验证指标
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)  # 写入验证结果

    # 记录训练开始时间
    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )  # 写入训练开始信息

    # 初始化最佳验证结果
    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state": ""}}

    # 训练循环
    for idx in range(start_iter, start_iter + args.iters, args.log_every):
        listner.logs = defaultdict(list)  # 初始化日志
        interval = min(args.log_every, args.iters - idx)  # 计算训练间隔
        iter = idx + interval  # 当前迭代次数

        # 训练
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # 使用训练数据训练
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # 使用 GT 数据训练
                listner.env = train_env
                listner.train(1, feedback=args.feedback)
                # 使用增强数据训练
                listner.env = aug_env
                listner.train(1, feedback=args.feedback)
                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        # 记录训练统计信息
        if default_gpu:
            total = max(sum(listner.logs['total']), 1)  # 总有效动作数
            length = max(len(listner.logs['critic_loss']), 1)  # 批量最大长度
            critic_loss = sum(listner.logs['critic_loss']) / total  # 批量损失
            policy_loss = sum(listner.logs['policy_loss']) / total  # 策略损失
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)  # 强化学习损失
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)  # 监督学习损失
            entropy = sum(listner.logs['entropy']) / total  # 熵
            writer.add_scalar("loss/critic", critic_loss, idx)  # 写入 TensorBoard
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss),
                record_file
            )  # 写入训练记录

        # 验证
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            listner.env = env
            listner.test(use_dropout=False, feedback='argmax', iters=None)  # 测试模式
            preds = listner.get_results()  # 获取预测结果
            preds = merge_dist_results(all_gather(preds))  # 合并分布式结果
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)  # 计算验证指标
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)
                # 选择最佳模型
                if env_name in best_val:
                    if score_summary['spl'] >= best_val[env_name]['spl']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))

        # 保存模型
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))
            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter) / args.iters), iter, float(iter) / args.iters * 100,
                                      loss_str)),
                record_file
            )  # 写入训练进度
            write_to_record_file("BEST RESULT TILL NOW", record_file)  # 写入最佳结果
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)  # 写入最佳结果


def valid(args, train_env, val_envs, rank=-1):
    """
    验证模型。

    参数:
        args: 参数对象，包含配置信息。
        train_env: 训练环境。
        val_envs: 验证环境字典。
        rank: 当前进程的排名（分布式训练中使用）。
    """
    # 检查是否为默认 GPU
    default_gpu = is_default_gpu(args)

    # 初始化代理类
    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)  # 创建代理对象

    # 如果有恢复文件，加载模型
    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))  # 加载模型并打印信息

    # 如果是默认 GPU，初始化日志文件
    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)  # 保存验证参数到文件
        record_file = os.path.join(args.log_dir, 'valid.txt')  # 验证记录文件
        write_to_record_file(str(args) + '\n\n', record_file)  # 写入验证参数

    # 遍历每个验证环境
    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'  # 根据参数选择输出前缀
        # if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
        #    continue  # 如果文件已存在，则跳过

        agent.logs = defaultdict(list)  # 初始化日志
        agent.env = env  # 设置当前环境

        iters = None  # 迭代次数
        start_time = time.time()  # 记录开始时间
        agent.test(use_dropout=False, feedback='argmax', iters=iters)  # 测试模式
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))  # 打印耗时
        preds = agent.get_results(detailed_output=args.detailed_output)  # 获取预测结果
        preds = merge_dist_results(all_gather(preds))  # 合并分布式结果

        # 如果是默认 GPU，记录验证结果
        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)  # 计算验证指标
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str + '\n', record_file)  # 写入验证结果

            # 如果需要提交结果，保存预测结果到文件
            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name)), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )


def main():
    """
    主程序入口。
    """
    # 解析命令行参数
    args = parse_args()

    # 初始化分布式训练
    torch.distributed.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = torch.device(device_id)
    torch.cuda.set_device(device)  # 设置当前 GPU 设备

    # 设置随机种子
    set_random_seed(args.seed + rank)

    # 构建数据集
    train_env, val_envs, aug_env = build_dataset(args, rank=rank, is_test=args.test)

    # 根据参数决定是训练还是验证
    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank)  # 训练模式
    else:
        valid(args, train_env, val_envs, rank=rank)  # 验证模式


if __name__ == '__main__':
    main()
