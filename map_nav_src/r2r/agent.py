import math
import sys

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from map_nav_src.models.graph_utils import GraphMap
from map_nav_src.models.model import VLNBert, Critic
from map_nav_src.models.ops import pad_tensors_wgrad
from map_nav_src.utils.ops import pad_tensors, gen_seq_masks
from .agent_base import Seq2SeqAgent


class GMapNavAgent(Seq2SeqAgent):
    """
    GMapNavAgent 类，用于基于图的导航任务。
    继承自 Seq2SeqAgent，实现了导航任务中的模型构建、特征提取、导航策略等功能。
    """
    def _build_model(self):
        """
        构建模型，初始化 VLNBert 和 Critic 模型。
        """
        self.vln_bert = VLNBert(self.args).cuda()  # 初始化 VLNBert 模型并移至 GPU
        self.critic = Critic(self.args).cuda()  # 初始化 Critic 模型并移至 GPU
        # 缓存变量
        self.scanvp_cands = {}

    def _language_variable(self, obs):
        """
        处理语言输入，将指令编码为张量。
        :param obs: 观测数据列表，包含每个样本的指令编码。
        :return: 包含文本 ID 和掩码的字典。
        """
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]  # 获取每个指令的长度

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)  # 初始化指令张量
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)  # 初始化掩码
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']  # 填充指令编码
            mask[i, :seq_lengths[i]] = True  # 设置掩码

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()  # 转换为张量并移至 GPU
        mask = torch.from_numpy(mask).cuda()  # 转换为张量并移至 GPU
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask  # 返回文本 ID 和掩码
        }

    def _panorama_feature_variable(self, obs):
        """
        提取全景特征，包括图像特征和位置特征。
        :param obs: 观测数据列表，包含每个样本的全景特征。
        :return: 包含全景特征的字典。
        """
        # 初始化用于存储批次数据的列表
        batch_view_img_fts,batch_view_text_fts, batch_loc_fts, batch_nav_types =[], [], [], []  # 图像特征、图像描述特征，位置特征和导航类型
        batch_view_lens, batch_cand_vpids = [], []  # 视图长度和候选视图 ID 列表

        # 遍历每个观测数据
        for i, ob in enumerate(obs):
            view_text_fts,view_img_fts, view_ang_fts, nav_types, cand_vpids = [],[], [], [], []  # 初始化当前样本的特征列表
            used_viewidxs = set()  # 用于记录已使用的视图索引

            # 提取候选视图特征
            for j, cc in enumerate(ob['candidate']):
                # 提取图像特征（前 self.args.image_feat_size 维）
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                # 提取角度特征（剩余部分）
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                # 提取图像描述特征
                view_text_fts.append(cc['objecttext_feat'])
                # 标记为候选视图（导航类型为 1）
                nav_types.append(1)
                # 记录候选视图的 ID
                cand_vpids.append(cc['viewpointId'])
                # 记录已使用的视图索引
                used_viewidxs.add(cc['pointId'])

            # 提取非候选视图特征
            # 遍历全景特征，提取未被标记为候选视图的特征
            view_img_fts.extend(
                [x[:self.args.image_feat_size] for k, x in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend(
                [x[self.args.image_feat_size:] for k, x in enumerate(ob['feature']) if k not in used_viewidxs])
            view_text_fts.extend(
                [x for k, x in enumerate(ob['objecttext_feat']) if k not in used_viewidxs]
            )
            # 补齐导航类型为 0（表示非候选视图）
            nav_types.extend([0] * (36 - len(used_viewidxs)))

            # 合并候选视图和非候选视图的特征
            view_img_fts = np.stack(view_img_fts, 0)  # 将图像特征堆叠成一个 NumPy 数组 (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)  # 将角度特征堆叠成一个 NumPy 数组
            view_text_fts = np.stack(view_text_fts, 0)  # 将图像描述特征堆叠成一个 NumPy 数组
            # 创建一个简单的框特征（固定值 [1, 1, 1]，表示每个视图的边界框）
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            # 将角度特征和框特征拼接成位置特征
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            # 将当前样本的特征添加到批次列表
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))  # 转换为 PyTorch 张量
            batch_view_text_fts.append(torch.from_numpy(view_text_fts))  # 转换为 PyTorch 张量
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))  # 转换为 PyTorch 张量
            batch_nav_types.append(torch.LongTensor(nav_types))  # 转换为 PyTorch 张量
            batch_cand_vpids.append(cand_vpids)  # 添加候选视图 ID 列表
            batch_view_lens.append(len(view_img_fts))  # 记录当前样本的视图数量

        # 对批次数据进行填充，以对齐长度
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()  # 填充图像特征并移至 GPU
        batch_view_text_fts = pad_tensors(batch_view_text_fts).cuda()  # 填充图像描述特征并移至 GPU
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()  # 填充位置特征并移至 GPU
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()  # 填充导航类型并移至 GPU
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()  # 转换为 PyTorch 张量并移至 GPU

        # 返回包含全景特征的字典
        return {
            'view_img_fts': batch_view_img_fts,  # 图像特征
            'view_text_fts': batch_view_text_fts,  # 图像描述特征
            'loc_fts': batch_loc_fts,  # 位置特征
            'nav_types': batch_nav_types,  # 导航类型
            'view_lens': batch_view_lens,  # 视图长度
            'cand_vpids': batch_cand_vpids,  # 候选视图 ID 列表
        }

    def _nav_gmap_variable(self, obs, gmaps):
        """
        构建导航图（Graph Map）相关的变量。
        该函数用于生成导航图的节点特征、位置特征、访问掩码、节点间距离矩阵等信息，
        用于后续的导航策略计算。

        :param obs: 观测数据列表，每个元素是一个字典，包含当前环境的观测信息。
        :param gmaps: 图结构列表，每个元素是一个 GraphMap 对象，表示当前导航任务的图结构。
        :return: 一个字典，包含导航图相关的变量。
        """
        # 获取批次大小
        batch_size = len(obs)

        # 初始化用于存储批次数据的列表
        batch_gmap_vpids, batch_gmap_lens = [], []  # 节点 ID 列表和长度
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []  # 节点图像嵌入、步数 ID 和位置特征
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []  # 节点间距离矩阵和访问掩码
        batch_no_vp_left = []  # 标记是否没有未访问的节点

        # 遍历每个图结构
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []  # 初始化已访问和未访问的节点 ID 列表

            # 遍历图中的所有节点
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:  # 如果启用行为访问节点
                    if k == obs[i]['viewpoint']:  # 当前视点视为已访问
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:  # 否则根据图的访问状态判断
                    if gmap.graph.visited(k):  # 如果节点已访问
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)

            # 记录是否没有未访问的节点
            batch_no_vp_left.append(len(unvisited_vpids) == 0)

            # 根据参数决定是否编码完整图
            if self.args.enc_full_graph:
                # 包含已访问和未访问的节点
                gmap_vpids = [None] + visited_vpids + unvisited_vpids  # 添加 [stop] 节点
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)  # 访问掩码
            else:
                # 只包含未访问的节点
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            # 获取每个节点的步数 ID
            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]

            # 获取每个节点的图像嵌入
            if len(gmap_vpids[1:]) > 0:  # 如果有节点
                gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
                gmap_img_embeds = torch.stack(
                    [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
                )  # 添加 [stop] 节点的嵌入
            else:
                print("gmap_vpids[1:] is empty, skipping embedding processing.")
                # 如果没有节点，打印警告信息（可以在此处添加默认值或跳过处理）

            # 获取每个节点的位置特征
            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            # 初始化节点间距离矩阵
            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)

            # 计算节点间的距离
            for i in range(1, len(gmap_vpids)):
                for j in range(i + 1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            # 将当前图的特征和掩码添加到批次列表
            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # 对批次数据进行整理和填充
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)  # 转换为张量
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()  # 生成图掩码并移至 GPU
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)  # 填充图像嵌入
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()  # 填充步数 ID 并移至 GPU
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()  # 填充位置特征并移至 GPU
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()  # 填充访问掩码并移至 GPU

        # 处理节点间距离矩阵
        max_gmap_len = max(batch_gmap_lens)  # 获取最大图长度
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()  # 初始化填充后的距离矩阵
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]  # 填充距离矩阵
        gmap_pair_dists = gmap_pair_dists.cuda()  # 移至 GPU

        # 返回导航图相关的变量
        return {
            'gmap_vpids': batch_gmap_vpids,  # 节点 ID 列表
            'gmap_img_embeds': batch_gmap_img_embeds,  # 节点图像嵌入
            'gmap_step_ids': batch_gmap_step_ids,  # 节点步数 ID
            'gmap_pos_fts': batch_gmap_pos_fts,  # 节点位置特征
            'gmap_visited_masks': batch_gmap_visited_masks,  # 节点访问掩码
            'gmap_pair_dists': gmap_pair_dists,  # 节点间距离矩阵
            'gmap_masks': batch_gmap_masks,  # 图掩码
            'no_vp_left': batch_no_vp_left,  # 是否没有未访问的节点
            'grid_fts': [obs[index]['grid_fts'].cuda() for index in range(len(obs))],  # 网格特征
            'grid_map': [obs[index]['grid_map'].cuda() for index in range(len(obs))],  # 网格地图
            'gridmap_pos_fts': torch.cat(
                [obs[index]['gridmap_pos_fts'].unsqueeze(0).cuda() for index in range(len(obs))], dim=0)  # 网格位置特征
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        """
        构建导航任务中视点（Viewpoint）相关的变量。
        该函数用于生成视点的图像嵌入、位置特征、导航掩码等信息，
        用于后续的导航策略计算。

        :param obs: 观测数据列表，每个元素是一个字典，包含当前环境的观测信息。
        :param gmaps: 图结构列表，每个元素是一个 GraphMap 对象，表示当前导航任务的图结构。
        :param pano_embeds: 全景图像的嵌入表示，形状为 (batch_size, num_views, embed_dim)。
        :param cand_vpids: 候选视点的 ID 列表，每个样本对应一个列表。
        :param view_lens: 每个样本的视图数量。
        :param nav_types: 导航类型张量，形状为 (batch_size, num_views)，候选视点为 1，非候选视点为 0。
        :return: 一个字典，包含视点相关的变量。
        """
        batch_size = len(obs)  # 获取批次大小

        # 添加 [stop] 令牌到图像嵌入中
        # 创建一个形状与 pano_embeds[:1] 相同的零张量，并将其与 pano_embeds 拼接
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []  # 初始化用于存储批次位置特征的列表
        # 遍历每个图结构和观测数据
        for i, gmap in enumerate(gmaps):
            # 获取当前候选视点的位置特征
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i],
                obs[i]['heading'], obs[i]['elevation']
            )
            # 获取起始视点的位置特征
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp],
                obs[i]['heading'], obs[i]['elevation']
            )
            # 初始化视点位置特征矩阵，形状为 (num_views+1, 14)，其中 num_views 是视图数量
            # 前 7 维用于存储起始视点特征，后 7 维用于存储候选视点特征
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts  # 填充起始视点特征
            # 填充候选视点特征，从索引 1 开始，因为索引 0 保留给 [stop] 令牌
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            # 将当前样本的位置特征添加到批次列表中
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        # 对批次位置特征进行填充，确保所有样本的长度一致，并移至 GPU
        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        # 构建导航掩码
        # 创建一个形状为 (batch_size, 1) 的全为 True 的张量，表示 [stop] 令牌
        # 并将其与 nav_types == 1 的掩码拼接，后者表示候选视点
        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        # 返回包含视点相关变量的字典
        return {
            'vp_img_embeds': vp_img_embeds,  # 视点的图像嵌入
            'vp_pos_fts': batch_vp_pos_fts,  # 视点的位置特征
            'vp_masks': gen_seq_masks(view_lens + 1),  # 视点的掩码，用于处理不同长度的视点序列
            'vp_nav_masks': vp_nav_masks,  # 导航掩码，用于区分候选视点和非候选视点
            'vp_cand_vpids': [[None] + x for x in cand_vpids],  # 候选视点的 ID 列表，包含 [stop] 令牌
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        提取教师动作。
        根据当前观测数据和目标路径，计算出教师动作。
        教师动作是指在监督学习中，根据已知的最佳路径（ground truth path）来指导智能体选择下一个动作。

        :param obs: 观测数据列表，每个元素是一个字典，包含当前环境的观测信息。
        :param vpids: 候选视点的 ID 列表，每个样本对应一个列表。
        :param ended: 一个布尔数组，表示每个样本的动作序列是否已经结束。
        :param visited_masks: 可选参数，表示每个候选视点是否已经被访问过。如果为 None，则不考虑访问状态。
        :return: 一个张量，表示每个样本的教师动作。
        """
        # 初始化动作数组，所有动作初始值为 0
        a = np.zeros(len(obs), dtype=np.int64)
        # 遍历每个观测数据
        for i, ob in enumerate(obs):
            # 如果当前样本的动作序列已经结束，则设置为忽略的动作 ID
            if ended[i]:
                a[i] = self.args.ignoreid
            # 如果当前视点已经是目标路径的最后一个视点，则设置动作为停止（动作索引为 0）
            elif ob['viewpoint'] == ob['gt_path'][-1]:
                a[i] = 0
                # 否则，计算最优动作
            else:
                # 获取当前环境的扫描 ID
                scan = ob['scan']
                # 获取当前视点的 ID
                cur_vp = ob['viewpoint']
                # 初始化最小索引和最小距离
                min_idx, min_dist = self.args.ignoreid, float('inf')
                # 遍历当前样本的候选视点
                for j, vpid in enumerate(vpids[i]):
                    # 如果候选视点未被访问（或者未提供访问掩码），则进行计算
                    if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                        # 计算从当前视点到候选视点，再到目标路径最后一个视点的总距离
                        dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                               + self.env.shortest_distances[scan][cur_vp][vpid]
                        # 如果当前距离更小，则更新最小距离和最小索引
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = j
                # 将最小索引作为当前样本的教师动作
                a[i] = min_idx
                # 如果最小索引仍然是忽略的动作 ID，说明没有找到有效的动作，打印警告信息
                if min_idx == self.args.ignoreid:
                    print('scan %s: all vps are searched' % (scan))

        # 将动作数组转换为 PyTorch 张量，并移至 GPU
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        将全景视图中的动作转换为等效的以自我为中心（Egocentric）视图中的动作。
        这是全景视图和以自我为中心视图之间的接口。
        它会将全景视图中的动作 a_t 转换为等效的以自我为中心视图中的动作，以便在模拟器中执行。

        :param a_t: 当前动作，是一个列表，每个元素对应一个样本的动作索引。
        :param gmaps: 图结构列表，每个元素是一个 GraphMap 对象，表示当前导航任务的图结构。
        :param obs: 观测数据列表，每个元素是一个字典，包含当前环境的观测信息。
        :param traj: 可选参数，表示每个样本的导航轨迹。如果提供，则更新轨迹信息。
        """
        for i, ob in enumerate(obs):  # 遍历每个观测数据
            action = a_t[i]  # 获取当前样本的动作
            if action is not None:  # 如果动作不是 None（None 表示停止动作）
                # 将当前动作对应的路径添加到轨迹中
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                # 获取上一个视点
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                # 获取当前动作对应的视图索引
                viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][action]
                # 计算水平方向（heading）和垂直方向（elevation）的角度
                heading = (viewidx % 12) * math.radians(30)  # 水平方向角度
                elevation = (viewidx // 12 - 1) * math.radians(30)  # 垂直方向角度
                # 在模拟器中执行新的动作
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        """
        更新扫描视点候选字典。
        该函数用于更新每个扫描环境中的视点候选信息，以便后续的动作转换。

        :param obs: 观测数据列表，每个元素是一个字典，包含当前环境的观测信息。
        """
        for ob in obs:  # 遍历每个观测数据
            scan = ob['scan']  # 获取当前扫描环境的 ID
            vp = ob['viewpoint']  # 获取当前视点的 ID
            scanvp = '%s_%s' % (scan, vp)  # 构造扫描视点的唯一标识
            self.scanvp_cands.setdefault(scanvp, {})  # 如果该扫描视点不存在，则初始化为空字典
            for cand in ob['candidate']:  # 遍历当前视点的候选视点
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})  # 初始化候选视点
                # 更新候选视点的索引信息
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True):
        """
        执行导航任务的 rollout 过程。
        该函数用于在环境中执行智能体的导航策略，可以用于训练（监督学习或强化学习）或测试阶段。

        :param train_ml: 监督学习（Maximum Likelihood）训练的权重。如果为 None，则不进行监督学习。
        :param train_rl: 是否进行强化学习（Reinforcement Learning）训练。
        :param reset: 是否重置环境。如果为 True，则调用 self.env.reset() 重新开始一个新任务。
        :return: 返回导航轨迹 traj，包含每个样本的路径信息。
        """
        # 重置环境或获取当前观测
        if reset:  # 如果需要重置环境
            obs = self.env.reset()  # 重置环境并获取初始观测
        else:
            obs = self.env._get_obs()  # 获取当前观测
        self._update_scanvp_cands(obs)  # 更新扫描视点候选信息

        batch_size = len(obs)  # 获取批次大小
        # 构建图结构，初始化每个样本的图
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)  # 更新图结构

        # 记录导航路径
        traj = [{
            'instr_id': ob['instr_id'],  # 指令 ID
            'path': [[ob['viewpoint']]],  # 当前路径
            'details': {},  # 详细信息
        } for ob in obs]

        # 处理语言输入
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)  # 获取语言嵌入

        # 初始化跟踪状态
        ended = np.array([False] * batch_size)  # 是否结束
        just_ended = np.array([False] * batch_size)  # 是否刚刚结束

        # 初始化日志
        masks = []
        entropys = []
        ml_loss = 0.  # 监督学习损失

        # 主循环：导航过程
        for t in range(self.args.max_action_len):  # 最大动作长度
            for i, gmap in enumerate(gmaps):
                if not ended[i]:  # 如果当前样本未结束
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1  # 更新节点步数

            # 提取全景特征
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)  # 计算平均全景嵌入

            # 更新图结构
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)  # 更新已访问节点
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):  # 更新未访问节点
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # 导航策略部分
            # 构建导航输入，包括图结构和视点信息
            nav_inputs = self._nav_gmap_variable(obs, gmaps)  # 获取基于图的导航输入
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )  # 更新视点相关的导航输入
            nav_inputs.update({
                'txt_embeds': txt_embeds,  # 添加语言嵌入
                'txt_masks': language_inputs['txt_masks'],  # 添加语言掩码
            })
            nav_outs = self.vln_bert('navigation', nav_inputs)  # 使用VLN-BERT模型获取导航输出

            # 根据融合策略选择导航日志和视点ID
            if self.args.fusion == 'local':  # 如果使用局部融合
                nav_logits = nav_outs['local_logits']  # 使用局部导航日志
                nav_vpids = nav_inputs['vp_cand_vpids']  # 使用候选视点ID
            elif self.args.fusion == 'global':  # 如果使用全局融合
                nav_logits = nav_outs['global_logits']  # 使用全局导航日志
                nav_vpids = nav_inputs['gmap_vpids']  # 使用全局图视点ID
            else:  # 如果使用混合融合
                nav_logits = nav_outs['fused_logits']  # 使用混合导航日志
                nav_vpids = nav_inputs['gmap_vpids']  # 使用全局图视点ID

            grid_logits = nav_outs['grid_logits']  # 获取网格导航日志
            nav_probs = torch.softmax(nav_logits, 1)  # 计算导航概率

            # 更新图结构中的停止分数
            for i, gmap in enumerate(gmaps):
                if not ended[i]:  # 如果当前样本未结束
                    i_vp = obs[i]['viewpoint']  # 当前视点
                    gmap.node_stop_scores[i_vp] = {  # 更新当前视点的停止分数
                        'stop': nav_probs[i, 0].data.item(),  # 停止概率
                    }

            # 如果进行监督学习
            if train_ml is not None:
                # 获取教师动作
                nav_targets = self._teacher_action(
                    obs, nav_vpids, ended,
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
                )
                # 计算监督学习损失
                ml_loss += self.criterion(nav_logits, nav_targets)

            # 确定下一个导航视点
            if self.feedback == 'teacher':  # 如果使用教师强制
                a_t = nav_targets  # 使用教师动作
            elif self.feedback == 'argmax':  # 如果使用学生强制 - argmax
                _, a_t = nav_logits.max(1)  # 选择概率最高的动作
                a_t = a_t.detach()
            elif self.feedback == 'sample':  # 如果使用采样
                c = torch.distributions.Categorical(nav_probs)  # 定义分类分布
                self.logs['entropy'].append(c.entropy().sum().item())  # 记录熵
                entropys.append(c.entropy())  # 用于优化
                a_t = c.sample().detach()  # 采样动作
            elif self.feedback == 'expl_sample':  # 如果使用探索采样
                _, a_t = nav_probs.max(1)  # 选择概率最高的动作
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # 探索概率
                if self.args.fusion == 'local':  # 如果使用局部融合
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()  # 获取导航掩码
                else:  # 如果使用全局融合
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs[
                        'gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:  # 如果进行探索
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]  # 获取候选动作
                        a_t[i] = np.random.choice(cand_a_t)  # 随机选择一个动作
            else:  # 如果反馈方式无效
                print(self.feedback)
                sys.exit('Invalid feedback option')  # 退出程序
            # Determine stop actions
            # 判断是否应该停止当前的导航动作
            if self.feedback == 'teacher' or self.feedback == 'sample':  # 如果处于训练阶段（教师强制或采样）
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]  # 判断当前视点是否在目标结束视点列表中
                # 修改为判断当前视点是否是目标路径的最后一个视点
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                # 如果不是训练阶段，则停止动作是选择的动作索引为0（即停止动作）
                a_t_stop = a_t == 0

            # Prepare environment action
            # 准备环境动作，即根据当前的动作选择来确定下一步的导航视点
            cpu_a_t = []  # 初始化环境动作列表
            for i in range(batch_size):
                # 如果当前样本满足以下任一条件，则将其动作设置为None，表示停止导航
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True  # 标记为刚刚结束
                else:
                    # 否则，选择对应的动作索引对应的视点作为下一步的导航视点
                    cpu_a_t.append(nav_vpids[i][a_t[i]])

            # Make action and get the new state
            # 执行动作并获取新的状态
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)  # 调用make_equiv_action函数执行动作
            for i in range(batch_size):
                # 如果当前样本未结束且刚刚结束，则进行以下操作
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}  # 初始化停止节点和停止分数
                    # 遍历当前样本的图结构中的停止分数
                    for k, v in gmaps[i].node_stop_scores.items():
                        # 如果当前节点的停止分数大于已记录的最大停止分数，则更新停止节点和停止分数
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    # 如果找到停止节点且当前视点不是停止节点，则将停止节点添加到路径中
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    # 如果需要详细输出，则记录每个节点的停止概率
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update graph
            # 获取新的观测并更新图结构
            obs = self.env._get_obs()  # 获取新的观测
            self._update_scanvp_cands(obs)  # 更新扫描视点候选信息
            for i, ob in enumerate(obs):
                # 如果当前样本未结束，则更新其图结构
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            # 更新结束状态
            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            # 如果所有样本都已结束，则提前退出循环
            if ended.all():
                break

            # 如果进行监督学习
            if train_ml is not None:
                # 计算平均监督学习损失
                ml_loss = ml_loss * train_ml / batch_size
                self.loss += ml_loss  # 累加损失
                self.logs['IL_loss'].append(ml_loss.item())  # 记录损失

            # 返回导航轨迹
        return traj