from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel

from .ops import pad_tensors_wgrad, gen_seq_masks
from .vilmodel import BertLayerNorm, BertOnlyMLMHead, GlocalTextPathCMT


class RegionClassification(nn.Module):
    """用于MRC（Masked Region Classification）任务的区域分类器"""
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        """
        参数:
            hidden_size: 输入特征的维度
            label_dim: 输出分类的维度（通常是图像区域类别数）
        """
        super().__init__()
        # 定义分类网络结构
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),  # 全连接层
                                 nn.ReLU(),  # 激活函数
                                 BertLayerNorm(hidden_size, eps=1e-12),  # BERT风格的层归一化
                                 nn.Linear(hidden_size, label_dim))  # 输出分类层

    def forward(self, input_):
        """前向传播
                参数:
                    input_: 输入特征 [batch_size, hidden_size]
                返回:
                    output: 分类logits [batch_size, label_dim]
                """
        output = self.net(input_)
        return output

class ClsPrediction(nn.Module):
    """通用的分类预测头，用于SAP和OG等任务"""
    def __init__(self, hidden_size, input_size=None):
        """
        参数:
            hidden_size: 隐藏层维度
            input_size: 可选，输入特征维度（默认与hidden_size相同）
        """
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),  # 适配输入维度
                                 nn.ReLU(),  # 激活函数
                                 BertLayerNorm(hidden_size, eps=1e-12),  # 层归一化
                                 nn.Linear(hidden_size, 1))  # 二分类输出

    def forward(self, x):
        """前向传播
            参数:
                x: 输入特征 [batch_size, input_size]
            返回:
                预测分数 [batch_size, 1]
        """
        return self.net(x)

class GlocalTextPathCMTPreTraining(BertPreTrainedModel):
    """全局-局部文本路径跨模态Transformer预训练模型"""
    def __init__(self, config):
        """初始化模型
        参数:
            config: 模型配置对象
        """
        super().__init__(config)

        self.config = config
        # 主干网络（多模态编码器）
        self.bert = GlocalTextPathCMT(config)
        # 根据预训练任务初始化不同头部
        self.global_sap_head = ClsPrediction(self.config.hidden_size)

        if 'mlm' in config.pretrain_tasks:
            # 掩码语言建模头
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'mrc' in config.pretrain_tasks:
            # 图像区域分类器
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
            # 对象分类器（可选）
            if self.config.obj_prob_size > 0 and self.config.obj_prob_size != self.config.image_prob_size:
                self.obj_classifier = RegionClassification(self.config.hidden_size, self.config.obj_prob_size)
            else:
                self.obj_classifier = None
        if 'sap' in config.pretrain_tasks:
            # 导航动作预测头
            self.global_sap_head = ClsPrediction(self.config.hidden_size)  # 全局动作
            self.local_sap_head = ClsPrediction(self.config.hidden_size)  # 局部动作
            self.grid_sap_head = ClsPrediction(self.config.hidden_size)  # 网格动作
            # 全局-局部融合层
            if config.glocal_fuse:
                self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size * 2)
            else:
                self.sap_fuse_linear = None
        if 'og' in config.pretrain_tasks:
            # 对象接地（Object Grounding）头
            self.og_head = ClsPrediction(self.config.hidden_size)
        # 初始化权重
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """绑定MLM头的解码器权重与词嵌入权重（BERT标准做法）"""
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True):
        """模型前向传播路由
        参数:
            batch: 输入数据字典
            task: 当前任务标识（mlm/mrc/sap/og等）
            compute_loss: 是否计算损失
        返回:
            根据任务类型返回相应的输出
        """
        # 将batch转换为默认字典（避免KeyError）
        batch = defaultdict(lambda: None, batch)
        # 根据任务类型路由到不同的前向方法
        if task.startswith('mlm'):
            return self.forward_mlm(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], batch['traj_text_feats'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['txt_labels'], batch['grid_fts'], batch['grid_map'], batch['target_patch_id'],
                batch['gridmap_pos_fts'], compute_loss
            )
        elif task.startswith('mrc'):
            return self.forward_mrc(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], batch['traj_text_feats'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['vp_view_mrc_masks'], batch['vp_view_probs'],
                batch['vp_obj_mrc_masks'], batch['vp_obj_probs'], batch['grid_fts'], batch['grid_map'],
                batch['target_patch_id'], batch['gridmap_pos_fts'], compute_loss
            )
        elif task.startswith('sap'):
            return self.forward_sap(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], batch['traj_text_feats'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'],
                batch['global_act_labels'], batch['local_act_labels'], batch['grid_fts'], batch['grid_map'],
                batch['target_patch_id'], batch['gridmap_pos_fts'], compute_loss
            )
        elif task.startswith('og'):
            return self.forward_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], batch['traj_text_feats'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['obj_labels'], batch['grid_fts'], batch['grid_map'], batch['target_patch_id'],
                batch['gridmap_pos_fts'], compute_loss
            )
        elif task.startswith('valid_sap_og'):
            return self.forward_sap_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], batch['traj_text_feats'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'], batch['global_act_labels'], batch['local_act_labels'],
                batch['obj_labels'], batch['grid_fts'], batch['grid_map'], batch['target_patch_id'],
                batch['gridmap_pos_fts'],
            )
        else:
            raise ValueError('invalid task')

    def forward_mlm(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_text_feats, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            txt_labels, grid_fts, grid_map, target_patch_id, gridmap_pos_fts, compute_loss
    ):
        """掩码语言建模任务前向传播
            实现细节:
                1. 通过BERT获取文本嵌入
                2. 仅计算被[MASK]位置的输出
                3. 预测原始token
        """
        txt_embeds = self.bert.forward_mlm(
            txt_ids, txt_lens, traj_view_img_fts, traj_text_feats, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts, grid_fts, grid_map,
            gridmap_pos_fts
        )

        # only compute masked tokens for better efficiency
        # 计算被掩码位置的输出
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)  # 预测被掩码的词

        if compute_loss:
            # 计算交叉熵损失
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1].to(torch.int64), reduction='none'
            )
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrc(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_text_feats, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            vp_view_mrc_masks, vp_view_probs, vp_obj_mrc_masks, vp_obj_probs, grid_fts, grid_map, target_patch_id,
            gridmap_pos_fts, compute_loss=True
    ):

        _, vp_embeds, _ = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_text_feats, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts, grid_fts, grid_map,
            gridmap_pos_fts=gridmap_pos_fts,
            return_gmap_embeds=False
        )

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens)]
        vp_view_embeds = pad_tensors_wgrad(
            [x[1:view_len + 1] for x, view_len in zip(vp_embeds, vp_view_lens)]
        )  # [stop] at 0
        # vp_view_mrc_masks = vp_view_mrc_masks[:, :vp_view_embeds.size(1)]

        # only compute masked regions for better efficient=cy
        view_masked_output = self._compute_masked_hidden(vp_view_embeds, vp_view_mrc_masks)
        view_prediction_soft_labels = self.image_classifier(view_masked_output)
        view_mrc_targets = self._compute_masked_hidden(vp_view_probs, vp_view_mrc_masks)

        if traj_obj_img_fts is not None:
            vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens)]
            vp_obj_embeds = pad_tensors_wgrad(
                [x[view_len + 1:view_len + obj_len + 1] for x, view_len, obj_len in
                 zip(vp_embeds, vp_view_lens, vp_obj_lens)]
            )
            # vp_obj_mrc_masks = vp_obj_mrc_masks[:, :vp_obj_embeds.size(1)]
            obj_masked_output = self._compute_masked_hidden(vp_obj_embeds, vp_obj_mrc_masks)
            if self.obj_classifier is None:
                obj_prediction_soft_labels = self.image_classifier(obj_masked_output)
            else:
                obj_prediction_soft_labels = self.obj_classifier(obj_masked_output)
            obj_mrc_targets = self._compute_masked_hidden(vp_obj_probs, vp_obj_mrc_masks)
        else:
            obj_prediction_soft_labels, obj_mrc_targets = None, None

        if compute_loss:
            view_prediction_soft_labels = F.log_softmax(view_prediction_soft_labels, dim=-1)
            view_mrc_loss = F.kl_div(view_prediction_soft_labels, view_mrc_targets, reduction='none').sum(dim=1)
            if obj_prediction_soft_labels is None:
                mrc_loss = view_mrc_loss
            else:
                obj_prediction_soft_labels = F.log_softmax(obj_prediction_soft_labels, dim=-1)
                obj_mrc_loss = F.kl_div(obj_prediction_soft_labels, obj_mrc_targets, reduction='none').sum(dim=1)
                mrc_loss = torch.cat([view_mrc_loss, obj_mrc_loss], 0)
            return mrc_loss
        else:
            return view_prediction_soft_labels, view_mrc_targets, obj_prediction_soft_labels, obj_mrc_targets

    def forward_sap(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_text_feats, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            gmap_visited_masks, global_act_labels, local_act_labels, grid_fts, grid_map, target_patch_id,
            gridmap_pos_fts, compute_loss
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds, gridmap_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_text_feats, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts, grid_fts, grid_map,
            gridmap_pos_fts=gridmap_pos_fts
        )

        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        grid_logits = self.grid_sap_head(gridmap_embeds).squeeze(2)
        grid_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        grid_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1] != 1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1) - 1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )  # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]  # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j + 1]
                else:
                    tmp[cand_vpid] = local_logits[i, j + 1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        if compute_loss:
            global_losses = F.cross_entropy(global_logits, global_act_labels.to(torch.int64), reduction='none')
            local_losses = F.cross_entropy(local_logits, local_act_labels.to(torch.int64), reduction='none')
            fused_losses = F.cross_entropy(fused_logits, global_act_labels.to(torch.int64), reduction='none')
            gridmap_losses = F.cross_entropy(grid_logits, global_act_labels.to(torch.int64), reduction='none')

            if global_losses[global_act_labels != 0].shape[0] != 0:
                stop_rate = global_losses[global_act_labels == 0].shape[0] / \
                            global_losses[global_act_labels != 0].shape[0]
            else:
                stop_rate = 1.

            global_losses[global_act_labels == 0] = global_losses[global_act_labels == 0] / stop_rate
            local_losses[local_act_labels == 0] = local_losses[local_act_labels == 0] / stop_rate
            fused_losses[global_act_labels == 0] = fused_losses[global_act_labels == 0] / stop_rate
            gridmap_losses[global_act_labels == 0] = gridmap_losses[global_act_labels == 0] / stop_rate

            losses = global_losses + local_losses + fused_losses + gridmap_losses
            return losses
        else:
            return global_logits, local_logits, fused_logits, global_act_labels, local_act_labels

    def forward_og(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_text_feats, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            obj_labels, grid_fts, grid_map, target_patch_id, gridmap_pos_fts, compute_loss
    ):
        gmap_embeds, vp_embeds, gridmap_embeds = self.bert.forward(
            txt_ids, txt_lens, traj_view_img_fts, traj_text_feats, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts, grid_fts, grid_map,
            gridmap_pos_fts=gridmap_pos_fts,
            return_gmap_embeds=False
        )

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1 + view_len: 1 + view_len + obj_len] for x, view_len, obj_len in
            zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        if compute_loss:
            losses = F.cross_entropy(obj_logits, obj_labels.to(torch.int64), reduction='none')
            return losses
        else:
            return obj_logits

    def forward_sap_og(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_text_feats, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            gmap_visited_masks, global_act_labels, local_act_labels, obj_labels, grid_fts, grid_map, target_patch_id,
            gridmap_pos_fts
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds, gridmap_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_text_feats, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts, grid_fts, grid_map,
            gridmap_pos_fts=gridmap_pos_fts
        )

        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1] != 1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1) - 1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )  # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]  # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j + 1]
                else:
                    tmp[cand_vpid] = local_logits[i, j + 1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1 + view_len: 1 + view_len + obj_len] for x, view_len, obj_len in
            zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        return global_logits, local_logits, fused_logits, obj_logits
