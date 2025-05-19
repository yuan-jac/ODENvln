import logging
import math

import torch
from torch import nn
from transformers import BertPreTrainedModel

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad

logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    """
    Bert模型中的中间层，主要包含一个全连接层和激活函数
    用于将注意力层的输出映射到更大的中间空间，再通过激活函数进行非线性变换
    """

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        # 中间层全连接层，将hidden_size映射到intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 激活函数选择，支持gelu、relu或swish
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """
        前向传播过程
        :param hidden_states: 输入张量，形状为(batch_size, seq_length, hidden_size)
        :return: 经过线性变换和激活函数后的张量
        """
        hidden_states = self.dense(hidden_states)  # 线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)  # 激活函数处理
        return hidden_states


class BertOutput(nn.Module):
    """
    Bert模型中的输出层，主要包含一个全连接层、LayerNorm和Dropout
    用于将中间层的输出映射回hidden_size空间，并进行残差连接和归一化
    """

    def __init__(self, config):
        super(BertOutput, self).__init__()
        # 输出层全连接层，将intermediate_size映射回hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm层，用于归一化隐藏状态
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        前向传播过程
        :param hidden_states: 中间层输出，形状为(batch_size, seq_length, intermediate_size)
        :param input_tensor: 来自上一层的输入，用于残差连接
                           形状与hidden_states相同
        :return: 经过线性变换、Dropout、LayerNorm和残差连接后的张量
        """
        hidden_states = self.dense(hidden_states)  # 线性变换
        hidden_states = self.dropout(hidden_states)  # Dropout
        # 残差连接 + LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """
    Bert模型中的基本构建块，包含注意力层和前馈神经网络(FFN)
    一个BertLayer由BertAttention和BertIntermediate+BertOutput组成
    """

    def __init__(self, config):
        super().__init__()
        # 注意力层，处理自注意力机制
        self.attention = BertAttention(config)
        # 中间层，处理非线性变换
        self.intermediate = BertIntermediate(config)
        # 输出层，处理残差连接和归一化
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        前向传播过程
        :param hidden_states: 输入张量，形状为(batch_size, seq_length, hidden_size)
        :param attention_mask: 注意力掩码，用于屏蔽某些位置的注意力分数
                             形状通常为(batch_size, 1, seq_length, seq_length)
        :param head_mask: 可选参数，用于屏蔽某些注意力头的输出
                        形状为(num_attention_heads,)或(num_layers, num_attention_heads)
        :return: 注意力层的输出和(可选的)注意力分数
                 注意力层输出形状与hidden_states相同
                 注意力分数形状为(batch_size, num_attention_heads, seq_length, seq_length)
        """
        # 通过注意力层
        attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask
        )
        attention_output = attention_outputs[0]  # 取第一个输出作为中间状态

        # 通过前馈神经网络(FFN)
        intermediate_output = self.intermediate(attention_output)
        # 最终输出，包含残差连接和归一化
        layer_output = self.output(intermediate_output, attention_output)

        # 返回输出和(可选的)注意力分数
        outputs = (layer_output,) + attention_outputs[1:]  # 如果需要注意力分数则包含
        return outputs


class BertEncoder(nn.Module):
    """
    Bert模型的编码器部分，由多个BertLayer堆叠而成
    实现了完整的Transformer编码器结构，包含多头自注意力机制和前馈神经网络
    """

    def __init__(self, config):
        """
        初始化Bert编码器

        参数:
            config: 配置对象，包含模型结构和训练参数
        """
        super().__init__()
        # 是否输出各层的注意力分数
        self.output_attentions = config.output_attentions
        # 是否输出各隐藏层状态
        self.output_hidden_states = config.output_hidden_states

        # 创建指定数量的BertLayer堆叠成编码器层
        # 每个BertLayer包含自注意力机制和前馈神经网络
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        前向传播过程

        参数:
            hidden_states: 输入张量，形状为(batch_size, seq_length, hidden_size)
            attention_mask: 注意力掩码，用于屏蔽某些位置的注意力分数
                          形状通常为(batch_size, 1, seq_length, seq_length)
            head_mask: 可选参数，用于屏蔽某些注意力头的输出
                      形状为(num_attention_heads,)或(num_layers, num_attention_heads)

        返回:
            outputs: 包含最终输出和(可选的)隐藏状态及注意力分数的元组
        """
        all_hidden_states = ()  # 存储所有隐藏状态(如果配置要求)
        all_attentions = ()  # 存储所有注意力分数(如果配置要求)

        # 遍历每一层BertLayer
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出各层隐藏状态，则保存当前状态
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 通过当前层进行处理
            # 参数head_mask处理：如果是None则传None，否则取对应层的mask
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                None if head_mask is None else head_mask[i],
            )

            # 更新hidden_states为当前层的输出
            hidden_states = layer_outputs[0]  # 取第一个输出作为中间状态

            # 如果需要输出注意力分数，则保存
            if self.output_attentions:
                # 将当前层的注意力分数添加到all_attentions中
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的hidden_state到输出
        # 注意：这里的设计可能有问题，通常应该添加hidden_states而不是重新添加
        # 正确的做法可能是：all_hidden_states = all_hidden_states + (hidden_states,)
        # 但原代码中这里重复添加了hidden_states变量，可能是笔误
        # 应该改为：
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 组织输出结果
        outputs = (hidden_states,)  # 最后一层的输出作为主要输出

        # 如果需要隐藏状态，则添加到输出中
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        # 如果需要注意力分数，则添加到输出中
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # 返回最后一层隐藏状态，(所有隐藏状态)，(所有注意力分数)


class BertPooler(nn.Module):
    """
    Bert模型的池化层，用于提取句子表示

    功能：
    1. 通过取第一个token([CLS])的隐藏状态作为句子表示
    2. 对该表示进行线性变换和tanh激活

    参数：
        config: 配置对象，包含模型参数
    """

    def __init__(self, config):
        super(BertPooler, self).__init__()
        # 线性变换层，将hidden_size映射回hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # tanh激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """
        前向传播过程

        参数:
            hidden_states: 输入张量，形状为(batch_size, seq_length, hidden_size)

        返回:
            pooled_output: 池化后的句子表示，形状为(batch_size, hidden_size)
        """
        # 取第一个token([CLS])的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 线性变换
        pooled_output = self.dense(first_token_tensor)
        # 激活函数处理
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    """
    预测头的转换层，用于将隐藏状态转换为预测空间

    功能：
    1. 线性变换
    2. 应用激活函数(根据配置)
    3. 层归一化

    参数：
        config: 配置对象，包含模型参数
    """

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        # 线性变换层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act

        # 层归一化
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        """
        前向传播过程

        参数:
            hidden_states: 输入张量，形状为(batch_size, seq_length, hidden_size)

        返回:
            transformed_states: 转换后的张量，形状与输入相同
        """
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 层归一化
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """
    Bert的语言模型预测头，用于下一句预测或掩码语言建模

    功能：
    1. 将隐藏状态转换为词汇表空间
    2. 添加偏置项
    3. 计算与词汇表的相似度

    参数：
        config: 配置对象，包含模型参数
    """

    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        # 转换层，将hidden_size映射到vocab_size
        self.transform = BertPredictionHeadTransform(config)

        # 输出投影层
        # 注意：没有偏置项(bias=False)，因为偏置会单独添加
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        # 偏置项，用于每个词汇表的得分
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """
        前向传播过程

        参数:
            hidden_states: 输入张量，形状为(batch_size, seq_length, hidden_size)

        返回:
            prediction_scores: 预测得分，形状为(batch_size, seq_length, vocab_size)
        """
        # 应用转换层
        hidden_states = self.transform(hidden_states)
        # 投影到词汇表空间并添加偏置
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    """
    仅用于掩码语言建模(MLM)任务的预测头

    功能：
    1. 包含一个LMPredictionHead层
    2. 将Transformer最后一层的隐藏状态映射回词汇表空间
    3. 计算每个位置的词汇表得分

    参数：
        config: 配置对象，包含模型参数
    """

    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        # 包含LMPredictionHead作为子模块
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        """
        前向传播过程

        参数:
            sequence_output: Transformer最后一层的隐藏状态
                         形状为(batch_size, seq_length, hidden_size)

        返回:
            prediction_scores: 预测得分
                           形状为(batch_size, seq_length, vocab_size)
        """
        # 通过LMPredictionHead计算预测得分
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOutAttention(nn.Module):
    """
    Bert的双注意力机制实现，用于处理跨模态或交叉注意力

    功能：
    1. 实现query-key-value的注意力计算
    2. 支持注意力掩码
    3. 可选择性地应用头掩码
    4. 返回注意力分数和上下文向量

    参数：
        config: 配置对象，包含模型参数
        ctx_dim: 可选，上下文特征的维度(默认与hidden_size相同)
    """

    def __init__(self, config, ctx_dim=None):
        super().__init__()
        # 检查隐藏大小是否是注意力头数的倍数
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 如果没有提供ctx_dim，则默认使用hidden_size
        if ctx_dim is None:
            ctx_dim = config.hidden_size

        # 定义query、key、value线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        # 注意力概率的dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        将输入张量重塑为多头注意力的形状

        参数:
            x: 输入张量

        返回:
            重塑后的张量，形状为(batch_size, num_heads, seq_length, attention_head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_size)

    def forward(self, hidden_states, context, attention_mask=None):
        """
        前向传播过程

        参数:
            hidden_states: 查询张量
                         形状为(batch_size, query_seq_length, hidden_size)
            context: 键和值张量
                    形状为(batch_size, key/value_seq_length, ctx_dim)
            attention_mask: 可选，注意力掩码
                          添加到注意力分数上，形状为(batch_size, num_heads, query_seq_length, key/value_seq_length)

        返回:
            context_layer: 注意力加权后的上下文向量
                         形状与hidden_states相同
            attention_scores: 注意力分数
                            形状为(batch_size, num_heads, query_seq_length, key/value_seq_length)
        """
        # 计算query、key和value
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        # 将线性变换后的张量转换为多头形式
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 缩放注意力分数
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用注意力掩码(如果提供)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 归一化注意力分数为概率分布
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 应用dropout
        attention_probs = self.dropout(attention_probs)

        # 计算上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)

        # 将多头注意力结果转换回原始形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_scores


class BertXAttention(nn.Module):
    """
    跨模态交叉注意力模块，用于处理视觉和语言模态之间的交互

    功能：
    1. 实现语言到视觉或视觉到语言的交叉注意力
    2. 包含自注意力子模块(BertSelfAttention)和前馈网络子模块(BertSelfOutput)
    3. 支持模态间的信息交互

    参数：
        config: 配置对象，包含模型参数
        ctx_dim: 可选，上下文特征的维度(默认与hidden_size相同)
    """

    def __init__(self, config, ctx_dim=None):
        super().__init__()
        # 定义注意力子模块
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        # 定义前馈网络子模块
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        """
        前向传播过程

        参数:
            input_tensor: 输入特征张量(语言或视觉特征)
                         形状取决于使用场景(语言或视觉)
            ctx_tensor: 上下文特征张量(另一种模态的特征)
                       形状取决于使用场景(视觉或语言)
            ctx_att_mask: 可选，上下文特征的注意力掩码
                          形状取决于使用场景

        返回:
            attention_output: 注意力后的输出特征
                            形状与input_tensor相同
            attention_scores: 注意力分数(如果配置输出)
                             形状取决于使用场景
        """
        # 计算交叉注意力
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        # 通过前馈网络进一步处理
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores


class GraphLXRTXLayer(nn.Module):
    """
    图结构视觉-语言交互层，结合了语言自注意力、视觉自注意力和跨模态注意力

    功能：
    1. 处理语言和视觉模态的自注意力机制
    2. 实现跨模态注意力（语言到视觉或视觉到语言）
    3. 支持图结构关系（通过graph_sprels参数）

    参数：
        config: 配置对象，包含模型参数
        use_lang2visn_attn: 是否启用语言到视觉的注意力
    """

    def __init__(self, config):
        super().__init__()

        # 语言自注意力和前馈网络（如果启用）
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # 视觉自注意力和前馈网络
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # 跨模态注意力层（视觉到语言）
        self.visual_attention = BertXAttention(config)

    def forward(
            self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
            graph_sprels=None
    ):
        """
        前向传播过程

        参数:
            lang_feats: 语言特征张量，形状为(batch_size, seq_len, hidden_size)
            lang_attention_mask: 语言注意力掩码，形状为(batch_size, 1, seq_len, seq_len)
            visn_feats: 视觉特征张量，形状为(batch_size, num_objs, hidden_size)
            visn_attention_mask: 视觉注意力掩码，形状为(batch_size, 1, num_objs, num_objs)
            graph_sprels: 图结构关系（可选），用于增强视觉注意力

        返回:
            视觉特征的最终输出，形状与visn_feats相同
        """
        # 跨模态注意力：视觉特征关注语言特征
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]

        # 应用图结构关系（如果提供）
        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels

        # 视觉自注意力
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]

        # 视觉前馈网络
        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

    def forward_lang2visn(
            self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask
    ):
        """
        语言关注视觉的反向注意力路径

        参数:
            lang_feats: 语言特征张量
            lang_attention_mask: 语言注意力掩码
            visn_feats: 视觉特征张量
            visn_attention_mask: 视觉注意力掩码

        返回:
            语言特征的最终输出，形状与lang_feats相同
        """
        # 语言关注视觉
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]

        # 语言自注意力
        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]

        # 语言前馈网络
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)

        return lang_output


class LanguageEncoder(nn.Module):
    """
    语言编码器，由多个BertLayer堆叠而成

    功能：
    1. 对输入的文本嵌入进行多层编码
    2. 支持冻结部分层（通过update_lang_bert参数控制）
    3. 输出最后一层的隐藏状态

    参数：
        config: 配置对象，包含模型参数
        update_lang_bert: 是否更新语言BERT的参数
    """

    def __init__(self, config):
        super().__init__()

        self.num_l_layers = config.num_l_layers  # 语言层数
        self.update_lang_bert = config.update_lang_bert  # 是否更新语言BERT

        # 创建指定数量的BertLayer
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )

        # 如果不需要更新语言BERT，则冻结参数
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        """
        前向传播过程

        参数:
            txt_embeds: 文本嵌入张量，形状为(batch_size, seq_len, hidden_size)
            txt_masks: 文本注意力掩码，形状为(batch_size, 1, seq_len, seq_len)

        返回:
            编码后的文本嵌入，形状与txt_embeds相同
        """
        # 扩展注意力掩码
        extended_txt_masks = extend_neg_masks(txt_masks)

        # 逐层编码
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]  # 只取隐藏状态输出

        # 如果不需要更新语言BERT，则detach以防止梯度流动
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()

        return txt_embeds


class CrossmodalEncoder(nn.Module):
    """
    跨模态编码器，处理语言和视觉特征的交互

    功能：
    1. 结合GraphLXRTXLayer实现多模态特征融合
    2. 支持图结构关系建模（通过graph_sprels参数）
    3. 输出增强后的视觉特征

    参数：
        config: 配置对象，包含模型参数
    """

    def __init__(self, config):
        super().__init__()

        self.num_x_layers = config.num_x_layers  # 跨模态层数
        # 创建指定数量的GraphLXRTXLayer
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        """
        前向传播过程

        参数:
            txt_embeds: 文本嵌入张量
            txt_masks: 文本注意力掩码
            img_embeds: 图像嵌入张量
            img_masks: 图像注意力掩码
            graph_sprels: 图结构关系（可选）

        返回:
            增强后的图像嵌入，形状与img_embeds相同
        """
        # 扩展注意力掩码
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks)  # (N, 1(H), 1(L_q), L_v)

        # 逐层跨模态编码
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                txt_embeds, extended_txt_masks,
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )

        return img_embeds


class ImageEmbeddings(nn.Module):
    """
    图像嵌入模块，将输入的图像特征转换为Transformer模型的隐藏表示

    功能：
    1. 处理多种输入图像特征（视觉特征、位置特征、导航类型）
    2. 包含线性变换、层归一化和残差连接
    3. 支持对象特征的额外处理（如果有）
    4. 最终输出融合所有特征的图像表示

    参数：
        config: 模型配置参数，包含以下关键属性：
            - image_feat_size: 输入图像特征的维度
            - hidden_size: 模型隐藏层的维度
            - angle_feat_size: 角度特征的维度
            - obj_feat_size: 对象特征的维度（可选）
            - layer_norm_eps: 层归一化的epsilon值
            - hidden_dropout_prob: 隐藏层dropout概率
    """

    def __init__(self, config):
        super().__init__()

        # 基础图像特征处理
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # 位置特征处理（角度+其他3D坐标）
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # 对象特征处理（如果有）
        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.obj_linear = None
            self.obj_layer_norm = None

        # 导航类型嵌入（0:不可导航, 1:可导航, 2:对象）
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # 标准层归一化
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
            self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):
        """
        前向传播过程

        参数:
            traj_view_img_fts: 轨迹视角图像特征 [batch_size, num_views, image_feat_size]
            traj_obj_img_fts: 轨迹对象图像特征 [batch_size, num_objects, obj_feat_size] (可选)
            traj_loc_fts: 轨迹位置特征 [batch_size, num_locations, angle_feat_size + 3]
            traj_nav_types: 导航类型 [batch_size, num_locations]
            traj_step_lens: 每个轨迹步长列表
            traj_vp_view_lens: 每个视角的位置长度列表
            traj_vp_obj_lens: 每个视角的对象长度列表
            type_embed_layer: 类型嵌入层

        返回:
            split_traj_embeds: 分割后的轨迹嵌入 [batch_size * num_steps, hidden_size]
            split_traj_vp_lens: 分割后的视角长度列表
        """
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None

        # 处理视角图像特征
        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))

        if has_obj:
            # 处理对象图像特征
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_fts))

            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                    traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    # 合并视角和对象特征
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        # 组合所有特征
        traj_embeds = (
                traj_img_embeds +
                self.loc_layer_norm(self.loc_linear(traj_loc_fts)) +
                self.nav_type_embedding(traj_nav_types) +
                type_embed_layer(torch.ones(1, 1).long().to(device))
        )

        # 应用层归一化和dropout
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        # 生成轨迹掩码
        traj_masks = gen_seq_masks(traj_vp_lens)

        # 如果配置了全景编码器，应用它
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        # 分割轨迹嵌入和时间步长
        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)

        return split_traj_embeds, split_traj_vp_lens


class LocalVPEncoder(nn.Module):
    """
    局部视点编码器(Local Viewpoint Encoder)，用于处理轨迹中的局部视点特征

    功能：
    1. 对轨迹中的视点特征进行位置嵌入
    2. 通过CrossmodalEncoder进行跨模态交互
    3. 输出增强后的视点特征

    参数：
        config: 模型配置对象，包含各种超参数
    """

    def __init__(self, config):
        super().__init__()

        # 视点位置嵌入层
        # 由两个线性层和层归一化组成，处理角度特征和额外特征
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size * 2 + 6, config.hidden_size),  # 输入特征维度: 角度特征*2 + 6
            BertLayerNorm(config.hidden_size, eps=1e-12)  # 层归一化
        )

        # 交叉模态编码器，用于处理文本和视点的交互
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        """
        视点输入嵌入处理

        参数:
            split_traj_embeds: 分割后的轨迹嵌入特征 [batch_size, seq_len, hidden_size]
            split_traj_vp_lens: 分割后的视点长度序列 [batch_size, seq_len]
            vp_pos_fts: 视点位置特征 [batch_size, seq_len, pos_feat_dim]

        返回:
            vp_embeds: 增强后的视点嵌入 [batch_size, seq_len, hidden_size]
            vp_masks: 视点掩码 [batch_size, seq_len]
        """

        # 获取每个样本的最后一个视点嵌入作为图像嵌入
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # 计算每个样本的有效视点长度
        vp_lens = torch.stack([x[-1] + 1 for x in split_traj_vp_lens], 0)  # +1因为包含[stop] token

        # 生成视点掩码
        vp_masks = gen_seq_masks(vp_lens)

        # 计算最大视点长度
        max_vp_len = max(vp_lens)
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        """
        前向传播过程

        参数:
            txt_embeds: 文本嵌入特征 [batch_size, seq_len, hidden_size]
            txt_masks: 文本掩码 [batch_size, seq_len]
            split_traj_embeds: 分割后的轨迹嵌入特征 [batch_size, seq_len, hidden_size]
            split_traj_vp_lens: 分割后的视点长度序列 [batch_size, seq_len]
            vp_pos_fts: 视点位置特征 [batch_size, seq_len, pos_feat_dim]

        返回:
            vp_embeds: 增强后的视点嵌入 [batch_size, seq_len, hidden_size]
        """
        # 处理视点输入嵌入
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )

        # 通过交叉模态编码器进行文本和视点的交互
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds


class GlobalMapEncoder(nn.Module):
    """
    全局地图编码器，用于处理全局地图特征和轨迹特征

    功能：
    1. 聚合轨迹特征生成全局地图特征
    2. 添加位置嵌入和步长嵌入
    3. 应用Transformer编码器增强地图特征

    参数：
        config: 配置对象，包含模型参数
    """

    def __init__(self, config):
        super().__init__()
        # 地图位置嵌入层
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )

        # 步长嵌入层（记录动作步骤）
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)

        # 特殊关系线性变换层（可选）
        self.sprel_linear = None

    def _aggregate_gmap_features(
            self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        """
        聚合轨迹特征生成全局地图特征

        参数:
            split_traj_embeds: 分割后的轨迹嵌入 [batch_size, seq_len, hidden_size]
            split_traj_vp_lens: 分割后的视角长度列表 [batch_size]
            traj_vpids: 轨迹中的视角ID [batch_size, max_len]
            traj_cand_vpids: 轨迹中的候选视角ID [batch_size, max_len, max_cands]
            gmap_vpids: 全局地图中的视角ID [batch_size, map_len]

        返回:
            batch_gmap_img_fts: 聚合后的全局地图特征 [batch_size, map_len, hidden_size]
        """
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []

        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}  # 存储已访问和未访问的视角特征

            # 计算轨迹掩码
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])

            # 计算轨迹嵌入（仅有效部分）并加权求和
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)

            # 遍历轨迹中的每个位置
            for t in range(len(split_traj_embeds[i])):
                vp_id = traj_vpids[i][t]
                visited_vp_fts[vp_id] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]

                # 记录候选视角特征
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            # 为全局地图中的每个视角ID生成特征
            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:  # 跳过第一个元素（通常是起始点）
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    # 如果视角未访问，使用候选视角的平均特征
                    gmap_img_fts.append(torch.mean(
                        torch.stack(unvisited_vp_fts[vp], 0),
                        0
                    ))
            # 堆叠所有视角特征并填充
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        # 在batch维度上填充特征
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)

        # 添加[stop] token（全零表示）
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device),
             batch_gmap_img_fts],
            dim=1
        )

        return batch_gmap_img_fts

    def gmap_input_embedding(
            self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
    ):
        """
        生成全局地图的输入嵌入

        参数:
            split_traj_embeds: 分割后的轨迹嵌入
            split_traj_vp_lens: 分割后的视角长度列表
            traj_vpids: 轨迹中的视角ID
            traj_cand_vpids: 轨迹中的候选视角ID
            gmap_vpids: 全局地图中的视角ID
            gmap_step_ids: 地图步骤ID（动作序列）
            gmap_pos_fts: 地图位置特征
            gmap_lens: 地图长度

        返回:
            gmap_embeds: 地图嵌入 [batch_size, map_len, hidden_size]
            gmap_masks: 地图掩码 [batch_size, map_len]
        """
        # 聚合地图特征
        batch_gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )

        # 添加步长嵌入和位置嵌入
        gmap_embeds = batch_gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)

        # 生成地图掩码
        gmap_masks = gen_seq_masks(gmap_lens)

        return gmap_embeds, gmap_masks

    def forward(
            self, txt_embeds, txt_masks,
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):
        """
        前向传播过程

        参数:
            txt_embeds: 文本嵌入 [batch_size, seq_len, hidden_size]
            txt_masks: 文本掩码 [batch_size, 1, seq_len, seq_len]
            split_traj_embeds: 分割后的轨迹嵌入
            split_traj_vp_lens: 分割后的视角长度列表
            traj_vpids: 轨迹中的视角ID
            traj_cand_vpids: 轨迹中的候选视角ID
            gmap_vpids: 全局地图中的视角ID
            gmap_step_ids: 地图步骤ID（动作序列）
            gmap_pos_fts: 地图位置特征
            gmap_lens: 地图长度
            graph_sprels: 图结构关系（可选）

        返回:
            map_embeds: 增强后的地图嵌入 [batch_size, map_len, hidden_size]
        """
        # 获取地图输入嵌入和掩码
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )

        # 这里原代码没有对gmap_embeds进一步处理就直接返回了
        # 可能需要添加Transformer编码器层来增强地图特征

        return gmap_embeds


class GlocalTextPathCMT(BertPreTrainedModel):
    """
    GlocalTextPathCMT 类集成了全局和局部特征编码器，用于处理文本和视觉轨迹的多模态数据。
    该模型旨在通过跨模态交互增强对全局地图和局部轨迹的理解。
    """

    def __init__(self, config):
        """
        GlocalTextPathCMT 类的初始化方法。

        参数:
            config: 配置对象，包含模型所需的各种参数和设置。
        """
        super().__init__(config)

        # 初始化嵌入层，用于将输入ID转换为嵌入向量
        self.embeddings = BertEmbeddings(config)

        # 语言编码器，处理文本嵌入
        self.lang_encoder = LanguageEncoder(config)

        # 图像嵌入模块，处理视觉特征
        self.img_embeddings = ImageEmbeddings(config)

        # 局部视点编码器，处理轨迹中的局部视点特征
        self.local_encoder = LocalVPEncoder(config)

        # 全局地图编码器，处理全局地图特征
        self.global_encoder = GlobalMapEncoder(config)

        # 网格编码器，使用单层Transformer编码器处理网格特征
        self.grid_encoder = create_transformer_encoder(
            config, 1, norm=True
        )

        # 配置跨模态编码器的层数为1
        config.num_x_layers = 1

        # 跨模态编码器，结合图像和文本特征
        self.grid_txt_encoder = CrossmodalEncoder(config)

        """
        网格位置嵌入层，将5维的网格位置信息映射到模型隐藏层维度。
        包含线性变换和层归一化。
        """
        self.grid_pos_embeddings = nn.Sequential(
            nn.Linear(5, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )

        # 文本投影层，将768维的文本嵌入转换为相同的维度（可能用于适配）
        self.text_proj = nn.Linear(768, 768)

        # 网格投影层，将768维的网格嵌入转换为相同的维度（使用float16以节省内存）
        self.grid_proj = nn.Linear(768, 768).to(torch.float16)

        # 初始化模型权重
        self.init_weights()

    def forward(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts, grid_fts, grid_map,
            target_patch_id=None, gridmap_pos_fts=None,
            return_gmap_embeds=True
    ):
        """
        前向传播过程，处理文本和视觉轨迹数据，生成增强的全局地图和局部视点嵌入。

        参数:
            txt_ids: 文本输入ID [batch_size, seq_len]
            txt_lens: 每个样本的文本长度列表 [batch_size]
            traj_view_img_fts: 轨迹视角图像特征 [batch_size, num_views, image_feat_size]
            traj_obj_img_fts: 轨迹对象图像特征 [batch_size, num_objects, obj_feat_size] (可选)
            traj_loc_fts: 轨迹位置特征 [batch_size, num_locations, angle_feat_size + 3]
            traj_nav_types: 导航类型 [batch_size, num_locations]
            traj_step_lens: 每个轨迹步长列表
            traj_vp_view_lens: 每个视角的位置长度列表
            traj_vp_obj_lens: 每个视角的对象长度列表
            traj_vpids: 轨迹中的视角ID
            traj_cand_vpids: 轨迹中的候选视角ID
            gmap_lens: 全局地图长度
            gmap_step_ids: 地图步骤ID（动作序列）
            gmap_pos_fts: 地图位置特征
            gmap_pair_dists: 地图对之间的距离（可选）
            gmap_vpids: 全局地图中的视角ID
            vp_pos_fts: 视点位置特征
            grid_fts: 网格特征 [batch_size, grid_size, feature_dim]
            grid_map: 网格映射关系 [batch_size, grid_size]
            target_patch_id: 目标补丁ID（可选）
            gridmap_pos_fts: 网格地图位置特征（可选）
            return_gmap_embeds: 是否返回全局地图嵌入（默认为True）

        返回:
            gmap_embeds: 增强后的全局地图嵌入 [batch_size, gmap_seq_len, hidden_size]
            vp_embeds: 增强后的局部视点嵌入 [batch_size, vp_seq_len, hidden_size]
            map_embeds: 映射后的网格嵌入 [batch_size, total_cell_num, hidden_size] (仅当return_gmap_embeds为False时)
        """
        batch_size = len(grid_fts)
        # 初始化网格地图输入张量
        grid_map_input = torch.zeros(batch_size, 16 * 16, 768).to(grid_fts[0].device)

        # 文本嵌入处理
        txt_token_type_ids = torch.zeros_like(txt_ids).to(torch.int32)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        txt_masks = gen_seq_masks(txt_lens)  # 生成文本序列掩码
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)  # 通过语言编码器处理

        # 将文本嵌入投影到另一维度并转置
        text_fts = self.text_proj(txt_embeds).permute(0, 2, 1).to(torch.float16)

        grid_masks = [[] for b in range(batch_size)]  # 初始化每个样本的网格掩码列表
        max_cell_num = 0  # 记录最大有效网格单元数量

        for b in range(batch_size):
            """
            计算每个网格单元与文本特征的关联权重，并生成网格地图输入。

            步骤:
            1. 计算网格特征与文本特征的关联权重。
            2. 对每个网格单元，如果存在对应的特征，则计算加权特征。
            3. 更新网格掩码，标记有效网格单元。
            """
            grid_fts_weight, _ = (grid_fts[b] @ text_fts[b]).max(dim=-1)  # 计算权重
            tmp_fts = self.grid_proj(grid_fts[b])  # 通过网格投影层

            for i in range(16 * 16):  # 遍历每个网格单元 (假设网格大小为16x16)
                cell_fts = tmp_fts[grid_map[b] == i]  # 获取属于当前网格单元的特征
                if cell_fts.shape[0] == 0:
                    grid_masks[b].append(0)  # 无特征，标记为无效
                else:
                    grid_masks[b].append(1)  # 有特征，标记为有效
                # 计算加权特征并赋值到网格地图输入
                grid_map_input[b, i] = (
                        cell_fts * torch.softmax(grid_fts_weight[grid_map[b] == i], dim=-1).unsqueeze(-1)
                ).sum(-2).to(torch.float32)

            # 更新最大有效网格单元数量
            if max_cell_num < sum(grid_masks[b]):
                max_cell_num = sum(grid_masks[b])

        # 将网格掩码列表转换为张量
        grid_masks = torch.tensor(grid_masks).to(txt_masks.device)

        # 初始化网格地图嵌入张量
        grid_map_embeds = torch.zeros(batch_size, max_cell_num, 768).to(grid_fts[0].device)

        # 添加网格位置嵌入
        grid_map_input = grid_map_input + self.grid_pos_embeddings(gridmap_pos_fts)

        for b in range(batch_size):
            grid_mask = grid_masks[b]
            # 填充有效网格嵌入
            grid_map_embeds[b, :grid_mask.sum()] = grid_map_input[b][grid_mask == 1]
            # 更新掩码，标记有效和无效网格
            grid_masks[b, :grid_mask.sum()] = 1
            grid_masks[b, grid_mask.sum():] = 0

        # 仅保留有效网格部分的掩码
        grid_masks = grid_masks[:, :max_cell_num].bool()

        """
        轨迹嵌入处理：提取轨迹中的图像特征（视角和对象）。
        """
        split_traj_embeds, split_traj_vp_lens = self.img_embeddings(
            traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens,
            self.embeddings.token_type_embeddings  # 使用嵌入层的token类型嵌入
        )

        """
        全局地图嵌入处理：通过全局地图编码器处理轨迹嵌入，生成全局地图特征。
        """
        gmap_embeds, gmap_masks = self.global_encoder.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )

        """
        局部视点嵌入处理：通过局部视点编码器处理轨迹嵌入，生成局部视点特征。
        """
        vp_embeds, vp_masks = self.local_encoder.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )

        """
        将网格地图嵌入和全局地图嵌入拼接，并应用网格编码器和跨模态编码器。
        """
        map_embeds = torch.cat([grid_map_embeds, gmap_embeds], dim=1)  # 拼接网格和全局地图嵌入
        map_masks = torch.cat([grid_masks, gmap_masks], dim=1)  # 拼接对应掩码

        # 通过单层Transformer网格编码器处理
        map_embeds = self.grid_encoder(map_embeds, src_key_padding_mask=map_masks.logical_not())

        # 通过跨模态编码器结合文本嵌入和网格嵌入
        map_embeds = self.grid_txt_encoder(txt_embeds, txt_masks, map_embeds, map_masks)

        # 提取全局地图嵌入部分
        gmap_embeds = map_embeds[:, max_cell_num:]

        """
        特征融合：将网格嵌入、文本嵌入和轨迹嵌入进行融合，通过局部视点编码器进一步处理。
        """
        kv_masks = torch.cat([map_masks, txt_masks], dim=1)  # 合并掩码
        kv_embeds = torch.cat([map_embeds, txt_embeds], dim=1)  # 合并嵌入

        vp_masks = torch.cat([gmap_masks, vp_masks], dim=1)  # 合并视点掩码
        vp_embeds = torch.cat([gmap_embeds, vp_embeds], dim=1)  # 合并视点嵌入

        # 通过局部视点编码器的Transformer层处理
        vp_embeds = self.local_encoder.encoder(kv_embeds, kv_masks, vp_embeds, vp_masks)

        # 分离全局地图嵌入和局部视点嵌入
        gmap_embeds = vp_embeds[:, :gmap_masks.shape[1]]
        vp_embeds = vp_embeds[:, gmap_masks.shape[1]:]

        """
        根据return_gmap_embeds决定是否返回全局地图嵌入。
        如果return_gmap_embeds为False，则将gmap_embeds设为None。
        """
        if not return_gmap_embeds:
            gmap_embeds = None

        # 返回全局地图嵌入、局部视点嵌入和映射后的网格嵌入（当return_gmap_embeds为False时）
        return gmap_embeds, vp_embeds, map_embeds[:, max_cell_num:]

    def forward_mlm(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts, grid_fts, grid_map,
            gridmap_pos_fts
    ):
        """
        掩码语言建模（Masked Language Modeling, MLM）前向传播过程。

        参数:
            txt_ids: 文本输入ID [batch_size, seq_len]
            txt_lens: 每个样本的文本长度列表 [batch_size]
            traj_view_img_fts: 轨迹视角图像特征 [batch_size, num_views, image_feat_size]
            traj_obj_img_fts: 轨迹对象图像特征 [batch_size, num_objects, obj_feat_size] (可选)
            traj_loc_fts: 轨迹位置特征 [batch_size, num_locations, angle_feat_size + 3]
            traj_nav_types: 导航类型 [batch_size, num_locations]
            traj_step_lens: 每个轨迹步长列表
            traj_vp_view_lens: 每个视角的位置长度列表
            traj_vp_obj_lens: 每个视角的对象长度列表
            traj_vpids: 轨迹中的视角ID
            traj_cand_vpids: 轨迹中的候选视角ID
            gmap_lens: 全局地图长度
            gmap_step_ids: 地图步骤ID（动作序列）
            gmap_pos_fts: 地图位置特征
            gmap_pair_dists: 地图对之间的距离（可选）
            gmap_vpids: 全局地图中的视角ID
            vp_pos_fts: 视点位置特征
            grid_fts: 网格特征 [batch_size, grid_size, feature_dim]
            grid_map: 网格映射关系 [batch_size, grid_size]
            gridmap_pos_fts: 网格地图位置特征（可选）

        返回:
            txt_embeds: 经过MLM处理后的文本嵌入 [batch_size, seq_len, hidden_size]
        """
        batch_size = len(grid_fts)
        # 初始化网格地图输入张量
        grid_map_input = torch.zeros(batch_size, 16 * 16, 768).to(grid_fts[0].device)

        # 文本嵌入处理
        txt_token_type_ids = torch.zeros_like(txt_ids)  # 假设token类型全为0
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        txt_masks = gen_seq_masks(txt_lens)  # 生成文本序列掩码
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)  # 通过语言编码器处理

        # 扩展文本掩码以适应跨模态需求
        extended_txt_masks = extend_neg_masks(txt_masks)

        # 将文本嵌入投影到另一维度并转置
        text_fts = self.text_proj(txt_embeds).permute(0, 2, 1).to(torch.float16)

        grid_masks = [[] for b in range(batch_size)]  # 初始化每个样本的网格掩码列表
        max_cell_num = 0  # 记录最大有效网格单元数量

        for b in range(batch_size):
            """
            计算每个网格单元与文本特征的关联权重，并生成网格地图输入。
            """
            grid_fts_weight, _ = (grid_fts[b] @ text_fts[b]).max(dim=-1)  # 计算权重
            tmp_fts = self.grid_proj(grid_fts[b])  # 通过网格投影层

            for i in range(16 * 16):  # 遍历每个网格单元 (假设网格大小为16x16)
                cell_fts = tmp_fts[grid_map[b] == i]  # 获取属于当前网格单元的特征
                if cell_fts.shape[0] == 0:
                    grid_masks[b].append(0)  # 无特征，标记为无效
                else:
                    grid_masks[b].append(1)  # 有特征，标记为有效
                # 计算加权特征并赋值到网格地图输入
                grid_map_input[b, i] = (
                        cell_fts * torch.softmax(grid_fts_weight[grid_map[b] == i], dim=-1).unsqueeze(-1)
                ).sum(-2).to(torch.float32)

            # 更新最大有效网格单元数量
            if max_cell_num < sum(grid_masks[b]):
                max_cell_num = sum(grid_masks[b])

        # 将网格掩码列表转换为张量
        grid_masks = torch.tensor(grid_masks).to(txt_masks.device)

        # 初始化网格地图嵌入张量
        grid_map_embeds = torch.zeros(batch_size, max_cell_num, 768).to(grid_fts[0].device)

        # 添加网格位置嵌入
        grid_map_input = grid_map_input + self.grid_pos_embeddings(gridmap_pos_fts)

        for b in range(batch_size):
            grid_mask = grid_masks[b]
            # 填充有效网格嵌入
            grid_map_embeds[b, :grid_mask.sum()] = grid_map_input[b][grid_mask == 1]
            # 更新掩码，标记有效和无效网格
            grid_masks[b, :grid_mask.sum()] = 1
            grid_masks[b, grid_mask.sum():] = 0

        # 仅保留有效网格部分的掩码
        grid_masks = grid_masks[:, :max_cell_num].bool()

        """
        轨迹嵌入处理：提取轨迹中的图像特征（视角和对象）。
        """
        split_traj_embeds, split_traj_vp_lens = self.img_embeddings(
            traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens,
            self.embeddings.token_type_embeddings  # 使用嵌入层的token类型嵌入
        )

        """
        全局地图嵌入处理：通过全局地图编码器处理轨迹嵌入，生成全局地图特征。
        """
        gmap_input_embeds, gmap_masks = self.global_encoder.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )
        extended_gmap_masks = extend_neg_masks(gmap_masks)  # 扩展全局地图掩码

        """
        局部视点嵌入处理：通过局部视点编码器处理轨迹嵌入，生成局部视点特征。
        """
        vp_input_embeds, vp_masks = self.local_encoder.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        extended_vp_masks = extend_neg_masks(vp_masks)  # 扩展局部视点掩码

        """
        拼接网格地图嵌入和全局地图嵌入，应用网格编码器和跨模态编码器。
        """
        map_embeds = torch.cat([grid_map_embeds, gmap_input_embeds], dim=1)
        map_masks = torch.cat([grid_masks, gmap_masks], dim=1)

        # 通过单层Transformer网格编码器处理
        map_embeds = self.grid_encoder(map_embeds, src_key_padding_mask=map_masks.logical_not())

        # 通过跨模态编码器结合文本嵌入和网格嵌入
        map_embeds = self.grid_txt_encoder(txt_embeds, txt_masks, map_embeds, map_masks)

        # 提取全局地图嵌入部分
        gmap_embeds = map_embeds[:, max_cell_num:]

        """
        特征融合：将网格嵌入、文本嵌入和轨迹嵌入进行融合，通过局部视点编码器进一步处理。
        """
        vp_masks = torch.cat([gmap_masks, vp_masks], dim=1)  # 合并全局地图和局部视点掩码
        vp_embeds = torch.cat([gmap_embeds, vp_input_embeds], dim=1)  # 合并全局地图和局部视点嵌入

        # 扩展局部视点掩码
        extended_vp_masks = extend_neg_masks(vp_masks)

        # 通过局部视点编码器的Transformer层处理（仅lang2visn方向）
        for layer_module in self.local_encoder.encoder.x_layers:
            txt_embeds = layer_module.forward_lang2visn(
                txt_embeds, extended_txt_masks,
                vp_embeds, extended_vp_masks,
            )

        # 返回经过MLM处理后的文本嵌入
        return txt_embeds
