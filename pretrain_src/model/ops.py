import torch

from .transformer import TransformerEncoder, TransformerEncoderLayer

# 尝试从 apex 库中导入 FusedLayerNorm 并将其重命名为 BertLayerNorm，
# 如果导入失败（由于 ImportError 或 AttributeError），则使用 PyTorch 自带的 LayerNorm 作为替代。
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # 如果导入失败，打印提示信息（这里注释掉了实际的日志打印代码），
    # 并使用 PyTorch 的 LayerNorm 类作为 BertLayerNorm。
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


def create_transformer_encoder(config, num_layers, norm=False):
    """
    根据给定的配置创建一个 Transformer 编码器。

    参数:
    config: 配置对象，应包含以下属性:
        - hidden_size: 隐藏层的大小。
        - num_attention_heads: 注意力头的数量。
        - intermediate_size: 前馈网络的隐藏层大小。
        - hidden_dropout_prob: 隐藏层的 dropout 概率。
        - hidden_act: 隐藏层的激活函数。
    num_layers: 整数，Transformer 编码器的层数。
    norm: 布尔值，如果为 True，则在编码器层后添加层归一化。

    返回:
    TransformerEncoder: 创建的 Transformer 编码器实例。
    """
    # 创建一个 Transformer 编码器层实例。
    # 参数说明:
    # - config.hidden_size: 隐藏层的大小。
    # - config.num_attention_heads: 注意力头的数量。
    # - dim_feedforward=config.intermediate_size: 前馈网络的隐藏层大小。
    # - dropout=config.hidden_dropout_prob: 隐藏层的 dropout 概率。
    # - activation=config.hidden_act: 隐藏层的激活函数。
    # - normalize_before=True: 在注意力机制之前进行归一化。
    enc_layer = TransformerEncoderLayer(
        config.hidden_size, config.num_attention_heads,
        dim_feedforward=config.intermediate_size,
        dropout=config.hidden_dropout_prob,
        activation=config.hidden_act,
        normalize_before=True
    )

    # 如果 norm 参数为 True，则创建一个层归一化实例，否则设置为 None。
    if norm:
        norm_layer = BertLayerNorm(config.hidden_size, eps=1e-12)
    else:
        norm_layer = None

    # 创建并返回一个 Transformer 编码器实例。
    # 参数说明:
    # - enc_layer: 创建的 Transformer 编码器层实例。
    # - num_layers: Transformer 编码器的层数。
    # - norm: 层归一化实例（可能为 None）。
    # - batch_first=True: 输入和输出的张量形状为 (batch, seq, feature)。
    return TransformerEncoder(enc_layer, num_layers, norm=norm_layer, batch_first=True)


def extend_neg_masks(masks, dtype=None):
    """
    将形状为 (N, L) 的掩码张量转换为 (N, 1(H), 1(L), L) 的形状，并将其值变为负数。
    这种转换通常用于 Transformer 模型中的注意力机制，以确保被掩码的位置在 softmax 计算中得到极小的值，
    从而在注意力权重计算中被忽略。

    参数:
    masks (torch.Tensor): 形状为 (N, L) 的布尔或数值掩码张量，其中 N 是批次大小，L 是序列长度。
                          掩码值为 1 表示需要保留的位置，0 表示需要屏蔽的位置。
    dtype (torch.dtype, 可选): 转换后掩码张量的数据类型。如果未提供，则默认为 torch.float。

    返回:
    torch.Tensor: 转换并取负后的掩码张量，形状为 (N, 1(H), 1(L), L)，数据类型为指定的 dtype。
    """
    # 如果未指定数据类型，则默认使用 torch.float
    if dtype is None:
        dtype = torch.float

    # 在第1维和第2维上增加两个维度，使张量形状从 (N, L) 变为 (N, 1, 1, L)
    # unsqueeze(1) 在第1维增加一个维度，结果形状为 (N, 1, L)
    # unsqueeze(2) 在第2维增加一个维度，结果形状为 (N, 1, 1, L)
    extended_masks = masks.unsqueeze(1).unsqueeze(2)

    # 将扩展后的掩码张量转换为指定的数据类型
    extended_masks = extended_masks.to(dtype=dtype)

    # 将掩码值取反（1 变为 0，0 变为 1），然后乘以 -10000.0，使其变为负数
    # 这样做的目的是在注意力机制中，被屏蔽的位置（原掩码值为 0）在 softmax 计算中得到极小的值
    extended_masks = (1.0 - extended_masks) * -10000.0

    # 返回转换并取负后的掩码张量
    return extended_masks


def gen_seq_masks(seq_lens, max_len=None):
    """
    生成序列掩码，用于屏蔽填充的部分。

    参数:
    seq_lens (torch.Tensor): 包含每个序列长度的张量，形状为 (batch_size,)。
    max_len (int, 可选): 序列的最大长度。如果未提供，则使用 seq_lens 中的最大值。

    返回:
    torch.Tensor: 形状为 (batch_size, max_len) 的布尔掩码张量，True 表示有效序列部分，False 表示填充部分。
    """
    # 如果未提供最大长度，则使用 seq_lens 中的最大值
    if max_len is None:
        max_len = max(seq_lens)

    # 获取批次大小
    batch_size = len(seq_lens)

    # 获取当前张量所在的设备（CPU 或 GPU）
    device = seq_lens.device

    # 生成一个形状为 (batch_size, max_len) 的掩码张量
    # torch.arange(max_len) 生成从 0 到 max_len-1 的序列
    # unsqueeze(0) 将其形状变为 (1, max_len)
    # repeat(batch_size, 1) 将其复制 batch_size 次，形状变为 (batch_size, max_len)
    masks = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1).to(device)

    # 将掩码张量与 seq_lens 进行比较，生成布尔掩码
    # masks < seq_lens.unsqueeze(1) 将 seq_lens 的形状从 (batch_size,) 变为 (batch_size, 1)
    # 比较操作会广播，最终生成形状为 (batch_size, max_len) 的布尔张量
    masks = masks < seq_lens.unsqueeze(1)

    # 返回生成的掩码张量
    return masks


def pad_tensors_wgrad(tensors, lens=None):
    """B x [T, ...] torch tensors"""
    # 如果未提供每个张量的长度，则计算每个张量的实际长度
    if lens is None:
        lens = [t.size(0) for t in tensors]

    # 获取所有张量中的最大长度
    max_len = max(lens)

    # 获取批次大小（即张量的数量）
    batch_size = len(tensors)

    # 获取每个张量的特征维度（假设张量的形状为 [T, ...]，这里获取除第一维外的其他维度）
    hid = list(tensors[0].size()[1:])

    # 获取第一个张量的设备信息（用于后续创建相同设备上的零张量）
    device = tensors[0].device

    # 获取第一个张量的数据类型（用于后续创建相同数据类型的零张量）
    dtype = tensors[0].dtype

    # 初始化一个空列表，用于存储填充后的张量
    output = []

    # 遍历批次中的每个张量
    for i in range(batch_size):
        # 如果当前张量的长度小于最大长度，则需要进行填充
        if lens[i] < max_len:
            # 创建一个全零的张量，其形状为 [max_len - lens[i]] + hid，并将其移动到与原始张量相同的设备上
            # 然后将原始张量与全零张量在第一个维度上进行拼接
            tmp = torch.cat(
                [tensors[i], torch.zeros([max_len - lens[i]] + hid, dtype=dtype).to(device)],
                dim=0
            )
        else:
            # 如果当前张量的长度等于最大长度，则不需要填充，直接使用原始张量
            tmp = tensors[i]

        # 将处理后的张量添加到输出列表中
        output.append(tmp)

    # 将输出列表中的所有张量在第一个维度上进行堆叠，形成一个批次张量
    output = torch.stack(output, 0)

    # 返回填充后的张量
    return output
