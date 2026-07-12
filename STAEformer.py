import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F

try:
    from model.Wo_AsymmetricCNN import Wo_AsymmetricCNN
    from model.Wo_ContinuityCompensation import Wo_ContinuityCompensation
    from model.Wo_TimeSplitBranches import Wo_TimeSplitBranches
    from model.Wo_VolatilityAwareness import Wo_VolatilityAwareness
    from model.Wo_VolatilityGuidedFusion import Wo_VolatilityGuidedFusion
except ModuleNotFoundError:
    from Wo_AsymmetricCNN import Wo_AsymmetricCNN
    from Wo_ContinuityCompensation import Wo_ContinuityCompensation
    from Wo_TimeSplitBranches import Wo_TimeSplitBranches
    from Wo_VolatilityAwareness import Wo_VolatilityAwareness
    from Wo_VolatilityGuidedFusion import Wo_VolatilityGuidedFusion


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

class VolatilityAwareEnhancer(nn.Module):
    """
    在完整时间维 T 上先做波动检测，再对输入特征做增强。

    Input:
        x: [B, T, N, D]
    Output:
        out:       [B, T, N, D]
        vol_mask:  [B, T, N, 1]
        vol_score: [B, T, N, 1]
    """

    def __init__(
        self,
        model_dim,
        window_size=3,
        hidden_ratio=0.25,
        dropout=0.1,
        eps=1e-6,
    ):
        super().__init__()

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")

        hidden_dim = max(8, int(model_dim * hidden_ratio))

        # 滑动窗口大小
        self.window_size = window_size
        # 极小数
        self.eps = eps

        self.pre_norm = nn.LayerNorm(model_dim)

        self.score_proj = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # raw / avg / max 三类波动聚合的可学习融合
        self.pool_logits = nn.Parameter(torch.zeros(3))
        # 定义了一个长度维度3的可学习向量[0,0,0]

        # 控制软 mask 的平滑度
        self.mask_temp = nn.Parameter(torch.tensor(1.0))
        # 定义了一个可学习的标量参数初始值为1

        # 波动引导的特征级门控与偏置
        self.gate_proj = nn.Sequential(
            nn.Linear(model_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.bias_proj = nn.Sequential(
            nn.Linear(model_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )

        # 创建一个可以被模型训练更新的数字, 初始值是1.0
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, return_aux=False):
        B, T, N, D = x.shape
        residual = x

        # Stage1-1: Volatility Encoding=================================================================================
        #  1. 预归一化 : 先把每个时间步的特种分布拉倒更稳定的尺度，方便后面判断“波动”
        x_norm = self.pre_norm(x)  # LayerNorm层 [B, T, N, D]

        #  两层MLP把每个时间步的D维度特种压缩为一个标量分数-> 这里的score不是原始流量
        #  而是模型学出来的 波动代理分数
        score = self.score_proj(x_norm).squeeze(-1)  # [B, T, N] 也就是：每个样本、每个时间步、每个节点，都对应一个标量分数。
        # x [B,T,N,D] -> Linear [B,T,N,HiddenDim] -> GELU -> Linear [B,T,N,1] -> squeeze(-1) 去掉最后一个维度-> [B,T,N]

        # 维度变形，为了方便变成1D-CNN所需要的格式
        # s: [B, T, N] -> Permute -> [B,N,T] -> reshape -> [B*N, 1, T]
        s = score.permute(0, 2, 1).reshape(B * N, 1, T)  # [B*N, 1, T]
        # 这里就变成了对于每一个节点序列，都有一个长度为T的分数序列

        # 2. 构造基础波动强度 diff
        diff = torch.zeros_like(s)  # [B*N, 1, T] 都是0. 先创建一个空的波动差分张量, 然后再填充时间差分值
        # s[:, :, 1: ]表示从[B*N,1,T]的最后一个维度中 取第1个时间步 到 最后[s1, s2, s3, ..., sT-1]
        # s[:, :, :-1]表示从[B*N,1,T]的最后一个维度中 取第0个时间步 到 倒数第二个时间步[s0, s1, s2, ..., sT-2]
        # s-s表示为[s1 - s0, s2 - s1, s3 - s2, ..., sT-1 - sT-2]
        # abs是绝对值
        # 那么diff表示的就是每个时间步相比于前一个时间步的变化幅度
        diff[:, :, 1:] = torch.abs(s[:, :, 1:] - s[:, :, :-1])
        # print(diff.shape)
        # 为什么不仅用diff? 例如s = [2.0, 2.1, 5.0, 5.1, 5.2]去计算diff得到diff = [0, 0.1, 2.9, 0.1, 0.1]
        # 可能只会认为t=2是突变点, 但是从交通状态来看, t=2,3,4其实都是较高值
        # [5.0,5.1,5.2]可能已经是拥堵状态，而不是仅有t=2一个时间步是重要的. 因此需要补充dev突变

        # 3. 构造局部波动 dev
        # 虽然diff能够看到前后两个时间步之间是否突变
        # 但是不能很好的判断当前点是否偏离局部趋势,
        # 例如[2.0, 2.1, 2.2, 2.3, 2.4]是一个平稳上升的趋势, [2.0, 2.1, 5.0, 2.2, 2.1]中间的5.0属于偏离，像一个局部突变或异常波动
        trend = F.avg_pool1d(
            s,
            kernel_size=self.window_size,  # window_size=3 表示滑动窗口大小, 表示每次看3个时间片[t-1, t, t+1]
            stride=1,  # 表示窗口每次移动一个时间步, 这样可以让每个时间步对应的值的含义是：得到一个对应的局部趋势值
            padding=self.window_size // 2,
            # 表示在时间序列两端补充值，让输出的长度仍然=T->[2.0, 3.5, 1.0, 1.2, 5.0]->pad->[0, 2.0, 3.5, 1.0, 1.2, 5.0, 0] 因此第一个和最后一个时间步也可以算局部平均
        )
        dev = torch.abs(s - trend)  # dev表示计算当前点与局部趋势之间的绝对值偏差, abs是绝对值
        # dev值越大代表当前时间步越偏离局部趋势，值越小说明当前时间步越接近附近!!!的时间段的平均状态.

        # stage1-2: Volatility Scoring=================================================================================
        # 1. 多尺度聚合
        # 到这里, base_vol已经综合了相邻时间突变强度diff和局部趋势偏离强度dev.
        # 这里将base_vol分成三种视角进行聚合(原始, 局部平均, 局部最大)
        base_vol = diff + dev  # 2和3的高波动点的融合判断(前面算的diff和这里的dev两种波动)
        # 为什么不只用dev? 如果s = [2.0, 2.5, 3.0, 3.5, 4.0] 这是一个持续上升的序列, dev也会变大, 但是diff = [0, 0.5, 0.5, 0.5, 0.5] 则会告诉模型, 这个节点的状态是持续变化的.
        # 因此 dev和diff互补

        # 2.1 计算base_vol 局部平均波动强度, 因此vol_avg表示当前时间步附近一小段时间内的平均波动水平. 更关注当前时间点附近整体波动是否较强
        # base_vol相比于diff来说，作用是把局部窗口内的波动做平均, 如果一个时间点附近整体都比较波动, 那么vol_avg会比较大.
        # 如果只是一个孤立的尖峰值, 那么平均之后会被适当平滑.
        vol_avg = F.avg_pool1d(
            base_vol,  # [B*N, 1, T]
            kernel_size=self.window_size,
            stride=1,
            padding=self.window_size // 2,
        )

        # 2.2 计算vol_max 局部窗口内的最大波动强度, 求的是当前时间步附近是否出现过强的波动
        vol_max = F.max_pool1d(
            base_vol,
            kernel_size=self.window_size,
            stride=1,
            padding=self.window_size // 2,
        )

        mix = torch.softmax(self.pool_logits, dim=0)  # pool是一个可学习参数长度为3->[a,b,c]. 分别对应三种分量的权重.
        # 当他们经过softmax之后可以保证mix[0], minx[1], minx[2]>=0. 并且softmax之后a+b+c=1

        # mix的初始值是[0,0,0],经过softmax之后为[1/3,1/3,1/3]. 也就是说一开始是对下面公式中三种突变是一视同仁的.
        # 训练过程中pool_logits会被反向传播自动更新
        q = mix[0] * base_vol + mix[1] * vol_avg + mix[2] * vol_max

        # [B*N, 1, T] -> [B, T, N]  # 表示第B个样本, 第T个时间步, 第N个节点的综合波动强度
        q = q.reshape(B, N, T).permute(0, 2, 1)

        # 3. vol_score 突出为波动水平 vol_mask 突出波动位置与关注强度
        # vol_score(可正可负, 保留了方向与强度, >0当前时间步比该节点平时更波动; = 0 当前时间步接近该节点的平均波动水平; <0 当前时间步比该节点平时更平稳)
        # vol_mask 当前时间步应该被关注的强度

        # 3.1 vol_score计算  突出波动水平
        q_mean = q.mean(dim=1, keepdim=True)  # T窗口内各个节点的平均波动水平, [B, T, N]->[B, 1, N] 表示对每个样本, 每个节点-> 在时间维度上求平均; Keepdim是为了让维度不变，不加的话会变成[B,N]
        q_std = q.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)  # 计算标准差, clamp_min是表示标准差不能低于eps， 防止后面除0
        q_norm = (q - q_mean) / q_std  # [B, T, N]  # 归一化, 相对波动程度: 当前时间步的波动强度相比于该节点自己的平均波动强了多少个标准差
        # q_norm>0, 当前时间步波动高于该节点的平均水平; q_norm=0: 当前时间步接近平均波动水平; q_norm<0: 当前时间步波动低于该节点的平均水平
        vol_score = q_norm.unsqueeze(-1)  # [B, T, N, 1]  增加一个维度

        # 3.2 vol_mask计算  突出位置
        # 波动软权重vol_mask，告诉模型哪里更重要
        # temp越大, vol越小, 代表平稳点
        # temp越小, vol越大, 代表波动点
        # vol=0.2 平稳点, vol=0.5 普通点, vol=0.9: 波动点
        temp = F.softplus(self.mask_temp) + self.eps  # 可训练 softplus=log(1+e^x)+eps 要确保temp作为被除数>0
        vol_mask = torch.sigmoid(q_norm / temp).unsqueeze(-1)  # [B, T, N, 1]  # unsqueeze是增加一个维度,-1代表在最后一维后面增加.

        # stage 1-3: Guided Enhancement=================================================================================
        # 1. 生成波动引导信号
        # 拼接操作: x_norm[B,T,N,D]: 最开始的输入特征. vol_mask[B,T,N,1]: 0-1的软波动权重. vol_score[B,T,N,1]: 标准化后的波动强度, 可正可负
        fuse_in = torch.cat([x_norm, vol_mask, vol_score], dim=-1)  # [B,T,N, D+2]

        # 2. 源于特征，生成一个门控, 用Sigmoid压缩特征0-1之间
        gate = torch.sigmoid(self.gate_proj(fuse_in))  # [B,T,N,D+2]->[B,T,N,D*ratio]->[B,T,N,D] 决定每个时间步, 每个时间点, 每个特种维度保留或者放大多少信息. 决定哪些特征应该被保留，增强，抑制
        bias = self.bias_proj(fuse_in)  # 偏置值设置, gate负责选择哪些原有的特征, bias 负责补充新的修正信息.
        # 定义的可训练参数  alpha更大, 说明模型更依赖筛选后的原始特征, beta更大, 说明模型更依赖波动修正项
        alpha = F.softplus(self.alpha)  # softplus=log(1+e^x): 这一步把self.alpha转成正数   # alpha: 控制筛选后特征的整体增强幅度 控制gate*x_norm这部分原始特征增强的强度
        beta = F.softplus(self.beta)    # softplus=log(1+e^x): 这一步把self.beta转成正数    # beta: 控制修正信息的强度  控制bias这部分额外修正信息的强度

        # 3. 门控融合
        # gate和bias从fuse_in得来, gate*x_norm代表对原始归一化特种做动态筛选，决定x_norm在每个样本，时间步，节点，特征维度上应该能通过多少;
        # bias: 提供额外的波动修正信息
        update = alpha * (gate * x_norm) + beta * bias

        # 把增强量dropout(update)反哺给原始特征, dropout用于随机丢弃一部分增强信息, 防止模型过度依赖波动增强模块
        out = self.norm(residual + self.dropout(update))

        if return_aux:
            return out, vol_mask, vol_score
        return out

class SharedGRUBranch(nn.Module):
    """
    共享 GRU 骨干:
        GRU -> Linear(optional)
    Input / Output:
        [B*N, L, D] -> [B*N, L, D]
    """

    def __init__(self, model_dim, gru_hidden_dim):
        super().__init__()

        self.gru = nn.GRU(
            input_size=model_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.proj = (
            nn.Identity()
            if gru_hidden_dim == model_dim
            else nn.Linear(gru_hidden_dim, model_dim)
        )

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.proj(x)
        return x

class TimeSplitGRUCNN(nn.Module):
    """
    最优版时序块:
    1) 全时间维波动识别增强
    2) 重叠二分
    3) 共享 GRU 主干
    4) 非对称 CNN:
       - 前分支: 较大卷积核, 偏趋势
       - 后分支: 较小卷积核 + 膨胀, 偏突变
    5) 波动引导门控融合
    6) 全时域连续性补偿
    7) Residual + FFN

    Input / Output:
        [B, T, N, D] -> [B, T, N, D]
    """

    def __init__(
        self,
        model_dim,
        feed_forward_dim=256,
        gru_hidden_dim=None,
        overlap=2,
        trend_kernel_size=5,
        event_kernel_size=3,
        event_dilation=2,
        full_kernel_size=3,
        dropout=0.1,
        vol_window_size=3,
    ):
        super().__init__()

        if gru_hidden_dim is None:
            gru_hidden_dim = model_dim
        if trend_kernel_size % 2 == 0:
            raise ValueError("trend_kernel_size must be odd.")
        if event_kernel_size % 2 == 0:
            raise ValueError("event_kernel_size must be odd.")
        if full_kernel_size % 2 == 0:
            raise ValueError("full_kernel_size must be odd.")

        self.model_dim = model_dim
        self.overlap = overlap

        self.pre_norm = nn.LayerNorm(model_dim)

        self.vol_enhancer = VolatilityAwareEnhancer(
            model_dim=model_dim,
            window_size=vol_window_size,
            dropout=dropout,
        )

        # 共享 GRU 主干
        self.shared_gru = SharedGRUBranch(
            model_dim=model_dim,
            gru_hidden_dim=gru_hidden_dim,
        )

        # 段标识嵌入
        self.seg_emb_first = nn.Parameter(torch.zeros(1, 1, model_dim))  # 初始化一个1,1,model_dim维度的可训练权重, 后面可以广播
        self.seg_emb_second = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.seg_emb_first, mean=0.0, std=0.02)   # 把上面定义的两个段标识嵌入参数进行初始化
        nn.init.normal_(self.seg_emb_second, mean=0.0, std=0.02)

        # 前分支：较大卷积核，更适合看连续局部的上下文, 偏趋势建模
        self.trend_conv = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=trend_kernel_size,  # 5
            padding=trend_kernel_size // 2,
        )

        # 后分支：较小卷积核 + dilation，偏近期突变. 更偏向捕捉跳变、间隔变化、突发事件
        self.event_conv = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=event_kernel_size,  # 3
            padding=event_dilation * (event_kernel_size // 2),
            dilation=event_dilation,  # 2 ,普通卷积卷积核这个默认等于1, 这边设置的2, 实际看的位置变成[t-2, t, t+2]
        )

        self.branch_act = nn.GELU()
        self.branch_dropout = nn.Dropout(dropout)

        # 基于两个分支的波动统计量生成分支级重权重
        hidden_dim = max(8, model_dim // 8)
        self.branch_reweight = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

        # 重叠区门控融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * model_dim + 2, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, model_dim),
            nn.Sigmoid(),
        )

        # 全时域连续性补偿
        self.full_conv = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=full_kernel_size,
            padding=full_kernel_size // 2,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(model_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(model_dim)

    def _get_split_points(self, T):
        # T=12
        mid = T // 2  # mid = 6
        # overlap = 2
        actual_overlap = min(self.overlap, max(T - 1, 0))  # min (2, max(11,0))=2
        first_end = min(T, mid + actual_overlap)  # min(12, 6+2)=8
        second_start = max(0, mid - actual_overlap)  # max(0, 6-2)=4

        # 保证两个分支至少覆盖全时间轴，且存在重叠/接壤
        if second_start > first_end:
            second_start = first_end

        return first_end, second_start

    def _apply_branch_conv(self, x, conv):
        # x: [B*N, L, D]
        x = conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.branch_act(x)
        x = self.branch_dropout(x)
        return x

    def forward(self, x, return_aux=False):
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        # Stage 1: Volatility Preprocessing======================================================
        # 1. 准备原始残差特征residual
        residual = x  # 原始特种作为残差

        # 2. 归一化
        x = self.pre_norm(x)  # 预归一化 LayerNorm [B, T, N, D]

        # 3.1 调用波动增强模块, 获取输出. 全时间维波动增强
        x, vol_mask, vol_score = self.vol_enhancer(x, return_aux=True)  # [B,T,N,D], [B,T,N,1], [B,T,N,1]

        # 3.2 三个输出的维度调整 [B,T,N,D] -> [B*N,T,D]
        x_bn = x.transpose(1, 2).reshape(B * N, T, D)  # [B*N,T,D]
        vol_mask_bn = vol_mask.transpose(1, 2).reshape(B * N, T, 1)  # [B*N,T,1]
        vol_score_bn = vol_score.transpose(1, 2).reshape(B * N, T, 1)  # [B*N,T,1]

        # Stage2-1: Overlapped Segmentation with Branch Embedding======================================================
        # 1. 计算T1和T2划分的下标位置, 用于划分原始特征序列: 对原始特征划x_bn划分前后特征得到x_first 前8个; 和x_second 后8个, 中间有重叠部分
        first_end, second_start = self._get_split_points(T)  # first_end=8, second_start=4

        # 2. 划分前后序列, 并叠加身份ID用于Shared GRU
        # 保留了重叠区, 避免中间断裂
        # 另外seg_emb_first和seg_emb_second是为了作为可训练的身份标识, 因为这里设置的是共享GRU, 需要一个身份标识来区分, 减少训练时间损耗
        # 原理：经过反向传播之后, seg_emb_first和seg_emb_second 会让GRU知道这两个向量会变成更有区分性的身份标识
        x_first = x_bn[:, :first_end, :] + self.seg_emb_first  # 前分支: [:, 0-8, : ] 覆盖了0-7  [B*N, 8, D]
        x_second = x_bn[:, second_start:, :] + self.seg_emb_second  # 后分支: [:, 4:12, : ] 覆盖了 4-11  [B*N, 8, D]

        # Stage2-2 双分支时序建模 Trend-Event Dual-branch Modeling======================================================
        # 1. Shared GRU: 把前后两段的特征都放到GRU里面去
        h_first = self.shared_gru(x_first)  # [B*N, 8, D]
        h_second = self.shared_gru(x_second)  # [B*N, 8, D]

        # 2. Trend-Event CNN:
        out_first = self._apply_branch_conv(h_first, self.trend_conv)  # [B*N, 8, D] 2720,8,152 # 前分支：较大卷积核, 更适合看连续局部上下文
        out_second = self._apply_branch_conv(h_second, self.event_conv)  # [B*N, 8, D]          # 后分支：较小卷积核 + dilation，捕捉跳变, 间隔变化, 突发事件

        # 2.1 分支波动重权重
        # 如果某个节点最近时间段波动明显, 后分支权重可能更高
        # 如果前半段特征的趋势变化明显, 前分支权重可能更高
        # 前半段用趋势 CNN 看平稳变化；
        # 后半段用事件 CNN 看突发变化；
        # 然后看哪个分支对应的时间段波动更明显；
        # 哪个分支更重要，就把哪个分支的特征放大一些。让模型不是固定地平均使用前后两个时间分支，而是根据当前样本、当前节点的波动情况，自适应地调整趋势分支和突变分支的重要性。
        stat_first = vol_score_bn[:, :first_end, :].abs().mean(dim=1)   # [B*N,1]  计算前分支覆盖的时间段里的平均波动强度. abs是绝对值, 这边是关注波动程度有多明显, 所以取绝对值[B*N,8,1]->[B*N,1]
        stat_second = vol_score_bn[:, second_start:, :].abs().mean(dim=1)  # [B*N,1]  计算后分支覆盖的时间段里的平均波动强度
        branch_stat = torch.cat([stat_first, stat_second], dim=-1)      # [B*N,2] 拼接两个分支的波动统计 -> [B*N,1] cat [B*N,1]->[B*N,2]
        branch_weight = self.branch_reweight(branch_stat)               # [B*N,2] -> [B*N,2] 通过一个MLP计算两个分支的权重 最终的brach_weight包含两个权重[前分支first的权重, 后分支second的权重] 让模型判断应该更重视前分支还是后分支

        # 2.2 非对称CNN的输出分别乘以各自的权重
        # 这里为什么要+1 -> 假设branch_weight=0.6  1+0.6=1.6那输出就变成了out_first*1.6
        # [B*N,2]->branch_weight[:, 0:1]取出第一列->[B*N,1]->unsqueeze(-1)->[B*N,1,1]  然后通过out_first广播到同一个维度->[B*N,8,D]
        out_first = out_first * (1.0 + branch_weight[:, 0:1].unsqueeze(-1))
        out_second = out_second * (1.0 + branch_weight[:, 1:2].unsqueeze(-1))

        # Stage2-4: Mask-gated Fusion===============================================
        # 1.1 初始化0值变量, 准备分离之前的特征
        full_first = x_bn.new_zeros(B * N, T, D)  # 初始化一个维度和x_bn一样的全0特征 准备把前分支结果放到0-7
        full_second = x_bn.new_zeros(B * N, T, D)  # 初始化一个维度和x_bn一样的全0特征 准备把后分支结果放到4-11
        valid_first = x_bn.new_zeros(B * N, T, 1)  # 初始化一个维度和x_bn一样的全0特征 准备标记前分支有效区域 [1,1,1,1,1,1,1,1,1,0,0,0,0]
        valid_second = x_bn.new_zeros(B * N, T, 1)  # 初始化一个维度和x_bn一样的全0特征 标记后分支有效区域 [0,0,0,0,1,1,1,1,1,1,1,1]

        # 1.2 将两个分支回填到完整时间轴 因为有重合的区域，需要进一步处理
        full_first[:, :first_end, :] = out_first
        full_second[:, second_start:, :] = out_second
        valid_first[:, :first_end, :] = 1.0
        valid_second[:, second_start:, :] = 1.0

        # valid_first:   1 1 1 1 1 1 1 1 0 0 0 0
        # valid_second:  0 0 0 0 1 1 1 1 1 1 1 1
        # 相乘结果 overlap_mask :       0 0 0 0 1 1 1 1 0 0 0 0
        overlap_mask = valid_first * valid_second  # [B*N,T,1] 这里是把两个0 1 标记的 标记向量哈达玛积，剩余的还是1的就是重叠区域的标记

        # valid_first:      1 1 1 1 1 1 1 1 0 0 0 0
        # 1-overlap_mask:   1 1 1 1 0 0 0 0 1 1 1 1
        # 相乘结果:           1 1 1 1 0 0 0 0 0 0 0 0  first_only_mask
        first_only_mask = valid_first * (1.0 - overlap_mask)  # 不包含重叠区域

        # valid_second      0 0 0 0 1 1 1 1 1 1 1 1
        # 1-overlap_mask:   1 1 1 1 0 0 0 0 1 1 1 1
        # 相乘结果            0 0 0 0 0 0 0 0 1 1 1 1  second_only_mask
        second_only_mask = valid_second * (1.0 - overlap_mask)  # 不包含重叠区域

        # 2. 重叠区门控融合 前半完整的和后半完整的门控融合，重要的是得到了重叠的中间部分，后面用掩码把他们筛选出来就行了
        # fusion_in: 拼接了前分支的特征, 后分支的特征, vol_mask, vol_score
        fusion_in = torch.cat([full_first, full_second, vol_mask_bn, vol_score_bn], dim=-1)  # [B*N,T,2D+2]
        # gate基于融合特征生成门控权重 动态的决定当前重叠区更应该相信哪个分支，因为前后两个分支CNN的建模偏好不同
        gate = self.fusion_gate(fusion_in)  # [B*N,T,D]
        # 门控融合了特征，后面主要通过重合区掩码overlap_mask来提取这一部分
        fused_overlap = gate * full_first + (1.0 - gate) * full_second

        # 得到完整的时序特征
        out = (first_only_mask * full_first + second_only_mask * full_second + overlap_mask * fused_overlap)  # [B*N,T,D]

        # Stage3: Temporal Refinement===============================================
        # 1. CNN全时域连续性补偿
        # 虽然前面做了重叠融合，但是完整的序列还是由两个分支拼接回来的, 所以用CNN做一个全时域连续性补偿, 减少!拼接边界处的不连续!
        out = self.full_conv(out.transpose(1, 2)).transpose(1, 2)  # 1D-CNN [B*N,T,D]
        out = self.branch_act(out)  # GELU

        # 2. 再做一次 波动重标定 让模型在输出预测结果前再次突出可能影响未来预测的波动点
        out = out * (1.0 + vol_mask_bn)

        # [B*N,T,D] -> [B,T,N,D] 转变维度
        out = out.reshape(B, N, T, D).transpose(1, 2)

        # 3. Residual + FFN  原始特征residual 残差链接 增强的时序波动信息
        out = self.ln1(residual + self.dropout1(out))

        residual_ffn = out
        out = self.feed_forward(out)
        out = self.ln2(residual_ffn + self.dropout2(out))

        if return_aux:
            aux = {
                "vol_mask": vol_mask,                            # [B,T,N,1]
                "vol_score": vol_score,                          # [B,T,N,1]
                "branch_weight": branch_weight.view(B, N, 2),    # [B,N,2]
                "split_points": (first_end, second_start),
            }
            return out, aux

        return out


class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        # 核心优化模块 / 消融实验入口
        # 默认使用完整模型 TimeSplitGRUCNN。
        # 跑消融时只需要把下面这一行改成对应类名，下面的参数保持不变即可：
        #   Wo_VolatilityAwareness
        #   Wo_VolatilityGuidedFusion
        #   Wo_TimeSplitBranches
        #   Wo_AsymmetricCNN
        #   Wo_VolatilityGuidedFusion
        #   Wo_ContinuityCompensation
        self.time_gru_cnn = Wo_VolatilityGuidedFusion(
            model_dim=self.model_dim,
            feed_forward_dim=feed_forward_dim,
            gru_hidden_dim=self.model_dim,
            overlap=2,              # 对 T=12 比较合适
            trend_kernel_size=5,    # 前分支偏趋势
            event_kernel_size=3,    # 后分支偏突变
            event_dilation=2,
            full_kernel_size=3,
            dropout=dropout,
            vol_window_size=3,
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]  # D8: 16 BatchSize大小
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]  # [16,12,170] [B,T,N] 取第二个通道的：time of day, 日内时间
            # print(tod.shape)
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]  # [16,12,170] [B,T,N] 取第三个通道的：day of week, 星期信息
            # print(dow.shape)
        x = x[..., : self.input_dim]  # [16,12,170,3] Ef=[B,T,N,input_dim=3]

        # 构造
        x = self.input_proj(x)  # [16,12,170,24] (batch_size, in_steps, num_nodes, input_embedding_dim) [B,T,N,In_ED=24]
        features = [x]  # 这里是构建了一个列表，后面所有特征append进去
        # print(features[0].shape) [16,12,170,24]

        # 把一天中的第几个时刻 变成 一个可学习的向量
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)

        # 把周几变成可学习向量
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)

        # 给每个节点一个固定的身份向量，表示“这是哪个传感器 / 路段”。
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)

        # 给第几个时间步+第几个节点分配一个可学习向量，这是STAEFormer的核心判断
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        # print(x.shape)  # [B,T,N,Model_dim]

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)  # [B,T,N,Model_dim]

        if self.training:
            x = self.time_gru_cnn(x)  # [B,T,N,D]
            self.latest_vol_mask = None
            self.latest_vol_score = None
        else:
            x, aux = self.time_gru_cnn(x, return_aux=True)  # [B,T,N,D]
            self.latest_vol_mask = aux["vol_mask"].detach()
            self.latest_vol_score = aux["vol_score"].detach()

        for attn in self.attn_layers_s:
            x = attn(x, dim=2)  # [B,T,N,Model_dim]
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # [B,T,N,D]->[B,N,T,D] (batch_size, num_nodes, in_steps, model_dim)

            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            # [B,N,T,D]->[B,N,T*D][16,170,1824]

            out = self.output_proj(out)
            # [B,N,T*D]->[B,N,out_steps*output_dim]=[B,N,12*1]=[B,N,12]  [16,170,12]

            out = out.view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            # [B,N,out_steps,1]

            out = out.transpose(1, 2)
            # (batch_size, out_steps, num_nodes, output_dim) [B,Out,N,out_dim]=[16,12,170,1]
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out

if __name__ == "__main__":
    model = STAEformer(207, 12, 12)
    summary(model, [64, 12, 207, 3])
