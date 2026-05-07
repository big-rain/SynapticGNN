
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import spmm
from torch_geometric.utils import to_torch_sparse_tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm


import torch
import torch.nn as nn
import warnings
# 过滤掉关于 CSR Beta 状态的警告
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

def get_dynamics_config(dataset_name: str):
    """
    针对 PeMS03/04/07/08 优化的动力学参数配置工厂
    核心逻辑：平衡“灵敏度”与“鲁棒性”
    """
    dataset_name = dataset_name.upper()


    # --- 基准默认参数 (Baseline) ---
    config = {
        'c1': 1.0,
        'log_neg_c3': -1.2,   # c3 ≈ 0.3。增加约束力，让波形回归 FHN 标准的“尖峰”形态
        'c0': 0.0,
        'alpha': 0.5,         # 稍微调高一点，加强 V-W 耦合，有助于降低 VF-MSE
        'beta': 0.05,         # 给一点点偏置，打破绝对对称，有利于启动震荡
        'log_tau_v': -1.6,    # tau_v ≈ 0.2。与 dt=0.1 配合，保证数值稳定性
        'log_tau_w': 1.6,     # tau_w ≈ 5.0。拉开尺度差距
        'v_threshold': 0.3,   # 进一步降低，让模型对微弱的 I_ext 信号更敏感
        'dt': 0.1
    }
    # --- 各数据集定向调优 ---

    if 'PEMS08' in dataset_name:
        # [PeMS08] 极度平稳，Delta Mean +1.52
        # 目标：低通滤波，防止拟合训练集噪声
        config.update({
            'c1': 1.0,          # 降低兴奋性，模型变得“沉稳”
            'beta': 0.05,       # 提高基准，适应极小的正向偏移
            'log_tau_v': -0.6,  # 惯性增加 (tau_v ≈ 0.55)，过滤高频抖动
            'v_threshold': 5.0  # 提高阈值，非极端情况不触发脉冲
        })
    elif 'FHN' in dataset_name:
        config.update({
            # --- 核心对齐 ---
            'dt': 0.2,  # 必须设为 0.2！(dt_gen * sample_step)

            # --- 动力学参数调优 ---
            'c1': 1.0,
            'log_neg_c3': -1.1,  # c3 ≈ 0.33。你的生成器 dv = v - v^3/3，c3=1/3=0.33 是标准值
            'c0': 0.0,
            'alpha': 1.0,  # 你的生成器 dw = eps * (v + a - b*w)，这里 alpha 对应变量耦合系数
            'beta': 0.7,  # 你的生成器 a_vec 均值 0.7，模型需要这个偏置来对齐起跳点

            # --- 时间尺度因子 ---
            'log_tau_v': 0.0,  # tau_v ≈ 1.0。因为你的 dv 公式里系数就是 1
            'log_tau_w': 2.5,  # tau_w = 1 / eps = 1 / 0.08 = 12.5。log(12.5) ≈ 2.5
            # 你的 eps 是 0.08，说明 w 极其慢，必须把 tau_w 调大！

            'v_threshold': 0.3,
        })

    elif 'Yemini2021' in dataset_name:
        config.update({
    'dt': 0.5,            # 直接对齐 2Hz 采样间隔
    'log_tau_v': -0.7,    # 增大 tau_v，适应 0.5s 的大步长
    'log_tau_w': 2.3,     # 保持慢变量的迟滞性
    'c1': 0.6,
    'log_neg_c3': -1.2,
    'v_threshold': 0.1,
    'alpha': 0.15,        # 降低耦合，增加低频稳定性
    'beta': 0.01
})


    elif 'PEMS03' in dataset_name:
        # [PeMS03] 数据最脏，Delta Mean -8.90 (严重低偏)
        # 目标：极高灵敏度，快速捕捉大幅跌落后的恢复
        config.update({
            'c1': 1.35,         # 增强兴奋性，对抗负向水位差
            'log_tau_v': -1.2,  # 极速响应 (tau_v ≈ 0.30)，追踪突发流量
            'v_threshold': 4.0, # 降低阈值，对变化更敏感
            'beta': 0.00        # 保持极低偏置，适应低水位环境
        })

    elif 'METR-LA' in dataset_name or 'METRLA' in dataset_name:
        # [METR-LA] 洛杉矶速度数据，特点：非线性突变多，数值分布在 0-70
        # 目标：平滑速度噪声，捕捉“速度崩溃”后的缓慢恢复
        config.update({
            'c1': 1.1,  # 适中兴奋性：速度变化比流量更具“惯性”
            'log_neg_c3': -2.0,  # 增强稳定性：防止速度预测出现非物理的剧烈跳变
            'alpha': 0.4,  # 较慢的追踪：w 追踪 v 变慢，产生更长的“记忆效应”
            'beta': 0.08,  # 较高的基准：适应速度数据较高的平均水位
            'log_tau_v': -0.7,  # 适中响应 (tau_v ≈ 0.50)：过滤高速行驶时的微小速度抖动
            'log_tau_w': 4.0,  # 极长记忆：捕捉早晚高峰这种长达数小时的状态演化
            'v_threshold': 5.2,  # 较高阈值：仅在速度发生显著跌落（拥堵）时触发动力学突变
            'dt': 0.1
        })

    elif 'PEMSBAY' in dataset_name:
        # [PEMS-BAY] 湾区速度数据，特点：极度平滑，数值分布在 0-70mph
        # 目标：保持高稳定性，仅对剧烈的速度崩溃（拥堵）产生反应
        config.update({
            'c1': 1.05,  # 进一步降低兴奋性：速度数据不应有剧烈自发波动
            'log_neg_c3': -2.0,  # 增强三次方抑制：确保系统在 65mph 左右极度稳定
            'alpha': 0.45,  # 较慢的追踪：w 追踪 v 变慢，增加系统的平滑惯性
            'beta': 0.12,  # 显著提高基准：适应速度数据的高水位分布
            'log_tau_v': -0.5,  # 反应变慢 (tau_v ≈ 0.60)：过滤传感器的小幅读数抖动
            'log_tau_w': 4.2,  # 极长记忆：捕捉由于早晚高峰引起的长时间状态偏移
            'v_threshold': 5.8,  # 极高阈值：速度数据背景噪音多，需拉高脉冲触发点
            'dt': 0.1
        })

    elif 'PEMS04' in dataset_name:
        # [PeMS04] 流量大，Delta Mean +9.64 (严重高偏)
        # 目标：动态范围优化，适应整体上移的流量
        config.update({
            'c1': 1.25,         # 略微增强兴奋性
            'beta': 0.10,       # 显著提高基准偏置，支撑高水位分布
            'log_tau_v': -1.1,  # 快速响应
            'v_threshold': 5.5  # 配合高水位，适当拉高脉冲触发点
        })

    elif 'PEMS07' in dataset_name:
        # [PeMS07] 规模庞大 (883节点)，Delta Mean -6.43
        # 目标：空间平滑与局部敏感的平衡
        config.update({
            'c1': 1.15,         # 中庸的兴奋性，防止在大图中产生数值风暴
            'log_tau_v': -0.8,  # 适中的反应速度
            'v_threshold': 4.8, # 略微拉高，配合 883 节点复杂的空间耦合
            'dt': 0.1           # 建议保持 0.1 以确保 883 节点的计算速度
        })

    return config


class OptimalNodeDynamics(nn.Module):
    def __init__(self, channels: int, config: dict):
        super().__init__()
        self.channels = channels
        self.dt = config.get('dt', 0.1)

        # 使用 config 中的值进行初始化
        self.log_neg_c3 = nn.Parameter(torch.ones(channels) * config['log_neg_c3'])
        self.c1 = nn.Parameter(torch.ones(channels) * config['c1'])
        self.c2 = nn.Parameter(torch.zeros(channels))
        self.c0 = nn.Parameter(torch.zeros(channels))

        self.alpha = nn.Parameter(torch.ones(channels) * config['alpha'])
        self.beta = nn.Parameter(torch.ones(channels) * config['beta'])

        self.log_tau_v = nn.Parameter(torch.ones(channels) * config['log_tau_v'])
        self.log_tau_w = nn.Parameter(torch.ones(channels) * config['log_tau_w'])

        self.v_threshold = nn.Parameter(torch.ones(channels) * config['v_threshold'])
        self.v_rest = nn.Parameter(torch.zeros(channels) * 0.5)

    def forward(self, V, W, I_ext=0, I_gap=0):
        """
        V, W: 当前状态 (Batch, Channels)
        I_ext: 节点自身特征注入
        I_gap: 邻居突触耦合输入
        """
        # 转换对数参数
        c3 = -torch.exp(self.log_neg_c3)
        tau_v = torch.exp(self.log_tau_v)
        tau_w = torch.exp(self.log_tau_w)

        # --- A. 计算连续动力学演化 ---
        # 兴奋驱动 f(V)
        f_v = c3 * torch.pow(V, 3) + self.c2 * torch.pow(V, 2) + self.c1 * V + self.c0

        # 微分更新
        dv_dt = (f_v - W + I_ext + I_gap) / tau_v
        dw_dt = (self.alpha * V + self.beta - W) / tau_w

        # 改进：对 dv_dt 进行梯度缩放，防止 ODE 步长过大崩溃
        dv_dt = torch.tanh(dv_dt / 10.0) * 10.0

        # 在 OptimalNodeDynamics.forward 中建议加入
        V_next = V + self.dt * dv_dt
        W_next = W + self.dt * dw_dt

        # 数值稳定性保护：防止 V 逃逸到不合理的范围
        V_next = torch.clamp(V_next, min=-10.0, max=10.0)

        spike_gate = torch.sigmoid(3.0 * (V_next - self.v_threshold))
        V_final = (1 - spike_gate) * V_next + spike_gate * (V_next * 0.8)  # 饱和衰减
        W_final = W_next + spike_gate * 0.1  # 增加较小的适应性惩罚

        return V_final, W_final


class SynapticMessagePassing(MessagePassing):
    def __init__(self, saturation_limit=2.0):
        super().__init__(aggr='add')
        self.g = torch.nn.Parameter(torch.Tensor([1.0]))
        self.saturation_limit = saturation_limit

    def forward(self, V, edge_index, edge_weight=None):
        """
        V: [B, N, C] 或 [N, C]
        """
        if V.dim() == 3:
            B, N, C = V.shape
            # [B, N, C] -> [N, B, C] -> [N, B*C]
            V_flat = V.transpose(0, 1).reshape(N, -1)

            # 此时 x 变成 2D 矩阵，MessagePassing 内部的 spmm 路径将正常工作
            out_flat = self.propagate(edge_index, x=V_flat, edge_weight=edge_weight)

            # [N, B*C] -> [N, B, C] -> [B, N, C]
            return out_flat.reshape(N, B, C).transpose(0, 1)

        return self.propagate(edge_index, x=V, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        # 1. 计算电位差 (此时 x 可能是 [E, C] 或 [E, B*C])
        delta_v = x_j - x_i

        # 2. 物理饱和机制
        I_ij = torch.tanh(delta_v / self.saturation_limit) * self.saturation_limit

        # 3. 结合突触电导
        res = self.g * I_ij

        if edge_weight is not None:
            # 修改点：使用 unsqueeze(-1) 自动适配 2D 或 3D 的特征广播
            # 不要用 view(-1, 1)，因为它假设 res 只有两维
            res = res * edge_weight.unsqueeze(-1)

        return res

    def update(self, aggr_out):
        return aggr_out


# class SynapticMessagePassing(MessagePassing):
#     """
#     基于 m(Vi, Vj) 的实现：计算 I_gap = Σ g_ij * σ(Vj - Vi)
#     物理含义：包含非线性/饱和机制的突触耦合
#     """

#     def __init__(self, saturation_limit=2.0):
#         super().__init__(aggr='add')  # 电流叠加

#         self.g = torch.nn.Parameter(torch.Tensor([1.0]))

#         self.saturation_limit = saturation_limit  # 物理饱和阈值

#     def forward(self, V, edge_index, edge_weight=None):
#         # edge_weight 可以代表突触的静态权重 (如解剖学连接)
#         return self.propagate(edge_index, x=V, edge_weight=edge_weight)

#     def message(self, x_i, x_j, edge_weight):
#         # 1. 计算电位差
#         delta_v = x_j - x_i

#         # 2. 引入物理约束：饱和机制 (Saturation)
#         I_ij = torch.tanh(delta_v / self.saturation_limit) * self.saturation_limit

#         # 3. 结合突触电导
#         res = self.g * I_ij

#         if edge_weight is not None:
#             res = res * edge_weight.view(-1, 1)
#         return res

#     def update(self, aggr_out):

#         return aggr_out

class warmSynapticMessagePassing(MessagePassing):
    """
    改造后的异质化突触耦合模块
    I_gap = g_i * Σ (edge_weight_ij * σ(Vj - Vi))
    """

    def __init__(self, num_nodes: int, saturation_limit=2.0):
        super().__init__(aggr='add')  # 电流在目标节点叠加

        # 1. 异质化电导参数：每个神经元拥有独立的接收增益， 初始化为 1.0，代表尊重原始 edge_weight 的量级
        self.g = nn.Parameter(torch.ones(num_nodes, 1))

        self.saturation_limit = saturation_limit

    def forward(self, V, edge_index, edge_weight=None):
        # V: [N, C]
        # edge_index: [2, E]
        # edge_weight: [E, 1] 来自线虫连接组数据 (如突触数量)
        return self.propagate(edge_index, x=V, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        """
        x_i: 目标节点状态 (接收者)
        x_j: 源节点状态 (发送者)
        """
        # 计算两端电位差
        delta_v = x_j - x_i

        # 引入物理饱和机制 (防止长程演化中电荷无限累积导致梯度爆炸)
        I_unit = torch.tanh(delta_v / self.saturation_limit) * self.saturation_limit

        # 结合静态解剖权重 (Static Physical Prior)
        if edge_weight is not None:
            # 确保 edge_weight 维度对齐 [E, 1]
            I_unit = I_unit * edge_weight.view(-1, 1)

        return I_unit

    def update(self, aggr_out):
        """
        aggr_out: 聚合后的总输入电流 [N, C]
        在此阶段乘以节点特定的自适应电导 g_i
        """
        # 这里的 self.g 会根据 299 个神经元各自的 Loss 进行独立更新
        return self.g * aggr_out

class SynapticMatrixCoupling(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        # 定义一个 buffer 存储 CSR 矩阵
        self.register_buffer('adj_csr', None)

    def forward(self, V, edge_index, edge_weight):
        # 如果是第一次运行，或者 edge_index 发生了改变（通常不会）
        if self.adj_csr is None:
            N = V.size(-2) # 自动识别节点维度
            # 1. 先转 COO
            adj_coo = torch.sparse_coo_tensor(edge_index, edge_weight, size=(N, N))
            # 2. 预先转成 CSR 格式并存入 buffer
            self.adj_csr = adj_coo.to_sparse_csr()

        # 后续直接使用 self.adj_csr，不再有转换警告
        # 注意：这里仍需配合你之前的 V_flat (reshape) 逻辑
        B, N, C = V.shape
        V_flat = V.transpose(0, 1).reshape(N, -1)
        adj_v_flat = torch.sparse.mm(self.adj_csr, V_flat)

        return self.g * (adj_v_flat.reshape(N, B, C).transpose(0, 1) - V)

# class SynapticMatrixCoupling(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.g = torch.nn.Parameter(torch.ones(1))

#     # 在 Layer.py 的 forward 中：
#     def forward(self, V, edge_index, edge_weight=None):
#         N = V.size(0)

#         # 1. 将普通的 edge_index/weight 转换为标准的 PyTorch 稀疏张量
#         # 注意：size=(N, N) 非常重要，确保矩阵维度正确
#         adj_sparse = to_torch_sparse_tensor(edge_index, edge_weight, size=(N, N))

#         # 2. 调用 spmm，现在的 src 是一个稀疏张量对象
#         adj_v = spmm(adj_sparse, V, reduce='sum')

#         # 3. 计算耦合电流 I_gap = g * (A*V - V)
#         return self.g * (adj_v - V)