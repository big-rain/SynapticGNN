import torch
import torch.nn as nn

from torch_geometric.utils import spmm
from model.Layer import OptimalNodeDynamics, SynapticMessagePassing, SynapticMatrixCoupling


class SynapticGNNLayer(nn.Module):
    def __init__(self, channels: int, config: dict, coupling_type='message', saturation_limit=2.0):
        super().__init__()
        self.channels = channels
        # 从 config 中提取 dt，如果没有则默认为 0.1
        self.dt = config.get('dt', 0.1)

        # 1. 节点内部动力学模块 (负责时间演化: f(V, W))
        self.node_dynamics = OptimalNodeDynamics(channels, config=config)

        # 2. 节点间突触耦合模块 (负责空间信息传递: I_gap)
        if coupling_type == 'matrix':
            # 矩阵耦合：效率高，适合 PeMS07/08
            self.coupling = SynapticMatrixCoupling()
        elif coupling_type == 'message':
            # 消息传递耦合：支持非线性/饱和机制，更符合生物物理
            self.coupling = SynapticMessagePassing(saturation_limit=saturation_limit)
        else:
            raise ValueError(f"Unknown coupling type: {coupling_type}")

    def forward(self, x, edge_index, edge_weight=None, I_ext=0):
        """
        x: [N, 2 * channels] 或 (V, W) 元组
        I_ext: 外部驱动（如时间特征 Embedding 或原始流量映射）
        """
        # 状态拆分 [N, C], [N, C]
        if isinstance(x, tuple):
            V, W = x
        else:
            V, W = torch.split(x, self.channels, dim=-1)

        # --- 空间维度 (Spatial) ---
        # 计算所有邻居对节点的协同驱动电流 Σ g_ij * (Vj - Vi)
        I_gap = self.coupling(V, edge_index, edge_weight)

        # --- 时间维度 (Temporal) ---
        # 将空间电流 I_gap 作为 ODE 的输入项，计算下一时刻状态
        V_next, W_next = self.node_dynamics(V, W, I_ext=I_ext, I_gap=I_gap)

        # 状态合并输出，保持通道数一致
        out = torch.cat([V_next, W_next], dim=-1)
        return out

