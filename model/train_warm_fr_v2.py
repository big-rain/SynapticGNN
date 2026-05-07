
import sys
import os

# 获取项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import random

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model.Cablelayer import SynapticGNNLayer
from dataloader.WormLoader  import WormDynamicDatasetLoader
import torch.nn.functional as F
from model.train_warm_fr_v1 import GPUMultiHorizonEvaluator

from scipy.stats import pearsonr
import pandas as pd




def worm_dynamic_loss(y_pred, y_true, y_mask, delta=1.0, tv_weight=0.05):

    mask = y_mask.float()
    num_valid = mask.sum() + 1e-8

    # Masked Huber Loss
    huber_loss = F.huber_loss(y_pred * mask, y_true * mask, reduction='sum', delta=delta)
    huber_loss = huber_loss / num_valid

    # Temporal TV Loss (约束变化率)
    if y_pred.shape[1] > 1:
        diff_pred = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        diff_true = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        diff_mask = mask[:, 1:, :, :]  # [B, T-1, N, 1]
        tv_loss = torch.sum(torch.abs(diff_pred - diff_true) * diff_mask)
        tv_loss = tv_loss / (diff_mask.sum() + 1e-8)
    else:
        tv_loss = 0.0

    return huber_loss + tv_weight * tv_loss



class WormSynapticSTGNN_v2(nn.Module):
    def __init__(self, channel_list, num_nodes, dataset_name, horizon=50):
        super().__init__()
        self.num_nodes = num_nodes
        self.channel_list = channel_list
        self.num_layers = len(channel_list)
        self.total_dim = sum(channel_list)

        # 1. 信号编码 (1维 V -> Embedding)
        self.signal_encoder = nn.Sequential(
            nn.Linear(1, channel_list[0]),
            nn.GELU(),
            nn.LayerNorm(channel_list[0])
        )

        # 2. 节点身份
        self.node_emb = nn.Parameter(torch.empty(num_nodes, 32))
        nn.init.xavier_uniform_(self.node_emb)
        self.h0_proj = nn.Linear(32, channel_list[0])

        # 3. 动力学层
        from model.Layer import get_dynamics_config
        dyn_config = get_dynamics_config(dataset_name)

        self.layers = nn.ModuleList([
            SynapticGNNLayer(channel_list[i], config=dyn_config)
            for i in range(self.num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(c) for c in channel_list])

        # 4. 学习步长 (代替固定的 beta)
        # 初始化为较小值，强制模型从小增量开始学
        self.beta = nn.Parameter(torch.tensor([-2.5]))

        # 5. 增量预测器 (去掉了 Tanh)
        self.feature_refiner = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim),
            nn.LayerNorm(self.total_dim),
            nn.GELU()
        )
        self.step_predictor = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim // 2),
            nn.GELU(),
            nn.Linear(self.total_dim // 2, 1) # 输出 V 的增量 ΔV
        )

    def cell_step(self, current_v, h_prev_list, edge_index, edge_weight):
        """
        输入 current_v: [B, N, 1]
        """
        batch_size = current_v.shape[0]
        feat = self.signal_encoder(current_v)

        new_h_list = []
        layer_outputs = []

        for i, layer in enumerate(self.layers):
            norm_feat = self.layer_norms[i](feat)

            if h_prev_list[i] is None:
                h0 = self.h0_proj(self.node_emb)
                V_h = h0.unsqueeze(0).expand(batch_size, -1, -1)
                W_h = torch.zeros_like(V_h) # 初始 W 设为 0，由网络后续演化
            else:
                V_h, W_h = torch.split(h_prev_list[i], self.channel_list[i], dim=-1)

            # 核心：即使没有外部 W，层内部依然维持 (V_h, W_h) 的动力学
            state_next = layer((V_h, W_h), edge_index, edge_weight, I_ext=norm_feat)
            V_n, _ = torch.split(state_next, self.channel_list[i], dim=-1)

            layer_outputs.append(V_n)
            new_h_list.append(state_next)

            # 残差更新特征，传递给下一层
            feat = V_n + 0.1 * feat

            # 融合多层物理特征
        fused = self.feature_refiner(torch.cat(layer_outputs, dim=-1))
        delta_v = self.step_predictor(fused)

        # 欧拉步进
        step_size = torch.sigmoid(self.beta)
        next_v = current_v + step_size * delta_v

        return next_v, new_h_list

    def forward(self, x_seq, edge_index, edge_weight, future_y=None, use_tf=False):
        """
        x_seq: [B, 20, N, 1]
        """
        h_list = [None] * self.num_layers

        # 1. Warm-up (对齐相位)
        for t in range(x_seq.shape[1]):
            _, h_list = self.cell_step(x_seq[:, t, :, :], h_list, edge_index, edge_weight)

        # 2. Rollout (自回归)
        preds = []
        current_v = x_seq[:, -1, :, :]
        horizon = future_y.shape[1] if future_y is not None else 50

        for t in range(horizon):
            next_v, h_list = self.cell_step(current_v, h_list, edge_index, edge_weight)
            preds.append(next_v.unsqueeze(1))

            if self.training and use_tf and future_y is not None:
                current_v = future_y[:, t, :, :]
            else:
                current_v = next_v

        return torch.cat(preds, dim=1)


def run_worm_training(device, data_name='Venkatachalam2024'):
    # --- 1. 数据准备 (保持不变，但需注意 unsqueeze 逻辑) ---

    loader = WormDynamicDatasetLoader(
        raw_data_dir=r'D:\PycharmProjects/D1/notebook/gnn_ready_data/',
        adj_path=r'D:\PycharmProjects/D1/notebook/gnn_ready_data/adj_matrix_gap.pt'
    )

    # 获得 5 元组 (注意这里多了一个 y_mask)
    (train_loader, test_loaders_dict,
     edge_index, edge_weight) = loader.get_index_dataset(
        data_name=data_name,
        input_len=30,
        pred_len=90,  # 统一训练长程
        batch_size=32,
        stride=5
    )


    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    num_nodes = edge_index.max().item() + 1

    # --- 2. 模型实例化 (使用改造后的 v2 版本) ---
    # 确保 channel_list 足够容纳隐变量动态
    model = WormSynapticSTGNN_v2(
        channel_list=[32, 32, 32],
        num_nodes=num_nodes,
        dataset_name=data_name,
        horizon=90
    ).to(device)

    cheackpoint = torch.load(f'D:/PycharmProjects/D1/result/{data_name}/Euler_unified_{data_name}_model_3x32_0.pth')
    model.load_state_dict(cheackpoint['model_state_dict'], strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)

    # 评估器保持不变
    save_path = f'D:/PycharmProjects/D1/result/{data_name}/Euler_unified_{data_name}_model_3x32_1.pth'
    Free_run_metrics = GPUMultiHorizonEvaluator(horizons=[30, 60, 90], num_nodes=num_nodes, device=device)
    best_mae = 10.0

    # --- 3. 训练循环 ---
    for epoch in range(100):
        model.train()

        # 课程学习配置 (可以稍微激进一点，因为 v2 稳定性更好)
        if epoch < 5:
            rollout_len, rollout_w = 5, 0.2
        elif epoch < 20:
            rollout_len, rollout_w = 20, 0.4
        elif epoch < 35:
            rollout_len, rollout_w = 40, 0.5
        elif epoch < 55:
            rollout_len, rollout_w = 60, 0.5
        elif epoch < 70:
            rollout_len, rollout_w = 70, 0.7
        else:
            rollout_len, rollout_w = 90, 1.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [RO_Len: {rollout_len}]")

        for x, y, y_mask in pbar:
            # 数据准备：确保是 [B, T, N, 1]
            x = x.to(device).unsqueeze(-1)
            y = y.to(device).unsqueeze(-1)
            y_mask = y_mask.to(device).unsqueeze(-1)

            optimizer.zero_grad()

            # --- 第一步：Teacher Forcing (TF) ---
            preds_tf = model(x, edge_index, edge_weight, future_y=y, use_tf=True)
            tf_loss = worm_dynamic_loss(preds_tf, y, y_mask)

            # --- 第二步：Free-Run (FR) 闭环演化 ---
            preds_fr = model(x.detach(), edge_index, edge_weight,
                             future_y=y[:, :rollout_len], use_tf=False)

            fr_loss = worm_dynamic_loss(preds_fr, y[:, :rollout_len], y_mask[:, :rollout_len])
            norm_fr_loss = fr_loss / (rollout_len / 10.0)

            # --- 联合 Loss ---
            total_loss = tf_loss + rollout_w * norm_fr_loss

            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            if torch.isnan(total_loss):
                optimizer.zero_grad()
                continue

            optimizer.step()
            pbar.set_postfix({"tf": f"{tf_loss.item():.4f}", "fr": f"{fr_loss.item():.4f}", "grad": f"{grad_norm:.2f}"})


        scheduler.step()

        # --- 4. 验证阶段 (纯闭环 Free-Run) ---
        model.eval()
        Free_run_metrics.reset()
        with torch.no_grad():
            for worm_name, test_loader in test_loaders_dict.items():
                for x_t, y_t, y_m in test_loader:
                    x_t = x_t.to(device).unsqueeze(-1)
                    y_t = y_t.to(device).unsqueeze(-1)
                    y_m = y_m.to(device).unsqueeze(-1)

                    # 验证时统一使用闭环预测模式
                    preds = model(x_t, edge_index, edge_weight, future_y=y_t, use_tf=False)
                    Free_run_metrics.update(preds, y_t, y_m)

        res_FR = Free_run_metrics.summarize(model_name=f"V2_Euler_FR")
        current_t90_mae = res_FR[res_FR['Horizon'] == 'T+90']['MAE'].values[0]

        if current_t90_mae < best_mae:
            best_mae = current_t90_mae
            # 建议保存模型状态字典以及相关 meta 信息
            save_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_fr': res_FR,

            }
            torch.save(save_data, save_path)
            print(f"⭐ New Best T+90 MAE: {best_mae:.4f} Saved!")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # run_worm_training(device, data_name='Venkatachalam2024')
    run_worm_training(device, data_name='Yemini2021')