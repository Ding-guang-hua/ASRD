import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
from calculate import compute_distinct, compute_sim
from params import args
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RL_Agent(nn.Module):
    def __init__(self, data):
        super(RL_Agent, self).__init__()
        self.data = data
        # entityEmbeddings = self.data.adj_matrix
        relEmbeddings = self.data.rel_embeddings
        self.rel_ids = self.data.rel_ids
        self.n_hid = args.latdim
        self.rel_dim = relEmbeddings.shape[1]
        self.rel_num = relEmbeddings.shape[0]

        # self.kg_embeds = nn.Linear(entityEmbeddings.shape[1], self.n_hid, bias=True)


        self.state_encoder = CosStateEncoder(self.rel_num, self.rel_dim, self.n_hid).to(device)
        self.policy_net = PolicyNetwork(self.n_hid, self.rel_num, hidden_dim=64).to(device)
        self.proxy_reward = ProxyRewardModel(input_dim=self.n_hid+2).to(device)

        self.gamma = 0.95  # 折扣因子（若多步奖励则使用，这里单步可设为1）

    def train_step(self, rel_embed, rel_id, relation_adjs, distinct_req):
        self.train()

        states, actions, rewards = [], [], []
        r_seq = [rel_id] # 记录已经选过的关系（防止重复）
        total_reward = 0.0# 总奖励

        for t in range(self.rel_num):

            state = self.state_encoder(r_seq, relation_adjs, distinct_req)
            action_probs, _ = self.policy_net(state)# 策略网络输出动作概率

            # 选择没有选过的关系
            available_actions = [r-1 for r in self.rel_ids if r not in r_seq]
            if not available_actions:
                break
            # 采样动作
            available_probs = action_probs[0, available_actions]
            available_probs = available_probs / (available_probs.sum() + 1e-8)
            action_idx = torch.multinomial(available_probs, 1).item()
            selected_rid = available_actions[action_idx]+1

            # 奖励计算
            distinct_r = distinct_req[selected_rid-1]
            sim_r_prev = compute_sim(relation_adjs[r_seq[-1]],relation_adjs[selected_rid])


            features = torch.tensor([[distinct_r, sim_r_prev]], dtype=torch.float32, device=device)
            reward_input = torch.cat([state, features], dim=1)
            reward = self.proxy_reward(reward_input)

            # 保存轨迹
            r_seq.append(selected_rid)
            states.append(state)
            actions.append(selected_rid)
            rewards.append(reward.item())
            total_reward += reward.item()
        # print(r_seq)
        # 策略梯度损失
        policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        discounted_reward = 0.0
        if len(rewards) > 0:
            for t in reversed(range(len(rewards))):
                discounted_reward = rewards[t] + self.gamma * discounted_reward
                action_probs, _ = self.policy_net(states[t])
                log_prob = torch.log(action_probs[0, actions[t]-1] + 1e-8)
                policy_loss = policy_loss - log_prob * discounted_reward
        # print(type(policy_loss))

        avg_reward = total_reward / len(rewards) if len(rewards) > 0 else 0.0
        return policy_loss, avg_reward


class ProxyRewardModel(nn.Module): #代理奖励模型（MLP）
    """
        代理奖励模型（MLP）
        输入：候选关系特征（区分度、相似度、信息增益） + 当前状态向量
        输出：0~1 之间的代理奖励分数
        """

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 加深一层，更稳定
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 最后不加Sigmoid，训练更稳定
        )
        # 输出用 Sigmoid 映射到 0~1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = 状态向量 + 候选关系特征
        score = self.mlp(x)
        reward = self.sigmoid(score)
        return reward.squeeze(-1)  # (batch,) → 奖励标量


class CosStateEncoder(nn.Module): # 状态编码
    # 将当前CoS片段的关系嵌入、关系区分度的池化、与上一个关系的相似性编码为状态向量。
    def __init__(self, rel_num, embed_dim, out_dim=16):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=rel_num, embedding_dim=embed_dim)# 关系嵌入层

        self.sim_proj = nn.Linear(1, embed_dim)  # 相似度：1维 → 映射到 hidden_dim

        self.fusion = nn.Linear(embed_dim + 1 + embed_dim, out_dim)

    def forward(self, rel_seq, rel_adj_dict, distinct_req):
        rel_seq_fixed = [x - 1 for x in rel_seq]
        rel_tensor = torch.tensor(rel_seq_fixed, device=device)
        rel_vectors = self.embedding_layer(rel_tensor)  # shape: [i, embed_dim]
        seq_embedding = torch.mean(rel_vectors, dim=0)  # shape: [embed_dim]

        distinct_scores = []
        for rid in rel_seq:
            dist_score = distinct_req[rid-1]
            distinct_scores.append(dist_score)
        distinct_tensor = torch.tensor(distinct_scores, device=device)
        # 对 Distinct 列表做池化 (这里使用均值，也可改为 max)
        distinct_pooled = torch.mean(distinct_tensor)  # 标量 [1]
        distinct_pooled = distinct_pooled.unsqueeze(0)

        if len(rel_seq) == 1:
            sim_val = torch.tensor([0.0], device=device)
        else:
            r_i_adj = rel_seq[-1]  # [embed_dim]
            r_prev_adj = rel_seq[-2]  # [embed_dim]
            sim_val = compute_sim(rel_adj_dict[r_i_adj], rel_adj_dict[r_prev_adj])  # [1]
            sim_val = torch.tensor([sim_val],device=device)
        sim_feature = self.sim_proj(sim_val)  # (batch_size, hidden_dim)

        fused = torch.cat([seq_embedding, distinct_pooled, sim_feature], dim=0)
        fused = fused.unsqueeze(0)
        state = self.fusion(fused)

        return state


class PolicyNetwork(nn.Module): # 策略网络
    """
        策略网络：
        输入：状态向量（来自 CosStateEncoder）
        输出：下一个关系的选择概率（36个关系 → 对应动作）
        """

    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
            # 注意：Softmax 放在 forward 外面或计算 loss 时更稳定
        )

    def forward(self, state):
        logits = self.fc(state)  # 原始输出分数
        probs = torch.softmax(logits, dim=-1)  # 概率分布
        return probs, logits