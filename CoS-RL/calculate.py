import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os

def compute_distinct(adj_matrix): #关系区分度（Distinct）
    """
        关系区分度（Distinct）
        :param adj_matrix:  当前关系的二值邻接矩阵 [num_entity, num_entity]，值为0/1
        :return: 区分度标量（值越大，区分能力越强）
        """
    rows, cols = adj_matrix.nonzero()
    e1_list = torch.from_numpy(rows)
    e2_list = torch.from_numpy(cols)

    def get_rows(indices):
        return torch.from_numpy(adj_matrix[indices.numpy()].toarray()).float()

    emb1 = get_rows(e1_list)
    emb2 = get_rows(e2_list)

    # 3. 计算余弦相似度平均值
    cos_sims = torch.cosine_similarity(emb1, emb2, dim=1)
    avg_cos = torch.mean(cos_sims)

    distinct_score = 1.0 / (1.0 + avg_cos)
    return distinct_score.item() if isinstance(distinct_score, torch.Tensor) else distinct_score

def compute_distinct_drug(adj_matrix):  # 关系区分度（Distinct）
    """
        关系区分度（Distinct）
        :param adj_matrix:  当前关系的二值邻接矩阵 [num_entity, num_entity]，值为0/1
        :return: 区分度标量（值越大，区分能力越强）
        """
    # 获取所有边索引
    rows, cols = adj_matrix.nonzero()
    total = len(rows)

    # 分批大小（根据内存自动调整，2048 最稳）
    batch_size = 2048
    all_cos = []

    # 内部分批处理，完全不修改外部传参
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)

        # 当前批次的索引
        batch_e1 = rows[i:end]
        batch_e2 = cols[i:end]

        # 原版逻辑：提取行 → 转 tensor
        emb1 = torch.from_numpy(adj_matrix[batch_e1].toarray()).float()
        emb2 = torch.from_numpy(adj_matrix[batch_e2].toarray()).float()

        # 原版余弦计算
        cos_sims = torch.cosine_similarity(emb1, emb2, dim=1)
        all_cos.append(cos_sims)

    # 合并所有批次结果，和原版完全一样计算
    all_cos = torch.cat(all_cos)
    avg_cos = torch.mean(all_cos)
    distinct_score = 1.0 / (1.0 + avg_cos)

    return distinct_score.item()

def compute_sim(adj_i, adj_j):
    """
    关系间结构相似性（Sim）
    :param adj_i: scipy csr_matrix  关系i邻接矩阵
    :param adj_j: scipy csr_matrix  关系j邻接矩阵
    :return: Jaccard 相似度 0~1
    """
    # 只取 行号（有连接的实体）
    rows_i, _ = adj_i.nonzero()
    rows_j, _ = adj_j.nonzero()

    set_i = set(rows_i)
    set_j = set(rows_j)

    intersection = len(set_i & set_j)
    union = len(set_i | set_j)

    return intersection / union if union != 0 else 0.0

