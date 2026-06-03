import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import torch as t
from scipy.sparse.linalg import lobpcg
from params import args
import warnings
# 屏蔽lobpcg的所有相关警告（不止收敛警告，一网打尽）
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='scipy.sparse.linalg._lobpcg'
)
warnings.filterwarnings(
    'ignore',
    message=r'Exited (at iteration|postprocessing).*',
    category=UserWarning
)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class HiESEncoder(nn.Module):
    """
    分层高效结构编码器 Hi-ESEEncoder
    替代随机游走位置编码，输出节点结构编码
    """
    def __init__(self, dim, feat_name='lap_pos_enc'):
        super().__init__()
        self.hidden_dim = dim
        self.k_eigen = args.k_eigen  # 拉普拉斯前k个特征向量
        self.feat_name = feat_name

        # 1. 局部统计分支 MLP
        self.mlp_lsb = nn.Sequential(
            nn.Linear(5, dim),  # [deg, mean, std, max, min]
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )

        # 2. 轻量频谱分支 MLP
        self.mlp_lpb = nn.Sequential(
            nn.Linear(self.k_eigen, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )

        # 3. 可学习图分支 2层 GIN
        self.gin_layers = nn.ModuleList([
            nn.Linear(dim, dim),
            nn.Linear(dim, dim)
        ])
        self.eps = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(2)])

        # 4. 门控融合层
        self.gate = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Sigmoid()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )
        self.to(device)

    def compute_local_statistics(self, adj):
        """
        局部统计分支：度、邻居度的均值/方差/最大/最小
        """
        num_nodes = adj.shape[0]

        # 无向图的度（入度=出度）
        deg = np.array(adj.sum(axis=1)).flatten()
        mean = np.zeros(num_nodes)
        std = np.zeros(num_nodes)
        max_ = np.zeros(num_nodes)
        min_ = np.zeros(num_nodes)

        for v in range(num_nodes):
            neighbors = adj[v].indices  # 无向邻居（双向）
            if len(neighbors) == 0:
                mean[v] = 0
                std[v] = 0
                max_[v] = 0
                min_[v] = 0
            else:
                nd = deg[neighbors]
                mean[v] = np.mean(nd)
                std[v] = np.std(nd)
                max_[v] = np.max(nd)
                min_[v] = np.min(nd)

        # 拼接特征（仅5维：度、均值、方差、最大值、最小值）
        stats = np.stack([deg, mean, std, max_, min_], axis=1)
        return torch.tensor(stats, dtype=torch.float32)

    def compute_spectral_embedding(self, adj):
        """
        轻量频谱分支：随机游走拉普拉斯矩阵的前k个特征向量（Lanczos算法近似）
        """
        num_nodes = adj.shape[0]
        # 1. 计算出度对角矩阵 D_r
        deg = np.array(adj.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0  # 避免除0
        # 构建 D^-0.5 逆平方根对角矩阵（替换原来的 D^-1）
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg), format='csr')
        # 2. 计算随机游走拉普拉斯矩阵 L_rw = I - D_r^-1 * A
        I = sp.eye(num_nodes, format='csr')  # 单位矩阵
        L_sym = I - D_inv_sqrt.dot(adj).dot(D_inv_sqrt)

        X = np.random.rand(num_nodes, self.k_eigen)
        eigvals, eigvecs = lobpcg(
            L_sym,
            X,
            largest=False,
            maxiter=200,  # 迭代次数从默认20→100（增加收敛概率）
            tol=1e-1  # 精度要求从默认~0.002→0.01（放宽要求）
        )

        # 直接用 eigsh 求前 k 个最小特征值 + 已排序特征向量
        eigvals, eigvecs = eigsh(
            L_sym,
            k=self.k_eigen,
            which='SM',  # 求最小的 k 个特征值
            maxiter=200,
            tol=1e-3
        )

        # 4. 提取节点v的频谱特征 u_v = U_rw[v, :]
        # 特征向量归一化（避免幅值差异）
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0, keepdims=True)
        return torch.tensor(eigvecs, dtype=torch.float32)

    def forward(self, stats, spectral, g):
        """
        主入口：输入DGL图，输出并保存结构编码
        """
        # num_nodes = g.num_nodes()
        #
        # # 1. 构建邻接矩阵
        # edges = g.edges()
        # src = edges[0].cpu().numpy()
        # dst = edges[1].cpu().numpy()
        # adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
        #
        # # 2. 局部统计分支
        # stats = self.compute_local_statistics(adj).to(device)
        h_local = self.mlp_lsb(stats)
        #
        # # 3. 轻量频谱分支
        # eigenvecs = self.compute_spectral_embedding(adj).to(device)
        h_spectral = self.mlp_lpb(spectral)

        # 4. 可学习图分支（2层GIN）
        x = h_local + h_spectral
        # x = stats + spectral
        for l in range(2):
            # ========== 正确的邻居聚合 ==========
            # 1. 消息传递：每个节点收集邻居特征的和
            g.ndata['h'] = x
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'agg'))
            agg = g.ndata.pop('agg')  # 形状：[num_nodes, hidden_dim]，每个节点的邻居和

            # 2. 按 GIN 公式计算
            x = self.gin_layers[l]((1 + self.eps[l]) * x + agg)

            if l != 1:
                x = F.leaky_relu(x)
        h_learned = x

        # 5. 门控融合
        gate_input = torch.cat([h_local, h_spectral, h_learned], dim=1)
        gv = self.gate(gate_input)
        fused_part = self.fusion_mlp(torch.cat([h_spectral, h_learned], dim=1))
        c_r = gv * h_local + (1 - gv) * fused_part


        # 保存到节点特征
        g.ndata[self.feat_name] = c_r
        return c_r