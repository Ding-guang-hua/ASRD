from statistics import mean
import torch
from torch import nn
import torch.nn.functional as F
from params import args
from sklearn.metrics import roc_auc_score
import numpy as np
import math
from Utils.Utils import cal_infonce_loss
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
from encoder import HiESEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HGDM(nn.Module):
    def __init__(self, f_dim, f_laps_static,nbclasses):
        super(HGDM, self).__init__()
        out_dims = eval(args.dims) + [args.latdim]
        in_dims = out_dims[::-1]
        self.user_denoise_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm)
        self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps)

        self.f_laps_static = f_laps_static
        self.struct_encoder = HiESEncoder(dim=args.con_dim, feat_name='lap_pos_enc')

        self.act = nn.LeakyReLU(0.5, inplace=True)
        self.helayers1 = nn.ModuleList()
        self.helayers2 = nn.ModuleList()
        self.main_layers = nn.ModuleList()
        self.weight = False
        for i in range(0, args.gcn_layer):
            self.helayers1.append(
                UUGCNLayer(args.latdim, args.latdim, weight=self.weight, bias=False, activation=self.act))
        for i in range(0, args.gcn_layer):
            self.helayers2.append(
                UUGCNLayer(args.latdim, args.latdim, weight=self.weight, bias=False, activation=self.act))
        for i in range(0, args.gcn_layer):
            self.main_layers.append(
                UUGCNLayer(args.latdim, args.latdim, weight=self.weight, bias=False, activation=self.act))

        self.transform_layer = torch.nn.Linear(f_dim, args.latdim, bias=True)
        nn.init.xavier_normal_(self.transform_layer.weight, gain=1.414)
        self.dense = torch.nn.Linear(args.latdim, nbclasses)
        self.pool = 'sum'
        self.a1 = torch.nn.Parameter(torch.tensor(0.5))
        self.a2 = torch.nn.Parameter(torch.tensor(0.5))

    def build_struct_feats(self, he_adjs):
        f_laps = {}
        for i in range(len(he_adjs)):
            stats, spectral = self.f_laps_static[i]
            f_laps[i] = self.struct_encoder(stats, spectral, he_adjs[i])
        return f_laps

    def forward(self, he_adjs, feature_list, is_training=True):
        embed = self.transform_layer(feature_list)
        target_embedding = [embed]
        source_embeddings1 = [embed]
        source_embeddings2 = [embed]

        for i, layer in enumerate(self.main_layers):
            embeddings = layer(he_adjs[0], target_embedding[-1])
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            target_embedding += [norm_embeddings]

        target_embedding = sum(target_embedding)

        for i, layer in enumerate(self.helayers1):
            embeddings = layer(he_adjs[1], source_embeddings1[-1])
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            source_embeddings1 += [norm_embeddings]

        source_embeddings1 = sum(source_embeddings1)

        for i, layer in enumerate(self.helayers2):
            embeddings = layer(he_adjs[2], source_embeddings2[-1])
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            source_embeddings2 += [norm_embeddings]

        source_embeddings2 = sum(source_embeddings2)

        return source_embeddings1, source_embeddings2, target_embedding

    def cal_loss(self, ancs, label, he_adjs, he_adjs_2, initial_feature):
        source_embeddings1, source_embeddings2, target_embedding = self.forward(he_adjs, initial_feature)  # GCN
        f_laps = self.build_struct_feats(he_adjs)

        diff_loss1, diff_embeddings1 = self.diffusion_model.training_losses2(self.user_denoise_model,
                                                                             source_embeddings1,
                                                                             source_embeddings2, ancs, [he_adjs_2[2]],
                                                                             f_laps[1] - f_laps[2])
        nextEmbed1 = self.a1 * (diff_embeddings1 + source_embeddings2) +(1 - self.a1) * source_embeddings1
        cl_loss1 = self.inter_step_triplet_loss(nextEmbed1, source_embeddings1, source_embeddings2, args.margin)
        diff_loss2, diff_embeddings2 = self.diffusion_model.training_losses2(self.user_denoise_model, target_embedding,
                                                                             self.a1 * (
                                                                                         diff_embeddings1 + source_embeddings2) +
                                                                             (1 - self.a1) * source_embeddings1, ancs,
                                                                             [he_adjs_2[2],he_adjs_2[1]], f_laps[0] - f_laps[1])
        diff_loss1 = diff_loss1.mean()
        diff_loss2 = diff_loss2.mean()
        diff_loss = diff_loss1 + diff_loss2
        all_embeddings = self.a2 * (diff_embeddings2 + source_embeddings1) + (1 - self.a2) * target_embedding
        cl_loss2 = self.inter_step_triplet_loss(all_embeddings, target_embedding, source_embeddings1, args.margin)
        cl_loss = cl_loss1 + cl_loss2

        scores = self.dense(all_embeddings)
        scores = F.log_softmax(scores, dim=1)

        batch_u = scores[ancs]
        batch_label = torch.argmax(label[ancs], dim=-1)
        nll_loss = F.nll_loss(batch_u, batch_label)
        return nll_loss, diff_loss, cl_loss

    def inter_step_triplet_loss(self, refined_emb, target_emb, source_emb, margin):
        """
        计算跨步三元组对比损失
        Args:
            refined_emb: 精炼后的表示 hat{e}_v^{r_{i+1}}, [N, d]
            target_emb: 目标关系的初始表示 e_v^{r_{i+1}}, [N, d]
            source_emb: 源关系的初始表示 e_v^{r_i}, [N, d]
            margin: 边界距离 m > 0
        Returns:
            loss: 跨步三元组损失的平均值
        """
        # L2 距离
        dist_pos = torch.norm(refined_emb - target_emb, p=2, dim=1)  # refined ↔ target
        dist_neg = torch.norm(refined_emb - source_emb, p=2, dim=1)  # refined ↔ source

        # 三元组损失
        loss = F.relu(dist_pos - dist_neg + margin)

        return loss.mean()


    def get_allembeds(self, he_adjs, he_adjs_2, initial_feature):  # 对应要修改
        source_embeddings1, source_embeddings2, target_embedding = self.forward(he_adjs, initial_feature)
        f_laps = self.build_struct_feats(he_adjs)
        diff_embeddings1 = self.diffusion_model.p_sample(self.user_denoise_model, source_embeddings2,
                                                         args.sampling_steps, [he_adjs_2[2]], f_laps[1] - f_laps[2])
        diff_embeddings2 = self.diffusion_model.p_sample(self.user_denoise_model,
                                                         self.a1 * (diff_embeddings1 + source_embeddings2) + (
                                                                     1 - self.a1) * source_embeddings1,
                                                         args.sampling_steps,
                                                         [he_adjs_2[2], he_adjs_2[1]], f_laps[0] - f_laps[1])
        all_embeddings = self.a2 * (diff_embeddings2 + source_embeddings1) + (1 - self.a2) * target_embedding

        scores = self.dense(all_embeddings)
        return all_embeddings, scores




class UUGCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 bias=False,
                 activation=None):
        super(UUGCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            init(self.u_w)
        self._activation = activation

    def forward(self, graph, u_f):
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
            node_f = u_f
            # D^-1/2
            # degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        in_dims_temp = [self.in_dims[0] + self.time_emb_dim + args.con_dim] + self.in_dims[1:]

        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.glu_W = nn.Linear(args.con_dim, args.con_dim)

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, f_lap, mess_dropout=True):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32) / (
                    self.time_emb_dim // 2)).to(device)
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)

        v_c = F.sigmoid(self.glu_W(f_lap))
        h = torch.cat([x, emb, v_c], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.W = nn.Linear(args.latdim, args.latdim)
        self.V = nn.Linear(args.latdim, args.latdim)

        # 关系感知噪声模块：根据不同关系（行为）生成图结构感知的噪声
        self.rel_noise = RelationAwareNoise(
            dim=args.latdim,
            num_relations_max=4,  # ijcai/tmall最多4种行为
            time_emb_dim=args.d_emb_size
        ).to(device)
        # 自适应噪声调度器：动态学习每一步的噪声强度 β_t
        self.adaptive_scheduler = AdaptiveNoiseScheduler(
            dim=args.latdim,
            time_emb_dim=args.d_emb_size,
            steps=self.steps,
            beta_max=0.999
        ).to(device)

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(device)
            if beta_fixed:
                self.betas[0] = 0.0001

            self.calculate_for_diffusion()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))
        return np.array(betas)

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]]).to(device)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(device)]).to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                    (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps, rel_adj_list, f_lap):# 后向去噪采样：从噪声逐步恢复原始数据
        if steps == 0:
            x_t = x_start
        else:
            t = torch.full((x_start.shape[0],), steps - 1, device=device, dtype=torch.long)
            eps = self.anisotropy_nosie(x_start, rel_adj_list, t)
            x_t = self.q_sample_adaptive(x_start, t, eps, graph_state=x_start)

        # for i in reversed(range(self.steps)):
        for i in reversed(range(steps)):
            t = torch.full((x_t.shape[0],), i, device=device, dtype=torch.long)
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t, f_lap)
            x_t = model_mean

        return x_t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t,
                                         x_start.shape) * x_start + self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def q_sample_adaptive(self, x_start, t, noise, graph_state=None):# 自适应前向扩散：使用学习到的 α_bar 进行加噪
        base_x = x_start if graph_state is None else graph_state
        alpha_bar_t = self.adaptive_scheduler.alpha_bar_t(base_x, t)  # [N,1]
        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1.0 - alpha_bar_t + 1e-8) * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def p_mean_variance(self, model, x, t, f_lap):
        model_output = model(x, t, f_lap, False)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t,
                                                x.shape) * model_output + self._extract_into_tensor(
            self.posterior_mean_coef2, t, x.shape) * x)

        return model_mean, model_log_variance

    def anisotropy_nosie(self, x_start, rel_adj_list, timesteps):# 生成【关系感知各向异性噪声】（调用专门的噪声模型）
        return self.rel_noise(x_start, rel_adj_list, timesteps)

    def training_losses2(self, model, targetEmbeds, x_start, batch, rel_adj_list, f_lap):
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,), device=device).long()

        ani_nosie = self.anisotropy_nosie(x_start, rel_adj_list, ts)

        if self.noise_scale != 0:
            x_t = self.q_sample_adaptive(x_start, ts, ani_nosie, graph_state=x_start)
        else:
            x_t = x_start

        model_output = model(x_t, ts, f_lap)

        mse = self.mean_flat((targetEmbeds - x_start - model_output) ** 2)

        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), torch.ones_like(weight), weight)

        diff_loss = (weight * mse)[batch]
        return diff_loss, model_output

    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

class RelationAwareNoise(nn.Module):
    def __init__(self, dim, num_relations_max, time_emb_dim):
        super().__init__()
        self.dim = dim
        self.num_relations_max = num_relations_max
        self.time_emb_dim = time_emb_dim

        # 每个关系一个可学习嵌入 e_r
        self.rel_emb = nn.Parameter(torch.randn(num_relations_max, dim) * 0.02)

        # 用当前节点表示、时间步、关系嵌入共同决定 alpha_r
        self.query_mlp = nn.Sequential(
            nn.Linear(dim + time_emb_dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )

        # 方向分解后的混合权重
        self.dir_mlp = nn.Sequential(
            nn.Linear(dim + time_emb_dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 2)
        )

    def time_embedding(self, timesteps, emb_dim, device):
        half = emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / max(half, 1)
        )
        args_ = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args_), torch.sin(args_)], dim=-1)
        if emb_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def relation_stats(self, x, adj2):
        # x: [N, d], adj2: [N, N] dense/sparse
        sum_neighbors = adj2 @ x
        # num_neighbors = adj2.sum(dim=1).clamp(min=1.0)
        row_sum = adj2.sum(dim=1)
        if hasattr(row_sum, "to_dense"):  # 稀疏求和结果有时仍是稀疏/特殊张量
            row_sum = row_sum.to_dense()
        num_neighbors = row_sum.reshape(-1, 1).clamp(min=1.0)
        mu = sum_neighbors / num_neighbors

        sum_sq = adj2 @ (x ** 2)
        mean_sq = sum_sq / num_neighbors
        var = (mean_sq - mu ** 2).clamp(min=0.0)
        sigma = torch.sqrt(var + 1e-8)
        return mu, sigma

    def forward(self, x, rel_adj_list, timesteps):
        device = x.device
        t_emb = self.time_embedding(timesteps, self.time_emb_dim, device)
        q = self.query_mlp(torch.cat([x, t_emb], dim=-1))  # [N, d]

        scores = []
        rel_stats_cache = []

        for rid, adj2 in enumerate(rel_adj_list):
            mu_r, sigma_r = self.relation_stats(x, adj2)
            rel_stats_cache.append((mu_r, sigma_r))

            # 不要 expand，直接广播
            score_r = (q * self.rel_emb[rid]).sum(dim=-1, keepdim=True)
            scores.append(score_r)

        alpha = torch.softmax(torch.cat(scores, dim=1), dim=1)  # [N, R]

        eps_bar = torch.zeros_like(x)
        for rid, (mu_r, sigma_r) in enumerate(rel_stats_cache):
            z_r = torch.randn_like(x)
            eps_r = mu_r + sigma_r * z_r
            eps_bar = eps_bar + alpha[:, rid:rid + 1] * eps_r

        mu_global = x.mean(dim=0, keepdim=True)
        center_dir = mu_global - x
        center_norm2 = (center_dir ** 2).sum(dim=1, keepdim=True).clamp_min(1e-8)

        eps_parallel = ((eps_bar * center_dir).sum(dim=1, keepdim=True) / center_norm2) * center_dir
        eps_perp = eps_bar - eps_parallel

        w = torch.softmax(self.dir_mlp(torch.cat([x, t_emb], dim=-1)), dim=-1)
        eps_final = w[:, 0:1] * eps_parallel + w[:, 1:2] * eps_perp
        return eps_final

class AdaptiveNoiseScheduler(nn.Module):
    def __init__(self, dim, time_emb_dim, steps, beta_max=0.999):
        super().__init__()
        self.dim = dim
        self.time_emb_dim = time_emb_dim
        self.steps = steps
        self.beta_max = beta_max

        self.mlp = nn.Sequential(
            nn.Linear(dim + time_emb_dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1)
        )

    def time_embedding(self, timesteps, emb_dim, device):
        half = emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / max(half, 1)
        )
        args_ = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args_), torch.sin(args_)], dim=-1)
        if emb_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def readout(self, x):
        return x.mean(dim=0, keepdim=True)  # [1, d]

    def build_alpha_bar_table(self, x):
        device = x.device
        s_t = self.readout(x)                                  # [1, d]
        s_t = s_t.expand(self.steps, -1)                       # [T, d]
        all_t = torch.arange(self.steps, device=device).long() # [T]
        t_emb = self.time_embedding(all_t, self.time_emb_dim, device)  # [T, time_dim]
        beta = torch.sigmoid(self.mlp(torch.cat([s_t, t_emb], dim=-1))) * self.beta_max  # [T,1]
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)                # [T,1]
        return alpha_bar

    def alpha_bar_t(self, x, timesteps):
        with torch.no_grad():
            alpha_bar_table = self.build_alpha_bar_table(x.detach())    # [T,1]
            return alpha_bar_table[timesteps]                           # [N,1]