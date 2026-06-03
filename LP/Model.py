import torch
from torch import nn
from params import args
import numpy as np
import math
from Utils.Utils import *
import dgl.function as fn
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from encoder import HiESEncoder
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
#Models

class HGDM(nn.Module):
    def __init__(self,data_handler):
        super(HGDM, self).__init__()
        self.n_user = data_handler.userNum
        self.n_item = data_handler.itemNum
        self.behavior_mats = data_handler.behavior_mats# 多行为图矩阵
        self.target_adj = data_handler.target_adj        # 目标行为（购买）图
        self.n_hid = args.latdim
        self.n_layers = args.gcn_layer
        self.embedding_dict = self.init_weight(self.n_user, self.n_item, self.n_hid)# 初始化用户/物品嵌入层
        self.act = nn.LeakyReLU(0.5, inplace=True)
        self.layers = nn.ModuleList()# 主GNN层
        
        self.hter_layers = nn.ModuleList()# 多行为GNN层（每种行为一个分支）
        self.weight = False
        for i in range(0, self.n_layers):# 添加主图GNN层
            self.layers.append(DGLLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act))
        for i in range(0,len(self.behavior_mats)):# 为每一种行为添加独立的GNN层
            single_layers = nn.ModuleList()
            for i in range(0, self.n_layers):
                single_layers.append(DGLLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act))
            self.hter_layers.append(single_layers)
        # 扩散过程（高斯扩散）
        self.diffusion_process = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).to(device)
        # 去噪网络 MLP 结构
        out_dims = eval(args.dims) + [args.latdim]
        in_dims = out_dims[::-1]
        self.denoiser = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)

        self.f_laps_static = data_handler.f_laps_static
        self.f_laps_static_target = data_handler.f_laps_static_target
        self.struct_encoder = HiESEncoder(dim=args.con_dim, feat_name='lap_pos_enc')


        # 最终激活函数
        self.final_act = nn.LeakyReLU(negative_slope=0.5)
        # 可学习的融合权重
        self.a1 = torch.nn.Parameter(torch.tensor(0.5))
        self.a2 = torch.nn.Parameter(torch.tensor(0.5))
        self.a3 = torch.nn.Parameter(torch.tensor(0.5))


    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
        })
        return embedding_dict

    def build_struct_feats(self):
        f_laps = {}
        for i in range(len(self.behavior_mats)):
            stats, spectral = self.f_laps_static[i]
            f_laps[i] = self.struct_encoder(stats, spectral, self.behavior_mats[i])
        stats, spectral = self.f_laps_static_target
        f_laps[len(self.behavior_mats)] = self.struct_encoder(stats, spectral, self.target_adj)
        return f_laps

    def forward(self):# 前向传播：GNN 编码
        init_embedding = torch.concat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
        init_heter_embedding = torch.concat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
        all_embeddings = [init_embedding]
        heter_embeddings = []
        # 主图GNN传播（目标行为：购买）
        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(self.target_adj, self.embedding_dict['user_emb'], self.embedding_dict['item_emb'])
            else:
                embeddings = layer(self.target_adj, embeddings[:self.n_user], embeddings[self.n_user:])

            norm_embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]
        ui_embeddings = sum(all_embeddings)# 求和得到最终主图嵌入

        for i in range(0, len(self.behavior_mats)):# 多行为图GNN传播（点击、加购、收藏等）
            sub_heter_embeddings = [init_heter_embedding]
            for j, layer in enumerate(self.layers):
                if j == 0:
                    embeddings = layer(self.behavior_mats[i], self.embedding_dict['user_emb'],
                                       self.embedding_dict['item_emb'])
                else:
                    embeddings = layer(self.behavior_mats[i], embeddings[:self.n_user], embeddings[self.n_user:])

                norm_embeddings = F.normalize(embeddings, p=2, dim=1)

                sub_heter_embeddings += [norm_embeddings]
            sub_heter_embeddings = sum(sub_heter_embeddings)# 求和得到最终单个行为图嵌入
            heter_embeddings.append(sub_heter_embeddings)

        return ui_embeddings, heter_embeddings

    def cal_loss(self,ancs, poss, negs, behavior_mats_2):# 计算总损失：BPR损失 + 正则损失 + 扩散损失
        tarEmbeds, he_Embeds = self.forward()
        f_laps = self.build_struct_feats()

        he_num = len(behavior_mats_2)
        if he_num == 3:# 根据行为数量执行分层扩散（3种行为：点击→加购→购买）

            diff_loss0, diff_Embeds0 = self.diffusion_process.training_losses2(
                self.denoiser, he_Embeds[1], he_Embeds[0], ancs,
                [behavior_mats_2[0]],
                f_laps[1] - f_laps[0]
            )
            nextEmbed1 = self.a1 * (he_Embeds[0] + diff_Embeds0) + (1 - self.a1) * he_Embeds[1]
            cl_loss1 = self.inter_step_triplet_loss(nextEmbed1, he_Embeds[1], he_Embeds[0], args.margin)
            diff_loss1, diff_Embeds1 = self.diffusion_process.training_losses2(
                self.denoiser, he_Embeds[2],
                self.a1 * (he_Embeds[0] + diff_Embeds0) + (1 - self.a1) * he_Embeds[1],
                ancs,
                [behavior_mats_2[0], behavior_mats_2[1]],
                f_laps[2] - f_laps[1]
            )
            nextEmbed2 = self.a2 * (he_Embeds[1] + diff_Embeds1) + (1 - self.a2) * he_Embeds[2]
            cl_loss2 = self.inter_step_triplet_loss(nextEmbed2, he_Embeds[2], he_Embeds[1], args.margin)
            diff_loss2, diff_Embeds2 = self.diffusion_process.training_losses2(
                self.denoiser, tarEmbeds,
                self.a2 * (he_Embeds[1] + diff_Embeds1) + (1 - self.a2) * he_Embeds[2],
                ancs,
                [behavior_mats_2[0], behavior_mats_2[1], behavior_mats_2[2]],
                f_laps[3] - f_laps[2]
            )
            # 最终嵌入融合
            Embeds_final = self.a3 * (he_Embeds[2] + diff_Embeds2) + (1 - self.a3) * tarEmbeds
            cl_loss3 = self.inter_step_triplet_loss(Embeds_final, tarEmbeds, he_Embeds[2], args.margin)
            cl_loss = cl_loss1 + cl_loss2 + cl_loss3
            # 总扩散损失
            # diff_loss = (diff_loss0.mean() + diff_loss0.mean() + diff_loss1.mean() + diff_loss1.mean() + diff_loss2.mean() + diff_loss2.mean())
            #0.0123   0.0048
            diff_loss = (diff_loss0.mean() + diff_loss1.mean() + diff_loss2.mean())
        elif he_num == 2:

            diff_loss0, diff_Embeds0 = self.diffusion_process.training_losses2(
                self.denoiser, he_Embeds[1], he_Embeds[0], ancs,
                [behavior_mats_2[0]],
                f_laps[1] - f_laps[0]
            )
            nextEmbed = self.a1 * (he_Embeds[0] + diff_Embeds0) + (1 - self.a1) * he_Embeds[1]
            cl_loss1 = self.inter_step_triplet_loss(nextEmbed, he_Embeds[1], he_Embeds[0], args.margin)

            diff_loss1, diff_Embeds1 = self.diffusion_process.training_losses2(
                self.denoiser, tarEmbeds,
                self.a1 * (he_Embeds[0] + diff_Embeds0) + (1 - self.a1) * he_Embeds[1],
                ancs,
                [behavior_mats_2[0], behavior_mats_2[1]],
                f_laps[2] - f_laps[1]
            )
            Embeds_final = self.a2 * (he_Embeds[1] + diff_Embeds1) + (1 - self.a2) * tarEmbeds
            cl_loss2 = self.inter_step_triplet_loss(Embeds_final, tarEmbeds, he_Embeds[1], args.margin)
            cl_loss = cl_loss1 + cl_loss2
            # diff_loss = (diff_loss0.mean() + diff_loss0.mean() + diff_loss1.mean() + diff_loss1.mean())
            diff_loss = (diff_loss0.mean() + diff_loss1.mean())
        else:
            print("Undefined")

        # 取出用户、正样本、负样本的嵌入
        ancEmbeds = Embeds_final[:self.n_user][ancs]
        posEmbeds = Embeds_final[self.n_user:][poss]
        negEmbeds = Embeds_final[self.n_user:][negs]
        # BPR 成对排序损失
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
        # L2 正则损失
        regLoss = ((torch.norm(ancEmbeds) ** 2 + torch.norm(posEmbeds) ** 2 + torch.norm(negEmbeds) ** 2) * args.reg)/args.batch
        loss = bprLoss + regLoss + diff_loss+ cl_loss * args.cl_weight
        return loss,bprLoss,regLoss,diff_loss

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

    def predict(self, behavior_mats_2):
        tarEmbeds, he_Embeds = self.forward()
        f_laps = self.build_struct_feats()
        be_num = len(he_Embeds)
        if be_num == 3:
            diff_Embeds0 = self.diffusion_process.p_sample(self.denoiser, he_Embeds[0], args.sampling_steps,
                                                              [behavior_mats_2[0]], f_laps[1] - f_laps[0])
            diff_Embeds1 = self.diffusion_process.p_sample(self.denoiser, self.a1*(he_Embeds[0]+diff_Embeds0)+(1-self.a1)*he_Embeds[1],
                                                              args.sampling_steps, [behavior_mats_2[0],behavior_mats_2[1]], f_laps[2] - f_laps[1])
            diff_Embeds2 = self.diffusion_process.p_sample(self.denoiser, self.a2*(he_Embeds[1]+diff_Embeds1)+(1-self.a2)*he_Embeds[2],
                                                              args.sampling_steps, [behavior_mats_2[0],behavior_mats_2[1],behavior_mats_2[2]], f_laps[3] - f_laps[2])

            Embeds_final = self.a3 * (he_Embeds[2] + diff_Embeds2) + (1 - self.a3) * tarEmbeds
        elif be_num == 2:
            diff_Embeds0 = self.diffusion_process.p_sample(self.denoiser, he_Embeds[0], args.sampling_steps,
                                                              [behavior_mats_2[0]], f_laps[1] - f_laps[0])
            diff_Embeds1 = self.diffusion_process.p_sample(self.denoiser, self.a1 * (he_Embeds[0] + diff_Embeds0) + (1 - self.a1) * he_Embeds[1],
                                                              args.sampling_steps, [behavior_mats_2[0],behavior_mats_2[1]],f_laps[2] - f_laps[1])
            Embeds_final = self.a2 * (he_Embeds[1] + diff_Embeds1) + (1 - self.a2) * tarEmbeds
        else:
            print('undefined_test')

        return Embeds_final[:self.n_user], Embeds_final[self.n_user:]

class DGLLayer(nn.Module):# DGL 图卷积层
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 bias=False,
                 activation=None):
        super(DGLLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            self.v_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            # self.e_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            xavier_uniform_(self.u_w)
            xavier_uniform_(self.v_w)
            # init.xavier_uniform_(self.e_w)
        self._activation = activation

    def forward(self, graph, u_f, v_f):# 图卷积前向传播
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
                v_f = torch.mm(v_f, self.v_w)
                # e_f = t.mm(e_f, self.e_w)
            node_f = torch.cat([u_f, v_f], dim=0)
            # 计算 D^-1/2 归一化
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            node_f = node_f * norm

            # DGL 消息传递：复制节点特征 → 聚合邻居
            graph.ndata['n_f'] = node_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']
            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
        
class Denoise(nn.Module):# 扩散模型的去噪网络 MLP
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm
        # 时间步嵌入层
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        # 输入层拼接：特征 + 时间嵌入 + 图位置编码
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim + args.con_dim] + self.in_dims[1:]
        out_dims_temp = self.out_dims
        # 编码器 MLP
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        # 解码器 MLP
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()
        # 位置编码门控
        self.glu_W = nn.Linear(args.con_dim, args.con_dim)

    def init_weights(self):# 初始化网络权重
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

    def forward(self, x, timesteps, f_lap, mess_dropout=True):# 前向：去噪预测
        # 时间步正弦嵌入
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(device)
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        # 归一化
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)

        # 图位置编码门控
        v_c = F.sigmoid(self.glu_W(f_lap))
        # 拼接：节点特征 + 时间嵌入 + 位置编码
        h = torch.cat([x, emb, v_c], dim=-1)
        # 编码层
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        # 解码层
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

        self.history_num_per_term = 10
        self.Lt_history = torch.zeros(steps, 10, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

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
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
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
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

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
            
    def q_sample(self, x_start, t, noise=None):# 前向扩散：加噪声
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def q_sample_adaptive(self, x_start, t, noise, graph_state=None):# 自适应前向扩散：使用学习到的 α_bar 进行加噪
        base_x = x_start if graph_state is None else graph_state
        alpha_bar_t = self.adaptive_scheduler.alpha_bar_t(base_x, t)  # [N,1]
        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1.0 - alpha_bar_t + 1e-8) * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):# 按时间步提取系数
        arr = arr.to(device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def p_mean_variance(self, model, x, t, f_lap):# 预测去噪后的均值和方差
        model_output = model(x, t, f_lap, False)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
        
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

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)
            assert pt_all.sum(-1) - 1. < 1e-5
            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            return t, pt
        elif method == 'uniform':  # uniform sampling
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()
            return t, pt
        else:
            raise ValueError

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