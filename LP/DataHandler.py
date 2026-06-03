import pickle
import datetime
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
import dgl
from Utils.TimeLogger import log
import torch as t
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from torch_geometric.utils import from_dgl
from encoder import HiESEncoder

device = "cuda" if t.cuda.is_available() else "cpu"
class DataHandler:
    def __init__(self):
        
        if args.data == 'ijcai_15':
            self.predir = './data/ijcai_15/'
            self.behaviors = ['click', 'fav', 'cart', 'buy']
            self.beh_meta_path = ['buy', 'click_buy', 'click_fav_buy', 'click_fav_cart_buy']

        elif args.data  == 'tmall':
            self.predir = './data/tmall/'
            self.behaviors = ['pv', 'fav', 'cart', 'buy']
            self.beh_meta_path = ['buy', 'pv_buy', 'pv_fav_buy', 'pv_fav_cart_buy']
        elif args.data == 'retail_rocket':
            self.predir = './data/retail_rocket/'
            self.behaviors = ['view', 'cart', 'buy']
            self.beh_meta_path = ['buy', 'view_buy', 'view_cart_buy']

        self.train_file = self.predir + 'train_mat_'
        self.val_file = self.predir + 'test_mat.pkl'
        self.test_file = self.predir + 'test_mat.pkl'

    def _load_data(self):
        test_mode = 'other'
        self.t_max = -1
        self.t_min = 0x7FFFFFFF
        self.time_number = -1

        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {}
        self.behavior_mats_2 = {}
        self.behaviors_data = {}
        # self.f_laps = {} # 存储各行为的位置编码特征

        for i in range(0, len(self.behaviors)):# 遍历每一种用户行为
            with open(self.train_file + self.behaviors[i] + '.pkl', 'rb') as fs:
                data = pickle.load(fs)
                
                if self.behaviors[i] == 'buy':
                    self.train_mat = data # 把购买行为矩阵作为主训练矩阵
                    self.trainLabel = 1 * (self.train_mat != 0) # 转为0-1矩阵（有交互=1，无交互=0）
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))# 按列求和，得到每个物品被交互的总次数（用于权重）
                    continue
                self.behaviors_data[i] = 1*(data != 0)
                if data.get_shape()[0] > self.user_num:
                    self.user_num = data.get_shape()[0]
                if data.get_shape()[1] > self.item_num:
                    self.item_num = data.get_shape()[1]
                if data.data.max() > self.t_max:
                    self.t_max = data.data.max()
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()
        self.test_mat = pickle.load(open(self.test_file, 'rb'))
        self.userNum = self.behaviors_data[0].shape[0]
        self.itemNum = self.behaviors_data[0].shape[1]
        self._data2mat() # 调用函数把行为数据转为标准矩阵格式
        if test_mode == 'muti':
            self.target_adj = self._dataTargetmat()
        elif test_mode == 'lightgcn':
            self.target_adj = self._make_bitorch_adj(self.train_mat)
        else: # 默认模式（图位置编码模式）

            # transform = RandomWalkPE(k=args.con_dim, feat_name='lap_pos_enc')
            # 自定义随机游走位置编码，维度为args.con_dim
            # transform = CustomRandomWalkPE(k=args.con_dim, feat_name='lap_pos_enc')
            self.struct_encoder = HiESEncoder(dim=args.con_dim, feat_name='lap_pos_enc')
            # transform = LapPE(k=args.con_dim, feat_name='lap_pos_enc')


            self.target_adj = self.makeBiAdj(self.train_mat,self.userNum,self.itemNum)

            # transform(self.target_adj)
            num_nodes = self.target_adj.num_nodes()
            edges = self.target_adj.edges()
            src = edges[0].cpu().numpy()
            dst = edges[1].cpu().numpy()
            adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
            stats_tar = self.struct_encoder.compute_local_statistics(adj).to(device)
            spectral_tar = self.struct_encoder.compute_spectral_embedding(adj).to(device)
            self.f_laps_static_target = (stats_tar, spectral_tar)
            self.target_adj = self.target_adj.to(device)
            # self.f_lap1 = self.target_adj.ndata['lap_pos_enc'] # 取出图节点的位置编码特征

            self.f_laps_static = {}  # 缓存静态特征
            for i in range(0, len(self.behaviors_data)):# 遍历所有行为，为每个行为构建图并计算位置编码

                self.behavior_mats[i] = self.makeBiAdj(self.behaviors_data[i],self.userNum,self.itemNum).to(device)
                tmp = self.behavior_mats[i].adj()# 获取邻接矩阵
                self.behavior_mats_2[i] = (tmp @ tmp).to(device)# 计算邻接矩阵的平方（二阶邻居）

                num_nodes = self.behavior_mats[i].num_nodes()
                edges = self.behavior_mats[i].edges()
                src = edges[0].cpu().numpy()
                dst = edges[1].cpu().numpy()
                adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
                # 局部统计
                stats = self.struct_encoder.compute_local_statistics(adj).to(device)
                # 谱特征
                spectral = self.struct_encoder.compute_spectral_embedding(adj).to(device)
                self.f_laps_static[i] = (stats, spectral)
                # transform(self.behavior_mats[i])
                # self.f_laps[i] = self.behavior_mats[i].ndata['lap_pos_enc'] # 保存位置编码
            # self.f_laps[i+1] = self.f_lap1# 把主图（购买）的编码也存入f_laps
            # DataHandler(1).py



        self.beh_degree_list = []
        # 遍历每个行为，计算用户交互次数（度）并放到GPU
        for i in range(len(self.behaviors_data)):
            self.beh_degree_list.append(torch.tensor(((self.behaviors_data[i] != 0) * 1).sum(axis=-1)).cuda())

    def _data2mat(self):# 把行为数据转为模型可用的矩阵格式
        time = datetime.datetime.now()
        print("Start building: ", time)
        for i in range(0, len(self.behaviors_data)):
            self.behaviors_data[i] = 1*(self.behaviors_data[i] != 0)
            self.behavior_mats[i] = self._get_use(self.behaviors_data[i])# 归一化并转为PyTorch稀疏张量
        time = datetime.datetime.now()
        print("End building: ", time)
    def _dataTargetmat(self):
         target_adj = 1*(self.train_mat!=0)
         target_adj = self._get_use(target_adj)
         return target_adj

    def _get_use(self, behaviors_data):# 对行为矩阵做归一化并转为PyTorch张量，返回A、AT、A_ori
        behavior_mats = {}
        behaviors_data = (behaviors_data != 0) * 1
        behavior_mats['A'] = self._matrix_to_tensor(self._normalize_adj(behaviors_data))# 归一化邻接矩阵并转为torch稀疏张量
        behavior_mats['AT'] = self._matrix_to_tensor(self._normalize_adj(behaviors_data.T))
        behavior_mats['A_ori'] = None
        return behavior_mats

    def _normalize_adj(self, adj):# 对称归一化邻接矩阵（GCN经典归一化方法）
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        rowsum_diag = sp.diags(np.power(rowsum+1e-8, -0.5).flatten())# 构造D^(-0.5)对角矩阵
        colsum = np.array(adj.sum(0))
        colsum_diag = sp.diags(np.power(colsum+1e-8, -0.5).flatten())
        return rowsum_diag*adj*colsum_diag# 返回 D^(-0.5) * A * D^(-0.5)

    def _matrix_to_tensor(self, cur_matrix):# 把scipy稀疏矩阵转为torch稀疏张量
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)
        shape = torch.Size(cur_matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()
    
    def makeBiAdj(self, mat,n_user,n_item):# 构建用户-物品二部图（DGL图）
        a = sp.csr_matrix((n_user, n_user))
        b = sp.csr_matrix((n_item, n_item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = mat.tocoo()
        edge_src,edge_dst = mat.nonzero()
        ui_graph = dgl.graph(data=(edge_src, edge_dst),
                            idtype=torch.int64,
                             num_nodes=mat.shape[0]
                             )

        return ui_graph

    def load_data(self):
        
        self._load_data()
        args.user_num, args.item_num = self.train_mat.shape

    
    
        test_data = AllRankTestData(self.test_mat, self.train_mat)
        self.test_dataloader = dataloader.DataLoader(test_data, batch_size=args.tstBat, shuffle=False, num_workers=0)
        train_dataset = PairwiseTrnData(self.trainLabel.tocoo())
        self.train_dataloader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)



    def _normalize_biadj(self, mat):
        """Laplacian normalization for mat in coo_matrix

        Args:
            mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

        Returns:
            scipy.sparse.coo_matrix: normalized adjacent matrix
        """
        # Add epsilon to avoid divide by zero
        degree = np.array(mat.sum(axis=-1)) + 1e-10
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _make_bitorch_adj(self, mat):
        """Transform uni-directional adjacent matrix in coo_matrix into bi-directional adjacent matrix in torch.sparse.FloatTensor

        Args:
            mat (coo_matrix): the uni-directional adjacent matrix

        Returns:
            torch.sparse.FloatTensor: the bi-directional matrix
        """
        if type(mat) != sp.csr_matrix:
            mat = mat.tocsr()
        a = csr_matrix((self.userNum,  self.userNum))
        b = csr_matrix((self.itemNum, self.itemNum))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
        mat = self._normalize_biadj(mat)

        # make torch tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()
        

class TrnData(data.Dataset):
    def __init__(self, coomat):
        coomat = coomat.tocoo()
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item_num)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        coomat = coomat.tocoo()
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.user_pos_lists = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])


class AllRankTestData(data.Dataset):# 全排序测试数据集类（推荐系统标准测试）
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0

        user_pos_lists = [list() for i in range(coomat.shape[0])]
        # user_pos_lists = set()
        test_users = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            user_pos_lists[row].append(col)
            test_users.add(row)
        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.csrmat[pck_user].toarray()
        pck_mask = np.reshape(pck_mask, [-1])
        return pck_user, pck_mask


class PairwiseTrnData(data.Dataset):# 成对训练数据集类（BPR、NCF等成对排序模型用）
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
	
	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item_num)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg
	
	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]
     

class DiffusionData(data.Dataset):# 扩散模型专用数据集（时间/序列扩散）
    def __init__(self,y_data):
        self.y_data = y_data
        self.x_data = np.arange(0,len(y_data))
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx],self.y_data[idx]
