import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
import dgl
import torch as t
from sklearn.preprocessing import OneHotEncoder
from dgl import LapPE
from dgl.transforms import RandomWalkPE
from encoder import HiESEncoder

device = "cuda" if t.cuda.is_available() else "cpu"

class DataHandler:
    def __init__(self):
        if args.data == 'DBLP':
            predir = './data/DBLP/'
        if args.data == 'aminer':
            predir = './data/aminer/'
        self.predir = predir

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        user,item = mat.shape[0],mat.shape[1]
        a = sp.csr_matrix((user, user))
        b = sp.csr_matrix((item, item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(device)

    def makeTorchuAdj(self, mat):
        """Create tensor-based adjacency matrix for user social graph.

        Args:
            mat: Adjacency matrix.

        Returns:
            Tensor-based adjacency matrix.
        """
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(device)

    def makeBiAdj(self, mat):
        n_user = mat.shape[0]
        n_item = mat.shape[1]
        a = sp.csr_matrix((n_user, n_user))
        b = sp.csr_matrix((n_item, n_item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = mat.tocoo()
        edge_src,edge_dst = mat.nonzero()
        ui_graph = dgl.graph(data=(edge_src, edge_dst),
                            idtype=t.int32,
                             num_nodes=mat.shape[0]
                             )

        return ui_graph

    def LoadData(self):
        if args.data == 'DBLP':
            features_list,apa_mat,ata_mat,ava_mat,train,val,test,labels = self.load_dblp_data()
            self.feature_list = t.FloatTensor(features_list).to(device)

            self.struct_encoder = HiESEncoder(dim=args.con_dim, feat_name='lap_pos_enc')
            # apa->ava->ata  Ma 0.9318+0.0027 Mi 0.9365+0.0025 Ma 0.9278+0.0027 Mi 0.9304+0.0028 Ma 0.9377+0.0020 Mi 0.9435+0.0018
            self.hete_adj1 = dgl.from_scipy(ata_mat).to(device)
            tmp = self.hete_adj1.adj()
            self.hete_adj1_2 = (tmp @ tmp).to(device)

            num_nodes = self.hete_adj1.num_nodes()
            edges = self.hete_adj1.edges()
            src = edges[0].cpu().numpy()
            dst = edges[1].cpu().numpy()
            adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
            stats = self.struct_encoder.compute_local_statistics(adj).to(device)
            spectral = self.struct_encoder.compute_spectral_embedding(adj).to(device)
            self.f_laps_static1 = (stats, spectral)

            self.hete_adj2 = dgl.from_scipy(ava_mat).to(device)
            tmp = self.hete_adj2.adj()
            self.hete_adj2_2 = (tmp @ tmp).to(device)
            num_nodes = self.hete_adj2.num_nodes()
            edges = self.hete_adj2.edges()
            src = edges[0].cpu().numpy()
            dst = edges[1].cpu().numpy()
            adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
            stats = self.struct_encoder.compute_local_statistics(adj).to(device)
            spectral = self.struct_encoder.compute_spectral_embedding(adj).to(device)
            self.f_laps_static2 = (stats, spectral)
            self.hete_adj3 = dgl.from_scipy(apa_mat).to(device)
            tmp = self.hete_adj3.adj()
            self.hete_adj3_2 = (tmp @ tmp).to(device)
            num_nodes = self.hete_adj3.num_nodes()
            edges = self.hete_adj3.edges()
            src = edges[0].cpu().numpy()
            dst = edges[1].cpu().numpy()
            adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
            stats = self.struct_encoder.compute_local_statistics(adj).to(device)
            spectral = self.struct_encoder.compute_spectral_embedding(adj).to(device)
            self.f_laps_static3 = (stats, spectral)

            #transform = LapPE(k=args.con_dim, feat_name='lap_pos_enc')
            # transform = RandomWalkPE(k=args.con_dim, feat_name='lap_pos_enc')
            # transform(self.hete_adj1)
            # transform(self.hete_adj2)
            # transform(self.hete_adj3)
            # self.f_lap1 = self.hete_adj1.ndata['lap_pos_enc']
            # self.f_lap2 = self.hete_adj2.ndata['lap_pos_enc']
            # self.f_lap3 = self.hete_adj3.ndata['lap_pos_enc']


            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels


            self.he_adjs = [self.hete_adj1, self.hete_adj2, self.hete_adj3]
            self.he_adjs_2 = [self.hete_adj1_2, self.hete_adj2_2, self.hete_adj3_2]
            self.f_laps_static = [self.f_laps_static1, self.f_laps_static2, self.f_laps_static3]


        if args.data == 'aminer':
            features_list,pap_mat,prp_mat,pos_mat,train,val,test,labels = self.load_aminer_data()
            self.feature_list = t.FloatTensor(features_list).to(device)

            self.struct_encoder = HiESEncoder(dim=args.con_dim, feat_name='lap_pos_enc')
            # pos->prp->pap
            # pap->prp->pos  pap->prp->pos
            # pap->prp->pos  Ma  0.6639+0.0143  Mi 0.7555+0.0222  Ma 0.7146+0.0118  Mi 0.7902+0.0146 Ma 0.7207+0.0104 Mi 0.7870+0.0135
            #  验证  Ma 0.6667+0.0145 Mi  0.7493+0.0193Ma  0.7158+0.0116Mi 0.7931+0.0093 Ma0.7169+0.0122  Mi0.7914+0.0164
            # repeat=20   Ma 0.6731+0.0111Mi 0.7509+0.0153 Ma0.7169+0.0115  Mi 0.7933+0.0135 Ma  0.7223+0.0083Mi0.7934+0.0104
            #ma 0.6698+0.0139 mi  0.7442+0.0138  ma0.7166+0.0080  mi 0.7924+0.0095 ma 0.7202+0.0060 mi 0.7936+0.0114
            self.hete_adj1 = dgl.from_scipy(pos_mat).to(device)
            tmp = self.hete_adj1.adj()
            self.hete_adj1_2 = (tmp @ tmp).to(device)
            num_nodes = self.hete_adj1.num_nodes()
            edges = self.hete_adj1.edges()
            src = edges[0].cpu().numpy()
            dst = edges[1].cpu().numpy()
            adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
            stats = self.struct_encoder.compute_local_statistics(adj).to(device)
            spectral = self.struct_encoder.compute_spectral_embedding(adj).to(device)
            self.f_laps_static1 = (stats, spectral)

            self.hete_adj2 = dgl.from_scipy(prp_mat).to(device)
            tmp = self.hete_adj2.adj()
            self.hete_adj2_2 = (tmp @ tmp).to(device)
            num_nodes = self.hete_adj2.num_nodes()
            edges = self.hete_adj2.edges()
            src = edges[0].cpu().numpy()
            dst = edges[1].cpu().numpy()
            adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
            stats = self.struct_encoder.compute_local_statistics(adj).to(device)
            spectral = self.struct_encoder.compute_spectral_embedding(adj).to(device)
            self.f_laps_static2 = (stats, spectral)

            self.hete_adj3 = dgl.from_scipy(pap_mat).to(device)
            tmp = self.hete_adj3.adj()
            self.hete_adj3_2 = (tmp @ tmp).to(device)
            num_nodes = self.hete_adj3.num_nodes()
            edges = self.hete_adj3.edges()
            src = edges[0].cpu().numpy()
            dst = edges[1].cpu().numpy()
            adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
            stats = self.struct_encoder.compute_local_statistics(adj).to(device)
            spectral = self.struct_encoder.compute_spectral_embedding(adj).to(device)
            self.f_laps_static3 = (stats, spectral)

            # transform = RandomWalkPE(k=args.con_dim, feat_name='lap_pos_enc')
            # transform(self.hete_adj1)
            # transform(self.hete_adj2)
            # transform(self.hete_adj3)
            # self.f_lap1 = self.hete_adj1.ndata['lap_pos_enc']
            # self.f_lap2 = self.hete_adj2.ndata['lap_pos_enc']
            # self.f_lap3 = self.hete_adj3.ndata['lap_pos_enc']

            self.train_idx = train
            self.val_idx = val
            self.test_idx = test
            self.labels = labels

            self.he_adjs = [self.hete_adj1,self.hete_adj2,self.hete_adj3]
            self.he_adjs_2 = [self.hete_adj1_2, self.hete_adj2_2, self.hete_adj3_2]
            self.f_laps_static = [self.f_laps_static1, self.f_laps_static2, self.f_laps_static3]
          

    def load_dblp_data(self):
        features_a = sp.load_npz(self.predir + 'a_feat.npz').astype("float32")
        # features_1 = sp.load_npz(self.predir + '/features_1.npz').toarray()
        # features_2 = sp.load_npz(self.predir + '/features_2.npy')
        features_a = t.FloatTensor(preprocess_features(features_a))
        
        apa_mat=sp.load_npz(self.predir + "apa.npz")
        ata_mat=sp.load_npz(self.predir + "apcpa.npz")
        ava_mat=sp.load_npz(self.predir + "aptpa.npz")
        labels = np.load(self.predir + 'labels.npy')
        labels = encode_onehot(labels)
        labels= t.FloatTensor(labels).to(device)
        train = [np.load(self.predir + "train_" + str(i) + ".npy") for i in args.ratio]
        test = [np.load(self.predir + "test_" + str(i) + ".npy") for i in args.ratio]
        val = [np.load(self.predir + "val_" + str(i) + ".npy") for i in args.ratio]
        train = [t.LongTensor(i) for i in train]
        val = [t.LongTensor(i) for i in val]
        test = [t.LongTensor(i) for i in test]
        
        return features_a,apa_mat,ata_mat,ava_mat,train,val,test,labels
    

    def load_aminer_data(self):
        type_num = [6564, 13329, 35890]
       
        # features_1 = sp.load_npz(self.predir + '/features_1.npz').toarray()
        # features_2 = sp.load_npz(self.predir + '/features_2.npy')
        features_p = sp.eye(type_num[0])
        features_p=t.FloatTensor(preprocess_features(features_p))
        pap = sp.load_npz(self.predir + "pap.npz")
        prp = sp.load_npz(self.predir + "prp.npz")
        pos = sp.load_npz(self.predir + "pos.npz")
        labels = np.load(self.predir + 'labels.npy')
        labels = encode_onehot(labels)
        labels = t.FloatTensor(labels).to(device)
        train = [np.load(self.predir + "train_" + str(i) + ".npy") for i in args.ratio]
        test = [np.load(self.predir + "test_" + str(i) + ".npy") for i in args.ratio]
        val = [np.load(self.predir + "val_" + str(i) + ".npy") for i in args.ratio]
        train = [t.LongTensor(i) for i in train]
        val = [t.LongTensor(i) for i in val]
        test = [t.LongTensor(i) for i in test]
        
        return features_p,pap,prp,pos,train,val,test,labels

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
