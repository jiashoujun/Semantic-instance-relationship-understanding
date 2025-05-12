import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import torch as th

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir
from utils import to_etype_name



READ_DATASET_PATH = get_download_dir()

class MovieLens(object):

    def __init__(self, name, device, mix_cpu_gpu=False,
                 use_one_hot_fea=False, symm=True,
                 test_ratio=0.1, valid_ratio=0.1):
        self._name = name
        self._device = device
        self._symm = symm
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        self._dir = '/media/sever/data1/jsj/dgl-master(1)/dgl-master/examples/pytorch/gcmc/tongji'
        print("Starting processing {} ...".format(self._name))
        self._load_raw_user_info()
        self._load_raw_movie_info()
        print('......')
        if self._name == 'tongji':
            self.all_train_rating_info = self._load_raw_rates(os.path.join(self._dir, 'u1.base'), '\s+')
            self.test_rating_info = self._load_raw_rates(os.path.join(self._dir, 'u1.test'), '\s+')
            self.all_rating_info = pd.concat([self.all_train_rating_info, self.test_rating_info])
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
            self.all_rating_info = self._load_raw_rates(os.path.join(self._dir, 'ratings.dat'), '::')
            num_test = int(np.ceil(self.all_rating_info.shape[0] * self._test_ratio))
            shuffled_idx = np.random.permutation(self.all_rating_info.shape[0])
            self.test_rating_info = self.all_rating_info.iloc[shuffled_idx[: num_test]]
            self.all_train_rating_info = self.all_rating_info.iloc[shuffled_idx[num_test: ]]
        else:
            raise NotImplementedError
        print('......')
        num_valid = int(np.ceil(self.all_train_rating_info.shape[0] * self._valid_ratio))
        shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
        self.valid_rating_info = self.all_train_rating_info.iloc[shuffled_idx[: num_valid]]
        self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[num_valid: ]]
        self.possible_rating_values = np.unique(self.train_rating_info["rating"].values)

        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print("\tAll train rating pairs : {}".format(self.all_train_rating_info.shape[0]))
        print("\t\tTrain rating pairs : {}".format(self.train_rating_info.shape[0]))
        print("\t\tValid rating pairs : {}".format(self.valid_rating_info.shape[0]))
        print("\tTest rating pairs  : {}".format(self.test_rating_info.shape[0]))

        self.user_info = self._drop_unseen_nodes(orign_info=self.user_info,
                                                 cmp_col_name="id",
                                                 reserved_ids_set=set(self.all_rating_info["user_id"].values),
                                                 label="user")
        self.movie_info = self._drop_unseen_nodes(orign_info=self.movie_info,
                                                  cmp_col_name="id",
                                                  reserved_ids_set=set(self.all_rating_info["movie_id"].values),
                                                  label="movie")

        self.global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
        self.global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['id'])}
        print('Total user number = {}, movie number = {}'.format(len(self.global_user_id_map),
                                                                 len(self.global_movie_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_movie = len(self.global_movie_id_map)


        if use_one_hot_fea:
            self.user_feature = None
            self.movie_feature = None
        else:
            # if mix_cpu_gpu, we put features in CPU
            if mix_cpu_gpu:
                self.user_feature = th.FloatTensor(self._process_user_fea())
                self.movie_feature = th.FloatTensor(self._process_movie_fea())
            else:
                self.user_feature = th.FloatTensor(self._process_user_fea()).to(self._device)
                self.movie_feature = th.FloatTensor(self._process_movie_fea()).to(self._device)
        if self.user_feature is None:
            self.user_feature_shape = (self.num_user, self.num_user)
            self.movie_feature_shape = (self.num_movie, self.num_movie)
        else:
            self.user_feature_shape = self.user_feature.shape
            self.movie_feature_shape = self.movie_feature.shape
        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)

        all_train_rating_pairs, all_train_rating_values = self._generate_pair_value(self.all_train_rating_info)
        train_rating_pairs, train_rating_values = self._generate_pair_value(self.train_rating_info)
        valid_rating_pairs, valid_rating_values = self._generate_pair_value(self.valid_rating_info)
        test_rating_pairs, test_rating_values = self._generate_pair_value(self.test_rating_info)

        def _make_labels(ratings):
            labels = th.LongTensor(np.searchsorted(self.possible_rating_values, ratings)).to(device)
            return labels

        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs, train_rating_values, add_support=True)
        self.train_dec_graph = self._generate_dec_graph(train_rating_pairs)
        self.train_labels = _make_labels(train_rating_values)
        self.train_truths = th.FloatTensor(train_rating_values).to(device)

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs)
        self.valid_labels = _make_labels(valid_rating_values)
        self.valid_truths = th.FloatTensor(valid_rating_values).to(device)

        self.test_enc_graph = self._generate_enc_graph(all_train_rating_pairs, all_train_rating_values, add_support=True)
        self.test_dec_graph = self._generate_dec_graph(test_rating_pairs)
        self.test_labels = _make_labels(test_rating_values)
        self.test_truths = th.FloatTensor(test_rating_values).to(device)

        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                rst += graph.number_of_edges(str(r))
            return rst

        print("Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('movie'),
            _npairs(self.train_enc_graph)))
        print("Train dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_dec_graph.number_of_nodes('user'), self.train_dec_graph.number_of_nodes('movie'),
            self.train_dec_graph.number_of_edges()))
        print("Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('movie'),
            _npairs(self.valid_enc_graph)))
        print("Valid dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_dec_graph.number_of_nodes('user'), self.valid_dec_graph.number_of_nodes('movie'),
            self.valid_dec_graph.number_of_edges()))
        print("Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('movie'),
            _npairs(self.test_enc_graph)))
        print("Test dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_dec_graph.number_of_nodes('user'), self.test_dec_graph.number_of_nodes('movie'),
            self.test_dec_graph.number_of_edges()))

    def _generate_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                                 dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        user_movie_R = np.zeros((self._num_user, self._num_movie), dtype=np.float32)
        user_movie_R[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {'user': self._num_user, 'movie': self._num_movie}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)
            user_ci = []
            user_cj = []
            movie_ci = []
            movie_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                user_ci.append(graph['rev-%s' % r].in_degrees())
                movie_ci.append(graph[r].in_degrees())
                if self._symm:
                    user_cj.append(graph[r].out_degrees())
                    movie_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    user_cj.append(th.zeros((self.num_user,)))
                    movie_cj.append(th.zeros((self.num_movie,)))
            user_ci = _calc_norm(sum(user_ci))
            movie_ci = _calc_norm(sum(movie_ci))
            if self._symm:
                user_cj = _calc_norm(sum(user_cj))
                movie_cj = _calc_norm(sum(movie_cj))
            else:
                user_cj = th.ones(self.num_user,)
                movie_cj = th.ones(self.num_movie,)
            graph.nodes['user'].data.update({'ci' : user_ci, 'cj' : user_cj})
            graph.nodes['movie'].data.update({'ci' : movie_ci, 'cj' : movie_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_user, self.num_movie), dtype=np.float32)
        g = dgl.bipartite_from_scipy(user_movie_ratings_coo, utype='_U', etype='_E', vtype='_V')
        return dgl.heterograph({('user', 'rate', 'movie'): g.edges()}, 
                               num_nodes_dict={'user': self.num_user, 'movie': self.num_movie})

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

    def _drop_unseen_nodes(self, orign_info, cmp_col_name, reserved_ids_set, label):
        if reserved_ids_set != set(orign_info[cmp_col_name].values):
            pd_rating_ids = pd.DataFrame(list(reserved_ids_set), columns=["id_graph"])
            data_info = orign_info.merge(pd_rating_ids, left_on=cmp_col_name, right_on='id_graph', how='outer')
            data_info = data_info.dropna(subset=[cmp_col_name, 'id_graph'])
            data_info = data_info.drop(columns=["id_graph"])
            data_info = data_info.reset_index(drop=True)
            return data_info
        else:
            orign_info = orign_info.reset_index(drop=True)
            return orign_info

    def _load_raw_rates(self, file_path, sep):
        rating_info = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.float32}, engine='python')
        return rating_info

    def _load_raw_user_info(self):

            self.user_info = pd.read_csv(os.path.join(self._dir, 'u.user'), sep='\s+', header=None,
                                    names=['id', 'X', 'Y', 'Z', 'R', 'G', 'B', 'L', 'M', 'H', 'I',
                                                 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20',
                                                 'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30',
                                                 'F31', 'F32', 'F33', 'F34', 'F35', 'F36', 'F37', 'F38', 'F39', 'F40',
                                                 'F41', 'F42', 'F43', 'F44', 'F45', 'F46', 'F47', 'F48', 'F49', 'F50',
                                                 'F51', 'F52', 'F53', 'F54', 'F55', 'F56', 'F57', 'F58', 'F59', 'F60',
                                                 'F61', 'F62', 'F63', 'F64', 'F65', 'F66', 'F67', 'F68', 'F69', 'F70',
                                                 'F71', 'F72', 'F73', 'F74', 'F75', 'F76', 'F77', 'F78', 'F79', 'F80',
                                                 'F81', 'F82', 'F83', 'F84', 'F85', 'F86', 'F87', 'F88', 'F89', 'F90',
                                                 'F91', 'F92', 'F93', 'F94', 'F95', 'F96', 'F97', 'F98', 'F99', 'F100',
                                                 'F101','F102','F103','F104','F105','F106','F107','F108','F109','F110'], engine='python')



    def _process_user_fea(self):

        if self._name == 'tongji' or self._name == 'ml-1m':
            user_features = np.zeros(shape=(self.user_info.shape[0], 110))
            user_features[:,0] = (self.user_info['X'])
            user_features[:,1] = (self.user_info['Y'])
            user_features[:,2] = (self.user_info['Z'])
            user_features[:,3] = (self.user_info['R'])
            user_features[:,4] = (self.user_info['G'])
            user_features[:,5] = (self.user_info['B'])
            user_features[:,6] = (self.user_info['L'])
            user_features[:,7] = (self.user_info['M'])
            user_features[:,8] = (self.user_info['H'])
            user_features[:,9] = (self.user_info['I'])
            user_features[:, 10] = (self.user_info['F11'])
            user_features[:, 11] = (self.user_info['F12'])
            user_features[:, 12] = (self.user_info['F13'])
            user_features[:, 13] = (self.user_info['F14'])
            user_features[:, 14] = (self.user_info['F15'])
            user_features[:, 15] = (self.user_info['F16'])
            user_features[:, 16] = (self.user_info['F17'])
            user_features[:, 17] = (self.user_info['F18'])
            user_features[:, 18] = (self.user_info['F19'])
            user_features[:, 19] = (self.user_info['F20'])
            user_features[:, 20] = (self.user_info['F21'])
            user_features[:, 21] = (self.user_info['F22'])
            user_features[:, 22] = (self.user_info['F23'])
            user_features[:, 23] = (self.user_info['F24'])
            user_features[:, 24] = (self.user_info['F25'])
            user_features[:, 25] = (self.user_info['F26'])
            user_features[:, 26] = (self.user_info['F27'])
            user_features[:, 27] = (self.user_info['F28'])
            user_features[:, 28] = (self.user_info['F29'])
            user_features[:, 29] = (self.user_info['F30'])
            user_features[:, 30] = (self.user_info['F31'])
            user_features[:, 31] = (self.user_info['F32'])
            user_features[:, 32] = (self.user_info['F33'])
            user_features[:, 33] = (self.user_info['F34'])
            user_features[:, 34] = (self.user_info['F35'])
            user_features[:, 35] = (self.user_info['F36'])
            user_features[:, 36] = (self.user_info['F37'])
            user_features[:, 37] = (self.user_info['F38'])
            user_features[:, 38] = (self.user_info['F39'])
            user_features[:, 39] = (self.user_info['F40'])
            user_features[:, 40] = (self.user_info['F41'])
            user_features[:, 41] = (self.user_info['F42'])
            user_features[:, 42] = (self.user_info['F43'])
            user_features[:, 43] = (self.user_info['F44'])
            user_features[:, 44] = (self.user_info['F45'])
            user_features[:, 45] = (self.user_info['F46'])
            user_features[:, 46] = (self.user_info['F47'])
            user_features[:, 47] = (self.user_info['F48'])
            user_features[:, 48] = (self.user_info['F49'])
            user_features[:, 49] = (self.user_info['F50'])
            user_features[:, 50] = (self.user_info['F51'])
            user_features[:, 51] = (self.user_info['F52'])
            user_features[:, 52] = (self.user_info['F53'])
            user_features[:, 53] = (self.user_info['F54'])
            user_features[:, 54] = (self.user_info['F55'])
            user_features[:, 55] = (self.user_info['F56'])
            user_features[:, 56] = (self.user_info['F57'])
            user_features[:, 57] = (self.user_info['F58'])
            user_features[:, 58] = (self.user_info['F59'])
            user_features[:, 59] = (self.user_info['F60'])
            user_features[:, 60] = (self.user_info['F61'])
            user_features[:, 61] = (self.user_info['F62'])
            user_features[:, 62] = (self.user_info['F63'])
            user_features[:, 63] = (self.user_info['F64'])
            user_features[:, 64] = (self.user_info['F65'])
            user_features[:, 65] = (self.user_info['F66'])
            user_features[:, 66] = (self.user_info['F67'])
            user_features[:, 67] = (self.user_info['F68'])
            user_features[:, 68] = (self.user_info['F69'])
            user_features[:, 69] = (self.user_info['F70'])
            user_features[:, 70] = (self.user_info['F71'])
            user_features[:, 71] = (self.user_info['F72'])
            user_features[:, 72] = (self.user_info['F73'])
            user_features[:, 73] = (self.user_info['F74'])
            user_features[:, 74] = (self.user_info['F75'])
            user_features[:, 75] = (self.user_info['F76'])
            user_features[:, 76] = (self.user_info['F77'])
            user_features[:, 77] = (self.user_info['F78'])
            user_features[:, 78] = (self.user_info['F79'])
            user_features[:, 79] = (self.user_info['F80'])
            user_features[:, 80] = (self.user_info['F81'])
            user_features[:, 81] = (self.user_info['F82'])
            user_features[:, 82] = (self.user_info['F83'])
            user_features[:, 83] = (self.user_info['F84'])
            user_features[:, 84] = (self.user_info['F85'])
            user_features[:, 85] = (self.user_info['F86'])
            user_features[:, 86] = (self.user_info['F87'])
            user_features[:, 87] = (self.user_info['F88'])
            user_features[:, 88] = (self.user_info['F89'])
            user_features[:, 89] = (self.user_info['F90'])
            user_features[:, 90] = (self.user_info['F91'])
            user_features[:, 91] = (self.user_info['F92'])
            user_features[:, 92] = (self.user_info['F93'])
            user_features[:, 93] = (self.user_info['F94'])
            user_features[:, 94] = (self.user_info['F95'])
            user_features[:, 95] = (self.user_info['F96'])
            user_features[:, 96] = (self.user_info['F97'])
            user_features[:, 97] = (self.user_info['F98'])
            user_features[:, 98] = (self.user_info['F99'])
            user_features[:, 99] = (self.user_info['F100'])
            user_features[:, 100] = (self.user_info['F101'])
            user_features[:, 101] = (self.user_info['F102'])
            user_features[:, 102] = (self.user_info['F103'])
            user_features[:, 103] = (self.user_info['F104'])
            user_features[:, 104] = (self.user_info['F105'])
            user_features[:, 105] = (self.user_info['F106'])
            user_features[:, 106] = (self.user_info['F107'])
            user_features[:, 107] = (self.user_info['F108'])
            user_features[:, 108] = (self.user_info['F109'])
            user_features[:, 109] = (self.user_info['F110'])

            XYZ = user_features[:, 0:3]
            I =  user_features[:, 9:10]
            RGB = user_features[:, 3:6]
            LMH = user_features[:, 6:9]
            F = user_features[:, 3:10]
            G = user_features[:, 10:55]
            P = user_features[:, 55:110]
            GP = user_features[:, 10:110]
            FGP = user_features[:, 3:110]
            XYZFGP = user_features[:, 0:110]
            FG = user_features[:, 3:55]

            user_features = FG     


        elif self._name == 'ml-10m':
            user_features = np.zeros(shape=(self.user_info.shape[0], 1), dtype=np.float32)
        else:
            raise NotImplementedError
        return user_features

    def _load_raw_movie_info(self):

            file_path = os.path.join(self._dir, 'u.item')
            self.movie_info = pd.read_csv(file_path, sep='\s+', header=None,
                                          names=['id', 'X', 'Y', 'Z', 'R', 'G', 'B', 'L', 'M', 'H', 'I',
                                                 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20',
                                                 'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30',
                                                 'F31', 'F32', 'F33', 'F34', 'F35', 'F36', 'F37', 'F38', 'F39', 'F40',
                                                 'F41', 'F42', 'F43', 'F44', 'F45', 'F46', 'F47', 'F48', 'F49', 'F50',
                                                 'F51', 'F52', 'F53', 'F54', 'F55', 'F56', 'F57', 'F58', 'F59', 'F60',
                                                 'F61', 'F62', 'F63', 'F64', 'F65', 'F66', 'F67', 'F68', 'F69', 'F70',
                                                 'F71', 'F72', 'F73', 'F74', 'F75', 'F76', 'F77', 'F78', 'F79', 'F80',
                                                 'F81', 'F82', 'F83', 'F84', 'F85', 'F86', 'F87', 'F88', 'F89', 'F90',
                                                 'F91', 'F92', 'F93', 'F94', 'F95', 'F96', 'F97', 'F98', 'F99', 'F100',
                                                 'F101','F102','F103','F104','F105','F106','F107','F108','F109','F110'],
                                          encoding='iso-8859-1')



    def _process_movie_fea(self):

        user_features = np.zeros(shape=(self.user_info.shape[0], 110))
        user_features[:, 0] = (self.user_info['X'])
        user_features[:, 1] = (self.user_info['Y'])
        user_features[:, 2] = (self.user_info['Z'])
        user_features[:, 3] = (self.user_info['R'])
        user_features[:, 4] = (self.user_info['G'])
        user_features[:, 5] = (self.user_info['B'])
        user_features[:, 6] = (self.user_info['L'])
        user_features[:, 7] = (self.user_info['M'])
        user_features[:, 8] = (self.user_info['H'])
        user_features[:, 9] = (self.user_info['I'])
        user_features[:, 10] = (self.user_info['F11'])
        user_features[:, 11] = (self.user_info['F12'])
        user_features[:, 12] = (self.user_info['F13'])
        user_features[:, 13] = (self.user_info['F14'])
        user_features[:, 14] = (self.user_info['F15'])
        user_features[:, 15] = (self.user_info['F16'])
        user_features[:, 16] = (self.user_info['F17'])
        user_features[:, 17] = (self.user_info['F18'])
        user_features[:, 18] = (self.user_info['F19'])
        user_features[:, 19] = (self.user_info['F20'])
        user_features[:, 20] = (self.user_info['F21'])
        user_features[:, 21] = (self.user_info['F22'])
        user_features[:, 22] = (self.user_info['F23'])
        user_features[:, 23] = (self.user_info['F24'])
        user_features[:, 24] = (self.user_info['F25'])
        user_features[:, 25] = (self.user_info['F26'])
        user_features[:, 26] = (self.user_info['F27'])
        user_features[:, 27] = (self.user_info['F28'])
        user_features[:, 28] = (self.user_info['F29'])
        user_features[:, 29] = (self.user_info['F30'])
        user_features[:, 30] = (self.user_info['F31'])
        user_features[:, 31] = (self.user_info['F32'])
        user_features[:, 32] = (self.user_info['F33'])
        user_features[:, 33] = (self.user_info['F34'])
        user_features[:, 34] = (self.user_info['F35'])
        user_features[:, 35] = (self.user_info['F36'])
        user_features[:, 36] = (self.user_info['F37'])
        user_features[:, 37] = (self.user_info['F38'])
        user_features[:, 38] = (self.user_info['F39'])
        user_features[:, 39] = (self.user_info['F40'])
        user_features[:, 40] = (self.user_info['F41'])
        user_features[:, 41] = (self.user_info['F42'])
        user_features[:, 42] = (self.user_info['F43'])
        user_features[:, 43] = (self.user_info['F44'])
        user_features[:, 44] = (self.user_info['F45'])
        user_features[:, 45] = (self.user_info['F46'])
        user_features[:, 46] = (self.user_info['F47'])
        user_features[:, 47] = (self.user_info['F48'])
        user_features[:, 48] = (self.user_info['F49'])
        user_features[:, 49] = (self.user_info['F50'])
        user_features[:, 50] = (self.user_info['F51'])
        user_features[:, 51] = (self.user_info['F52'])
        user_features[:, 52] = (self.user_info['F53'])
        user_features[:, 53] = (self.user_info['F54'])
        user_features[:, 54] = (self.user_info['F55'])
        user_features[:, 55] = (self.user_info['F56'])
        user_features[:, 56] = (self.user_info['F57'])
        user_features[:, 57] = (self.user_info['F58'])
        user_features[:, 58] = (self.user_info['F59'])
        user_features[:, 59] = (self.user_info['F60'])
        user_features[:, 60] = (self.user_info['F61'])
        user_features[:, 61] = (self.user_info['F62'])
        user_features[:, 62] = (self.user_info['F63'])
        user_features[:, 63] = (self.user_info['F64'])
        user_features[:, 64] = (self.user_info['F65'])
        user_features[:, 65] = (self.user_info['F66'])
        user_features[:, 66] = (self.user_info['F67'])
        user_features[:, 67] = (self.user_info['F68'])
        user_features[:, 68] = (self.user_info['F69'])
        user_features[:, 69] = (self.user_info['F70'])
        user_features[:, 70] = (self.user_info['F71'])
        user_features[:, 71] = (self.user_info['F72'])
        user_features[:, 72] = (self.user_info['F73'])
        user_features[:, 73] = (self.user_info['F74'])
        user_features[:, 74] = (self.user_info['F75'])
        user_features[:, 75] = (self.user_info['F76'])
        user_features[:, 76] = (self.user_info['F77'])
        user_features[:, 77] = (self.user_info['F78'])
        user_features[:, 78] = (self.user_info['F79'])
        user_features[:, 79] = (self.user_info['F80'])
        user_features[:, 80] = (self.user_info['F81'])
        user_features[:, 81] = (self.user_info['F82'])
        user_features[:, 82] = (self.user_info['F83'])
        user_features[:, 83] = (self.user_info['F84'])
        user_features[:, 84] = (self.user_info['F85'])
        user_features[:, 85] = (self.user_info['F86'])
        user_features[:, 86] = (self.user_info['F87'])
        user_features[:, 87] = (self.user_info['F88'])
        user_features[:, 88] = (self.user_info['F89'])
        user_features[:, 89] = (self.user_info['F90'])
        user_features[:, 90] = (self.user_info['F91'])
        user_features[:, 91] = (self.user_info['F92'])
        user_features[:, 92] = (self.user_info['F93'])
        user_features[:, 93] = (self.user_info['F94'])
        user_features[:, 94] = (self.user_info['F95'])
        user_features[:, 95] = (self.user_info['F96'])
        user_features[:, 96] = (self.user_info['F97'])
        user_features[:, 97] = (self.user_info['F98'])
        user_features[:, 98] = (self.user_info['F99'])
        user_features[:, 99] = (self.user_info['F100'])
        user_features[:, 100] = (self.user_info['F101'])
        user_features[:, 101] = (self.user_info['F102'])
        user_features[:, 102] = (self.user_info['F103'])
        user_features[:, 103] = (self.user_info['F104'])
        user_features[:, 104] = (self.user_info['F105'])
        user_features[:, 105] = (self.user_info['F106'])
        user_features[:, 106] = (self.user_info['F107'])
        user_features[:, 107] = (self.user_info['F108'])
        user_features[:, 108] = (self.user_info['F109'])
        user_features[:, 109] = (self.user_info['F110'])

        XYZ = user_features[:, 0:3]
        I =   user_features[:, 9:10]
        RGB = user_features[:, 3:6]
        LMH = user_features[:, 6:9]
        F = user_features[:, 3:10]
        G = user_features[:, 10:55]
        P = user_features[:, 55:110]
        GP = user_features[:, 10:110]
        FGP = user_features[:, 3:110]
        XYZFGP = user_features[:, 0:110]
        FG = user_features[:, 3:55]

        movie_features = FG


        print(movie_features.shape)
        return movie_features

if __name__ == '__main__':
    MovieLens("tongji", device=th.device('cpu'), symm=True)
