# -*- coding: utf-8 -*-
# @Time    : 2019/3/4  16:27
# @Author  : Dyn
import gc
import numpy as np
from tqdm import tqdm
from model.base_model.base_recommender import BaseRecommendationModel
from utils.early_stopping import EarlyStopping
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from config.config import *
from model.geo.KMeans_grid import KMeansGeoGrid
import pickle


class GeoPairMatrixFactorization(BaseRecommendationModel):
    def __init__(self, k, learning_rate, lambda_=0.0, kesai=0.0, negative_rate=1.0, max_epoch=100,
                 geo_rate=0.1, beta=1.0, cluster_num=100):
        BaseRecommendationModel.__init__(self)
        self.k = k
        self.user_factors = None
        self.item_factors = None
        # mask matrix stands for the weights of loss
        self.mask_matrix = None
        self.predict_matrix = None
        self.geo_grid = None
        self.lambda_ = lambda_
        self.eta = learning_rate
        self.epoch = max_epoch
        self.kesai = kesai
        self.negative_rate = negative_rate
        self.rating_num = 0
        self.csr_matrix = None
        self.sgd_index = None
        self.log_loss = []

        self.poi_influence_area = None
        self.user_active_area = None
        self.geo_rate = geo_rate
        self.beta = beta
        self.cluster_num = cluster_num

    def residual_compute(self):
        matrix = self.dataset.check_ins_matrix
        residual = self.mask_matrix * (matrix - np.dot(self.user_factors, self.item_factors.T) -
                                       np.dot(self.user_active_area, self.poi_influence_area.T))
        return residual

    def build(self, low=-0.001, high=0.001, x_grid=0.005, y_grid=0.005):

        if self.user_factors is not None:
            del self.user_factors, self.item_factors, self.mask_matrix
            gc.collect()
        self.user_factors = np.random.uniform(low, high, size=[self.user_num, self.k])
        self.item_factors = np.random.uniform(low, high, size=[self.item_num, self.k])
        self.grid = KMeansGeoGrid(self.dataset, self.beta, self.cluster_num)
        self.poi_influence_area = self.grid.poi_influence_area
        self.user_active_area = np.random.uniform(low, high, size=[self.user_num, self.grid.num_grid])

        matrix = self.dataset.check_ins_matrix
        # mask matrix for the positve/negative instance
        self.mask_matrix = np.zeros_like(matrix)
        self.mask_matrix[matrix != 0] = 1
        self.rating_num = int(np.sum(self.mask_matrix[matrix != 0]))
        # set negative instance loss weight to kesai
        if self.kesai and self.negative_rate:
            self.mask_matrix[matrix != 0] = 1 + self.kesai
            self.mask_matrix[matrix == 0] = self.kesai

    def _sparse_data(self):
        index = []
        for i in range(self.dataset.user_num):
            for j in range(self.dataset.item_num):
                if self.dataset.check_ins_matrix[i, j] != 0:
                    index.append((i, j))
        # negative sampling
        if self.kesai != 0 and self.negative_rate != 1 and self.negative_rate != 0:
            total_neg_num = int((self.user_num * self.item_num - len(index))
                                * self.negative_rate)
            negative_set = set()
            for i in range(total_neg_num):
                while True:
                    u = np.random.randint(low=0, high=self.user_num)
                    v = np.random.randint(low=0, high=self.item_num)
                    if self.dataset.check_ins_matrix[u, v] == 0 and (u, v) not in negative_set:
                        negative_set.add((u, v))
                        break
            index.extend(list(negative_set))
        np.random.shuffle(index)
        self.sgd_index = index

    def als_update(self):
        if self.csr_matrix is None:
            self.csr_matrix = self.dataset.data_csr_matrix()
            self.user_factors = csr_matrix(np.random.normal(size=(self.user_num, self.k)))
            self.item_factors = csr_matrix(np.random.normal(size=(self.item_num, self.k)))
            self.UI = sparse.eye(self.user_num)
            self.VI = sparse.eye(self.item_num)
            self.I = sparse.eye(self.k)
            self.lambda_I = self.lambda_ * self.I

        vtv = self.item_factors.T.dot(self.item_factors)
        for i in range(self.user_num):
            ci = sparse.diags(self.mask_matrix[i, :][np.newaxis, :], [0])
            left = vtv + self.item_factors.T.dot(ci - self.VI).dot(self.item_factors)
            ri = self.dataset.check_ins_matrix[i, :][:, np.newaxis]
            ri[ri != 0] = 1.0
            ri_sub_YX_u = ri - np.dot(self.poi_influence_area, self.user_active_area[i])[:, np.newaxis]
            right = self.item_factors.T.dot(ci).dot(ri_sub_YX_u)
            self.user_factors[i] = spsolve(left, right)

        utu = self.user_factors.T.dot(self.user_factors)
        for i in range(self.item_num):
            ci = sparse.diags(self.mask_matrix[:, i][np.newaxis, :], [0])
            left = utu + self.user_factors.T.dot(ci - self.UI).dot(self.user_factors)
            ri = self.dataset.check_ins_matrix[:, i][:, np.newaxis]
            ri[ri != 0] = 1.0
            ri_sub_YX_u = ri - np.dot(self.user_active_area, self.poi_influence_area[i])[:, np.newaxis]
            right = self.user_factors.T.dot(ci).dot(ri_sub_YX_u)
            self.item_factors[i] = spsolve(left, right)

        for i in range(self.user_num):
            ci = sparse.diags(self.mask_matrix[i, :][np.newaxis, :], [0]).toarray()
            YTCi = np.dot(self.poi_influence_area.T, ci)
            ri = self.dataset.check_ins_matrix[i, :][:, np.newaxis]
            ri[ri != 0] = 1.0
            left_part1 = np.dot(self.poi_influence_area, self.user_active_area[i][:, np.newaxis])
            left_part2 = ri - np.dot(self.item_factors, self.user_factors[i].T)
            left_part = left_part1 - left_part2
            partial_gradients = np.dot(YTCi, left_part) + self.lambda_
            self.user_active_area[i] -= self.eta * np.array(np.squeeze(partial_gradients).tolist()[0])
            self.user_active_area[i][self.user_active_area[i] < 0] = 0

        residual = self.mask_matrix * (self.dataset.check_ins_matrix - self.user_factors.dot(self.item_factors.T).toarray() -
                                       np.dot(self.user_active_area, self.poi_influence_area.T))

        return float(np.sum(residual ** 2))

        # vtv = np.dot(self.item_factors.T , self.item_factors)
        # for i in tqdm(range(self.user_num)):
        #     ci = np.diag(self.mask_matrix[i, :])
        #     before_inv = vtv + self.lambda_ * np.eye(self.k)+ np.dot(np.dot(self.item_factors.T,
        #                                      ci-np.eye(self.item_num)), self.item_factors)
        #     inv = np.linalg.inv(before_inv)
        #     ui = np.dot(inv, self.item_factors.T)
        #     ui = np.dot(ui, ci)
        #     ui = np.dot(ui, self.mask_matrix[i, :][:, np.newaxis])
        #     self.user_factors[i] = np.squeeze(ui)
        #
        # utu = np.dot(self.user_factors.T , self.user_factors)
        # for i in tqdm(range(self.item_num)):
        #     ci = np.diag(self.mask_matrix[:, i])
        #     before_inv = utu + self.lambda_ * np.eye(self.k)+ np.dot(np.dot(self.user_factors.T,
        #                                      ci-np.eye(self.user_num)), self.user_factors)
        #     inv = np.linalg.inv(before_inv)
        #     vi = np.dot(inv, self.user_factors.T)
        #     vi = np.dot(vi, ci)
        #     vi = np.dot(vi, self.mask_matrix[:, i][:, np.newaxis])
        #     self.item_factors[i] = np.squeeze(vi)
        # residual = self.residual_compute()
        # return residual

    def sgd_update(self):
        if self.sgd_index is None:
            self._sparse_data()
        for r in self.sgd_index:
            user_id = r[0]
            location_id = r[1]
            rate = self.dataset.check_ins_matrix[user_id, location_id]
            weight = self.mask_matrix[user_id, location_id]

            error = weight * (rate - np.dot(self.user_factors[user_id], self.item_factors[location_id].T))

            u = self.user_factors[user_id] + self.eta * (error * self.item_factors[location_id] -
                                                         self.lambda_ * self.user_factors[user_id])
            v = self.item_factors[location_id] + self.eta * (error * self.user_factors[user_id] -
                                                             self.lambda_ * self.item_factors[location_id])

            self.user_factors[user_id] = u
            self.item_factors[location_id] = v

        residual = self.residual_compute()
        return float(np.sum(residual ** 2))

    def batch_update(self):
        residual = self.residual_compute()
        temp_user_factors = self.user_factors * (1 - self.eta * self.lambda_) + \
            self.eta * np.dot(residual, self.item_factors)
        temp_item_factors = self.item_factors * (1 - self.eta * self.lambda_) + \
            self.eta * np.dot(residual.T, self.user_factors)
        temp_user_active_area_factors = self.user_active_area * (1 - self.eta * self.lambda_) + \
            self.eta * np.dot(residual, self.poi_influence_area)
        self.user_factors = temp_user_factors
        self.item_factors = temp_item_factors
        self.user_active_area = temp_user_active_area_factors
        # self.user_active_area[self.user_active_area < 0] = 0

        return float(np.sum(residual ** 2))

    def fit(self, alg='batch', k=3):
        es = EarlyStopping(k=k)
        user_latent_factors = None
        items_latent_factors = None
        for i in tqdm(range(self.epoch)):
            if alg == 'batch':
                loss = self.batch_update()
            elif alg == 'sgd':
                loss = self.sgd_update()
            elif alg == 'als':
                loss = self.als_update()
            print('[INFO] epoch:{0} loss:{1}'.format(i + 1, loss))
            if es.loss_compare(loss):
                user_latent_factors = np.array(self.user_factors)
                items_latent_factors = np.array(self.item_factors)
            if not es.log(loss):
                break
            self.eval(self.dataset.valid_ground_truth)
            self.eval(self.dataset.test_ground_truth)

        # restore best parameters
        self.user_factors = user_latent_factors
        self.item_factors = items_latent_factors

    def recommend(self, k=30):
        matrix = self.dataset.check_ins_matrix
        if isinstance(self.user_factors, np.ndarray):
            self.predict_matrix = np.dot(self.user_factors, self.item_factors.T) + np.dot(self.user_active_area,
                                                                                          self.poi_influence_area.T)
        elif isinstance(self.user_factors, sparse.csr_matrix):
            self.predict_matrix = self.user_factors.dot(self.item_factors.T).toarray()
        else:
            raise Exception("Factor dataset format should be np.ndarray or sparse.csr_matrix")

        self.predict_matrix[matrix != 0] = -np.inf
        return np.argsort(-self.predict_matrix)[:, :k]

    def eval(self, ground_truth):
        rec_list = self.recommend()
        precisions = []
        recalls = []
        hit_rates = []
        NDCGS = []
        for i in range(0, 31, 5):
            if i == 0:
                k = i + 1
            else:
                k = i
            precision = Estimator.precision_at_k(rec_list, ground_truth, k)
            recall = Estimator.recall_at_k(rec_list, ground_truth, k)
            hit_rate = Estimator.hit_rate_at_k(rec_list, ground_truth, k)
            NDCG = Estimator.NDCG_at_k(rec_list, ground_truth, k)
            precisions.append(round(precision, 5))
            recalls.append(round(recall, 5))
            hit_rates.append(round(hit_rate, 5))
            NDCGS.append(round(NDCG, 5))

        print('[INFO] precision {0}'.format(precisions))
        print('[INFO] recall {0}'.format(recalls))
        print('[INFO] hit rate {0}'.format(hit_rates))
        print('[INFO] NDCG {0}'.format(NDCGS))

        # return hit_rates

    def save(self):
        d = {'num_users': self.user_num, 'num_locations': self.item_num,
                'users_factors': self.user_factors, 'locations_factors': self.item_factors}
        pickle.dump(d, open(MF_MODEL, mode='wb'))


if __name__ == '__main__':
    from evaluate.estimator import Estimator
    from data.dataset import Dataset
    from utils.timer import Timer

    timer = Timer()
    print('[INFO] reading dataset')
    timer.start('reading dataset')
    # data = Dataset(rating_binary=True, rating_normalize=True)
    data = Dataset(geo=True, sparse=True)
    print(data.train.shape[0] / (data.user_num * data.item_num))
    print(data.user_num)
    print(data.item_num)
    timer.end()
    np.random.seed(15)

    # mf = GeoMatrixFactorization(k=50, learning_rate=0.01,
    #                          lambda_=0.001, kesai=0.002,
    #                          max_epoch=50, negative_rate=0.002)
    # print(mf.k, mf.eta, mf.lambda_, mf.kesai)
    mf = GeoPairMatrixFactorization(k=40, learning_rate=0.01,
                                    lambda_=0.001, kesai=0.002,
                                    max_epoch=50, negative_rate=0.002,
                                    cluster_num=200)
    print('k:', mf.k, 'learning_rate', mf.eta, 'lambda_', mf.lambda_,
          'kesat:', mf.kesai)
    mf.set_dataset(data)
    mf.build(0, 0.01, 0.005, 0.005)
    mf.fit(alg='batch')