# -*- coding: utf-8 -*-
# @Time    : 2018/6/17  22:15
# @Author  : Dyn

import os

import pandas as pd
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from config.config import *


class Dataset(object):
    def __init__(self, rating_binary=True, rating_normalize=True, sparse=False, weight=0, geo=False,
                 restart=False):
        self.rating_binary = rating_binary
        self.rating_normalize = rating_normalize

        self.user_num = None
        self.item_num = None

        self.train, self.valid, self.test = None, None, None
        self.ground_truth = None
        # read from disk
        self._read_from_disk()
        self.check_ins_matrix = None
        # set check-ins matrix
        self._check_ins()

        self.sparse_data = None
        self.weight = weight

        self.restart = restart

        # if sparse, create sparse check-ins dataset
        if sparse:
            self.sparse_data = self._sparse_data()

        # if geo, calculate locations distance
        if geo:
            self._location_matrix()

    def data_csr_matrix(self):
        rows = self.train.user_id.values
        cols = self.train.location_id.values
        values = np.asarray([1 for x in rows])
        return csr_matrix((values, (rows, cols)), shape=(self.user_num, self.item_num))

    def _sparse_data(self):
        if self.restart and os.path.exists(os.sep.join([DATA_ROOT, 'sparse.pkl'])):
            print('[INFO] remove sparse.pkl')
            os.remove(os.sep.join([DATA_ROOT, 'sparse.pkl']))

        if not os.path.exists(os.sep.join([DATA_ROOT, 'sparse.pkl'])):
            sparse_data = {}
            count = 0
            for i in range(self.user_num):
                sparse_data[i] = []
                for j in range(self.item_num):
                    if self.check_ins_matrix[i, j] != 0:
                        sparse_data[i].append(j)
                        count += 1
                    # else:
                    #     sparse_data.append([i, j, self.check_ins_matrix[i, j], self.weight])
            self.rating_count = count
            pickle.dump(sparse_data, open(os.sep.join([DATA_ROOT, 'sparse.pkl']), mode='wb'))
            return sparse_data
        else:
            print('[INFO] reading sparse file from disk')
            return pickle.load(open(os.sep.join([DATA_ROOT, 'sparse.pkl']), mode='rb'))

    def _read_from_disk(self):
        self.train = pd.read_csv(os.sep.join([DATA_ROOT, 'train.csv']))
        self.valid = pd.read_csv(os.sep.join([DATA_ROOT, 'valid.csv']))
        self.test = pd.read_csv(os.sep.join([DATA_ROOT, 'test.csv']))

        self.user_num = np.max([self.train.user_id.max(),
                                self.valid.user_id.max(),
                                self.test.user_id.max()]) + 1
        self.item_num = np.max([self.train.location_id.max(),
                                self.valid.location_id.max(),
                                self.test.location_id.max()]) + 1

        self.valid_ground_truth = pickle.load(open(os.sep.join([DATA_ROOT, 'valid_ground_truth.pkl']), 'rb'))
        self.test_ground_truth = pickle.load(open(os.sep.join([DATA_ROOT, 'test_ground_truth.pkl']), 'rb'))

    def _read_with_filtering(self, data_path):
        location_id = set(pickle.load(open(os.sep.join([CONFIG, 'location_id.pkl']), mode='rb')))
        user_id = set(pickle.load(open(os.sep.join([CONFIG, 'user_id.pkl']), mode='rb')))

        print('[INFO] reading dataset from disk')
        df_origin = pd.read_csv(data_path)
        df_origin = df_origin.drop_duplicates(subset=['user_id', 'location_id']).copy()
        df_origin = df_origin[(df_origin.user_id in user_id) &
                              (df_origin.location_id in location_id)]
        lb = LabelEncoder()
        df_origin['user_id'] = lb.fit_transform(df_origin.user_id)
        df_origin['location_id'] = lb.fit_transform(df_origin.location_id)
        df_origin.reset_index(drop=True, inplace=True)
        return df_origin

    def get_sparse(self):
        return self.sparse_data

    def _check_ins(self):
        self.check_ins_matrix = np.zeros([self.user_num, self.item_num])

        for location_id, user_id, visited_num in self.train[['location_id', 'user_id', 'visited_num']].values:
            if self.rating_binary:
                self.check_ins_matrix[user_id][location_id] = 1
            else:
                self.check_ins_matrix[user_id][location_id] = visited_num

        if not self.rating_binary and self.rating_normalize:
            self.check_ins_matrix = self.check_ins_matrix / \
                                    np.sum(self.check_ins_matrix, axis=1)[:, np.newaxis]

    def _location_matrix(self):
        if self.restart and os.path.exists(os.sep.join([DATA_ROOT, 'distance.pkl'])):
            print('[INFO] remove distance.pkl')
            os.remove(os.sep.join([DATA_ROOT, 'distance.pkl']))

        if not os.path.exists(os.sep.join([DATA_ROOT, 'distance.pkl'])):
            temp = self.train[['location_id', 'longitude', 'latitude']].drop_duplicates(subset='location_id')
            sort_location = temp.sort_values(by='location_id')

            self.longitude_vector = sort_location.longitude.values
            self.latitude_vector = sort_location.latitude.values

            longitude_radian = np.radians(self.longitude_vector)
            latitude_radian = np.radians(self.latitude_vector)

            longitude_matrix_1 = longitude_radian.T[:, np.newaxis].repeat(self.item_num, 1)
            longitude_matrix_2 = longitude_radian[np.newaxis, :].repeat(self.item_num, 0)
            longitude_matrix = longitude_matrix_1 - longitude_matrix_2

            latitude_matrix_1 = latitude_radian.T[:, np.newaxis].repeat(self.item_num, 1)
            latitude_matrix_2 = latitude_radian[np.newaxis, :].repeat(self.item_num, 0)
            latitude_matrix = latitude_matrix_1 - latitude_matrix_2

            matrix = np.sin(latitude_matrix / 2) ** 2 + \
                     np.cos(latitude_matrix_1) * np.cos(latitude_matrix_2) * (np.sin(longitude_matrix / 2) ** 2)

            self.distance_matrix = 2 * 6371 * np.arcsin(np.sqrt(matrix))
            pickle.dump(self.distance_matrix, open(os.sep.join([DATA_ROOT, 'distance.pkl']), mode='wb'))
        else:
            print('[INFO] reading distance file from disk')
            self.distance_matrix = pickle.load(open(os.sep.join([DATA_ROOT, 'distance.pkl']), mode='rb'))


if __name__ == '__main__':
    data = Dataset(geo=True)
    print(data.distance_matrix)