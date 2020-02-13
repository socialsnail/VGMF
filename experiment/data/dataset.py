# -*- coding: utf-8 -*-
# @Time    : 2018/6/17  22:15
# @Author  : Dyn

import os

import pandas as pd
import pickle
import numpy as np
from scipy.sparse import csr_matrix

from config.config import *


class Dataset(object):
    def __init__(self, rating_binary=True, rating_normalize=True, weight=0, geo=False,
                 restart=False, sparse=False):
        """Dataset class for training
        Args:
            rating_binary(bool): using 0/1 rating or frequency
            rating_normalize(bool): normalize rating  or not, only using when rating_binary = True
                    rating [2, 1, 0, 0] when normalize [2 / 3, 1 / 3, 0, 0]
            weight(float): sample negative instance proportion for the dataset
            geo(bool): need cal geo distance
        """
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
        if sparse:
            self.sparse_data = self._sparse_data()
        # if geo, calculate locations distance
        if geo:
            self._location_matrix()

    def _sparse_data(self):
        """Gen sparse dataset
        format (user_id, location_id, rating, weight)
        """
        if self.restart and os.path.exists(SPARSE_DATA):
            print('[INFO] remove sparse.pkl')
            os.remove(SPARSE_DATA)

        if not os.path.exists(SPARSE_DATA):
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
            pickle.dump(sparse_data, open(SPARSE_DATA, mode='wb'))
            return sparse_data
        else:
            print('[INFO] reading sparse file from disk')
            return pickle.load(open(SPARSE_DATA, mode='rb'))

    def data_csr_matrix(self):
        """Generate csr format dataset
        """
        rows = self.train.user_id.values
        cols = self.train.location_id.values
        values = np.asarray([1 for x in rows])
        return csr_matrix((values, (rows, cols)), shape=(self.user_num, self.item_num))

    def _read_from_disk(self):
        """Read train valid test dataset from disk
        """
        self.train = pd.read_csv(TRAIN)
        self.valid = pd.read_csv(VALID)
        self.test = pd.read_csv(TEST)

        self.user_num = np.max([self.train.user_id.max(),
                                self.valid.user_id.max(),
                                self.test.user_id.max()]) + 1
        self.item_num = np.max([self.train.location_id.max(),
                                self.valid.location_id.max(),
                                self.test.location_id.max()]) + 1

        self.valid_ground_truth = pickle.load(open(VALID_GROUND_TRUTH, 'rb'))
        self.test_ground_truth = pickle.load(open(TEST_GROUND_TRUTH, 'rb'))

    def get_sparse(self):
        """Get sparse format of the dataset
        """
        return self.sparse_data

    def _check_ins(self):
        """Set check_ins matrix
        """
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
        """Init longitude and latitude matrix and distance matrix between locations
        Given two geo location, long1 lat1 long2 lat2, their distance will be calculate as follows
            # transform to radians
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371
            return c * r #* 1000
        """

        if self.restart and os.path.exists(DISTANCE):
            print('[INFO] remove distance.pkl')
            os.remove(DISTANCE)

        if not os.path.exists(DISTANCE):
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

            self.distance_matrix = 2 * 6371 * np.arcsin(np.sqrt(matrix)) * 1000
            pickle.dump(self.distance_matrix, open(DISTANCE, mode='wb'))
        else:
            print('[INFO] reading distance file from disk')
            self.distance_matrix = pickle.load(open(DISTANCE, mode='rb'))


if __name__ == '__main__':
    # data = Dataset(restart=True, sparse=True)
    data = Dataset(restart=True, geo=True)
    # print(data.check_ins_matrix.shape)
    print(data.distance_matrix)
    print(data.distance_matrix.max())