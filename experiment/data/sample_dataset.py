# -*- coding: utf-8 -*-
# @Time    : 2018/12/18  16:38
# @Author  : Dyn
import gc

import pandas as pd
import pickle
import numpy as np


class DataReader(object):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

        self.user_num = self.data.user_id.unique().shape[0]
        self.item_num = self.data.location_id.unique().shape[0]

        # dataset for train, valid and test
        self.train, self.valid, self.test = self.train_test_gen()
        del self.data
        gc.collect()

        # check_ins matrix of users
        self.check_ins_matrix = self._check_ins_()

    def get_check_in_matrix(self):
        return self.check_ins_matrix

    def train_test_gen(self, valid_size=0.2, test_size=0.2):
        train_index = []
        valid_index = []
        test_index = []
        print('[INFO] gen dataset set')
        for user in self.data.user_id.unique():
            index = self.data[self.data.user_id == user].sort_values(by='time_stamp').index.values
            n = len(index)
            valid_size_ = int(np.ceil(n * valid_size))
            test_size_ = int(np.ceil(n * test_size))
            train_size_ = n - valid_size_ - test_size_
            train = index[:train_size_]
            valid = index[train_size_:train_size_+valid_size_]
            test = index[train_size_+valid_size_:]
            train_index.extend(train.tolist())
            valid_index.extend(valid.tolist())
            test_index.extend(test.tolist())

        return self.data.iloc[train_index].reset_index(drop=True), \
               self.data.iloc[valid_index].reset_index(drop=True), \
               self.data.iloc[test_index].reset_index(drop=True)

    def _check_ins_(self):
        check_ins_matrix = np.zeros([self.user_num, self.item_num])

        for location_id, user_id in self.train[['location_id', 'user_id']].values:
            check_ins_matrix[user_id][location_id] += 1
        return check_ins_matrix


if __name__ == '__main__':
    # 1
    suffix = '_1'
    data_set = DataReader(data_path='/mnt/xk/experiment/dataset/checkins_2000%s.csv'% suffix)
    train = data_set.train
    valid = data_set.valid
    test = data_set.test
    # saving train, valid and test dataset
    train.to_csv('/mnt/xk/experiment/dataset/train_2000%s.csv'% suffix, index=False)
    valid.to_csv('/mnt/xk/experiment/dataset/valid_2000%s.csv'% suffix, index=False)
    test.to_csv('/mnt/xk/experiment/dataset/test_2000%s.csv'% suffix, index=False)

    print('[INFO] valid ground_truth gen')
    ground_truth = []
    print(train.user_id.max()+1)
    for i in range(train.user_id.max()+1):
        temp = valid[valid.user_id == i].location_id.tolist()
        ground_truth.append(temp)
    pickle.dump(ground_truth, open('/mnt/xk/experiment/dataset/valid_ground_truth_2000%s.pkl'% suffix, 'wb'))

    print('[INFO] test ground_truth gen')
    ground_truth = []
    for i in range(train.user_id.max()+1):
        temp = test[test.user_id == i].location_id.tolist()
        ground_truth.append(temp)
    pickle.dump(ground_truth, open('/mnt/xk/experiment/dataset/test_ground_truth_2000%s.pkl'% suffix, 'wb'))

    # 2
    suffix = '_2'
    data_set = DataReader(data_path='/mnt/xk/experiment/dataset/checkins_2000%s.csv'% suffix)
    train = data_set.train
    valid = data_set.valid
    test = data_set.test
    # saving train, valid and test dataset
    train.to_csv('/mnt/xk/experiment/dataset/train_2000%s.csv'% suffix, index=False)
    valid.to_csv('/mnt/xk/experiment/dataset/valid_2000%s.csv'% suffix, index=False)
    test.to_csv('/mnt/xk/experiment/dataset/test_2000%s.csv'% suffix, index=False)

    print('[INFO] valid ground_truth gen')
    ground_truth = []
    for i in range(train.user_id.max()+1):
        temp = valid[valid.user_id == i].location_id.tolist()
        ground_truth.append(temp)
    pickle.dump(ground_truth, open('/mnt/xk/experiment/dataset/valid_ground_truth_2000%s.pkl'% suffix, 'wb'))

    print('[INFO] test ground_truth gen')
    ground_truth = []
    for i in range(train.user_id.max()+1):
        temp = test[test.user_id == i].location_id.tolist()
        ground_truth.append(temp)
    pickle.dump(ground_truth, open('/mnt/xk/experiment/dataset/test_ground_truth_2000%s.pkl'% suffix, 'wb'))

    # # 3
    # suffix = '_3'
    # data_set = DataReader(data_path='/mnt/xk/experiment/dataset/checkins_2000_3%s.csv'% suffix)
    # train = data_set.train
    # valid = data_set.valid
    # test = data_set.test
    # # saving train, valid and test dataset
    # train.to_csv('/mnt/xk/experiment/dataset/train_2000%s.csv'% suffix, index=False)
    # valid.to_csv('/mnt/xk/experiment/dataset/valid_2000%s.csv'% suffix, index=False)
    # test.to_csv('/mnt/xk/experiment/dataset/test_2000%s.csv'% suffix, index=False)
    #
    # print('[INFO] valid ground_truth gen')
    # ground_truth = []
    # for i in range(train.user_id.max()+1):
    #     temp = valid[valid.user_id == i].location_id.tolist()
    #     ground_truth.append(temp)
    # pickle.dump(ground_truth, open('/mnt/xk/experiment/dataset/valid_ground_truth_2000%s.pkl'% suffix, 'wb'))
    #
    # print('[INFO] test ground_truth gen')
    # ground_truth = []
    # for i in range(train.user_id.max()+1):
    #     temp = test[test.user_id == i].location_id.tolist()
    #     ground_truth.append(temp)
    # pickle.dump(ground_truth, open('/mnt/xk/experiment/dataset/test_ground_truth_2000%s.pkl'% suffix, 'wb'))
