# -*- coding: utf-8 -*-
# @Time    : 2018/6/17  22:08
# @Author  : Dyn


if __name__ == '__main__':
    import os
    import pickle

    from config.config import *
    from data.time_data_gen import DataReader

    data_set = DataReader(data_path=DATA_PATH, need_restart=True, filter_num=20)
    train = data_set.train
    valid = data_set.valid
    test = data_set.test

    # saving train, valid and test dataset
    train.to_csv(os.sep.join([DATA_ROOT, 'train_20.csv']), index=False)
    valid.to_csv(os.sep.join([DATA_ROOT, 'valid_20.csv']), index=False)
    test.to_csv(os.sep.join([DATA_ROOT, 'test_20.csv']), index=False)

    # train = pd.read_csv(os.sep.join([DATA_ROOT, 'train_20.csv']))
    # valid = pd.read_csv(os.sep.join([DATA_ROOT, 'valid_20.csv']))
    # test = pd.read_csv(os.sep.join([DATA_ROOT, 'test_20.csv']))

    user_id = set(pickle.load(open(os.sep.join([CONFIG, 'user_id2.pkl']), mode='rb')))

    print('[INFO] valid ground_truth gen')
    ground_truth = []
    for i in range(len(user_id)):
        temp = valid[valid.user_id == i].location_id.tolist()
        ground_truth.append(temp)
    print(len(ground_truth))
    pickle.dump(ground_truth, open(os.sep.join([DATA_ROOT, 'valid_ground_truth_20.pkl']), 'wb'))

    print('[INFO] test ground_truth gen')
    ground_truth = []
    for i in range(len(user_id)):
        temp = test[test.user_id == i].location_id.tolist()
        ground_truth.append(temp)
    pickle.dump(ground_truth, open(os.sep.join([DATA_ROOT, 'test_ground_truth_20.pkl']), 'wb'))
