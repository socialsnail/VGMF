# -*- coding: utf-8 -*-
# @Time    : 2018/12/11  12:36
# @Author  : Dyn

import numpy as np
import bisect
from tqdm import tqdm
from model.base_model.base_recommender import BaseRecommendationModel
from config.config import DATA_ROOT
import pickle
import os


class GeoGrid(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.id2pos_dict = None
        self.pos2id_dict = None

    def grid(self, x_bins, y_bins):
        locations = self.dataset.train[['location_id', 'longitude', 'latitude']].drop_duplicates(subset='location_id')
        x = locations['longitude']
        y = locations['latitude']
        max_x = max(x)
        min_x = min(x)
        max_y = max(y)
        min_y = min(y)
        x_ticks = np.arange(min_x, max_x + x_bins, x_bins)
        y_ticks = np.arange(min_y, max_y + y_bins, y_bins)
        id2pos_dict = {}
        pos2id_dict = {}
        def get_pos(a, b, x, y):
            return bisect.bisect_left(a, x), bisect.bisect_left(b, y)
        for index, row in locations.iterrows():
            location_id = row['location_id']
            x = row['longitude']
            y = row['latitude']
            x_pos, y_pos = get_pos(x_ticks, y_ticks, x, y)
            id2pos_dict[location_id] = (x_pos, y_pos)
            if (x_pos, y_pos) not in pos2id_dict:
                pos2id_dict[(x_pos, y_pos)] = set()
            pos2id_dict[(x_pos, y_pos)].add(location_id)
        self.min_x = min_x
        self.max_x = max_x
        self.max_y = max_y
        self.min_y = min_y
        self.x_limit = len(x_ticks) - 1
        self.y_limit = len(y_ticks) - 1
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.id2pos_dict = id2pos_dict
        self.pos2id_dict = pos2id_dict

    def location2pos(self, location_id):
        x, y = self.id2pos_dict[location_id]
        x_mean = self.min_x + (x + 0.5) * self.x_bins
        y_mean = self.min_y + (y + 0.5) * self.y_bins
        return x_mean, y_mean

    def user_history_gen(self):
        group_by_users = self.dataset.train.groupby('user_id')['location_id'].unique()
        user_history_grid = {}
        for index, locations in zip(group_by_users.index, group_by_users):
            user_history_grid[index] = {}
            user_history_grid[index]['total'] = len(locations)
            for location in locations:
                x, y = self.id2pos_dict[location]
                if (x, y) not in user_history_grid[index]:
                    user_history_grid[index][(x, y)] = 0
                user_history_grid[index][(x, y)] += 1
        self.user_history_grid = user_history_grid

    def check_pos(self, x, y):
        if x < 0 or y < 0 or x > self.x_limit or y > self.y_limit:
            return False
        else:
            return True

    def near_by_position(self, x, y, n):
        near_pos = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                if abs(i) >= n or abs(j) >= n:
                    if self.check_pos(x + i, y + j):
                        near_pos.append((x + i, y + j))
        return near_pos

    def nearby_gen(self, n=2):
        nearby_cnt = {}
        for user_id in tqdm(range(self.dataset.user_num)):
            nearby_cnt[user_id] = {}
            for line in self.user_history_grid[user_id]:
                if line != 'total':
                    x = line[0]
                    y = line[1]
                    for i in range(n):
                        nearby_pos = self.near_by_position(x, y, i)
                        for p in nearby_pos:
                            if p not in nearby_cnt[user_id]:
                                nearby_cnt[user_id][p] = [0] * n
                            nearby_cnt[user_id][p][i] += self.user_history_grid[user_id][line]
        self.user_near_by_history_grid = nearby_cnt

    def visited_num_gen(self, alpha):
        matrix = np.zeros(shape=[self.dataset.user_num, self.dataset.item_num])
        for user_id in tqdm(range(self.dataset.user_num)):
            for location_id in range(self.dataset.item_num):
                x, y = self.id2pos_dict[location_id]
                cnt_list = self.user_near_by_history_grid[user_id].get((x, y), [0] * len(alpha))
                if self.dataset.check_ins_matrix[user_id, location_id]:
                    cnt_list[0] -= 1
                p = 0
                for i, j in zip(cnt_list, alpha):
                    p += i * j
                matrix[user_id, location_id] = p
        return matrix


if __name__ == '__main__':
    from data.dataset import Dataset
    from evaluate.estimator import Estimator
    import pickle
    from config.config import *

    dataset = Dataset(geo=True, sparse=True)
    geo_grid = GeoGrid(dataset)
    geo_grid.grid(0.005, 0.005)
    geo_grid.user_history_gen()
    geo_grid.nearby_gen(n=2)
    matrix = geo_grid.visited_num_gen([2, 0.002])
    pickle.dump(matrix, open(GRID, mode='wb'))
    matrix[geo_grid.dataset.check_ins_matrix != 0] = -np.inf
    rec_list = np.argsort(-matrix)[:, :30]
    ground_truth = geo_grid.dataset.valid_ground_truth
    precisions = []
    recalls = []
    for i in range(0, 31, 5):
        if i == 0:
            k = i + 1
        else:
            k = i
        precision = Estimator.precision_at_k(rec_list, ground_truth, k)
        recall = Estimator.recall_at_k(rec_list, ground_truth, k)
        precisions.append(round(precision, 5))
        recalls.append(round(recall, 5))
    print('[INFO] precision {0}'.format(precisions))
    print('[INFO] recall {0}'.format(recalls))

    matrix = pickle.load(open(GRID, mode='rb'))
    print(np.shape(matrix))
    print(type(matrix))