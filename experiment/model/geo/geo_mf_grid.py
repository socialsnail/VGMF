# -*- coding: utf-8 -*-
# @Time    : 2019/3/4  10:40
# @Author  : Dyn

import numpy as np
import bisect
from tqdm import tqdm
from config.config import DATA_ROOT
import pickle
import os


class GeoGrid(object):
    def __init__(self, dataset, beta=1.0):
        self.dataset = dataset
        self.location_id2pos_dict = None
        self.pos2location_id_dict = None
        self.beta = beta

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
        location_id2pos_dict = {}
        pos2location_id_dict = {}
        def get_pos(a, b, x, y):
            return bisect.bisect_left(a, x), bisect.bisect_left(b, y)
        for index, row in locations.iterrows():
            location_id = row['location_id']
            x = row['longitude']
            y = row['latitude']
            x_pos, y_pos = get_pos(x_ticks, y_ticks, x, y)
            location_id2pos_dict[location_id] = (x_pos, y_pos)
            if (x_pos, y_pos) not in pos2location_id_dict:
                pos2location_id_dict[(x_pos, y_pos)] = set()
            pos2location_id_dict[(x_pos, y_pos)].add(location_id)
        self.min_x = min_x
        self.max_x = max_x
        self.max_y = max_y
        self.min_y = min_y
        self.x_limit = len(x_ticks) - 1
        self.y_limit = len(y_ticks) - 1
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.location_id2pos_dict = location_id2pos_dict
        self.pos2location_id_dict = pos2location_id_dict

        all_grid = []
        for key in pos2location_id_dict:
            all_grid.append(key)

        self.pos2grid_id = {}
        index = 0
        for key in pos2location_id_dict:
            self.pos2grid_id[key] = index
            index += 1
        all_grid = np.array(all_grid)

        num_locations = len(location_id2pos_dict.keys())
        num_grid = all_grid.shape[0]
        self.poi_influence_area = np.zeros(shape=[num_locations, num_grid])
        self.poi_influence_area_gen()
        self.poi_influence_area = np.power(self.poi_influence_area, self.beta)
        mins = np.min(self.poi_influence_area, axis=1)
        maxs = np.max(self.poi_influence_area, axis=1)
        diff = maxs - mins
        self.poi_influence_area = (self.poi_influence_area - mins[:, np.newaxis]) / diff[:, np.newaxis]
        e = np.sum(np.exp(self.poi_influence_area), axis=1)
        self.poi_influence_area /= e[:, np.newaxis]
        self.num_grid = num_grid

    def pos2grid_id(self, x, y):
        return x * (self.x_limit + 1) + y

    def location2pos(self, location_id):
        x, y = self.location_id2pos_dict[location_id]
        x_mean = self.min_x + (x + 0.5) * self.x_bins
        y_mean = self.min_y + (y + 0.5) * self.y_bins
        return x_mean, y_mean

    def poi_influence_area_gen(self):
        group_by_users = self.dataset.train.groupby('user_id')['location_id'].unique()
        print('[INFO] gen influence area')
        poi_pair = np.zeros(shape=[self.poi_influence_area.shape[1], self.poi_influence_area.shape[1]])
        for index, locations in zip(group_by_users.index, group_by_users):
            for i in range(len(locations)):
                for j in range(i+1, len(locations)):
                    l1 = locations[i]
                    l2 = locations[j]
                    grid1 = self.location_id2pos_dict[l1]
                    grid1 = self.pos2grid_id[grid1]
                    grid2 = self.location_id2pos_dict[l2]
                    grid2 = self.pos2grid_id[grid2]
                    poi_pair[grid1, grid2] += 1
                    poi_pair[grid2, grid1] += 1

        for location in range(self.poi_influence_area.shape[0]):
            for grid_id in range(self.poi_influence_area.shape[1]):
                location_pos = self.location_id2pos_dict[location]
                location_grid_id = self.pos2grid_id[location_pos]
                self.poi_influence_area[location, grid_id] = poi_pair[location_grid_id, grid_id]

    def user_history_gen(self):
        group_by_users = self.dataset.train.groupby('user_id')['location_id'].unique()
        user_history_grid = {}
        for index, locations in zip(group_by_users.index, group_by_users):
            user_history_grid[index] = {}
            user_history_grid[index]['total'] = len(locations)
            for location in locations:
                x, y = self.location_id2pos_dict[location]
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
                x, y = self.location_id2pos_dict[location_id]
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
    # print(np.sum(np.abs(np.array([[1, 2], [3, 4], [5, 6]]) - np.array([[5, 6]])), axis=1))