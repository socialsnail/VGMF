# -*- coding: utf-8 -*-
# @Time    : 2019/4/4  10:24
# @Author  : Dyn
import numpy as np
import bisect
from tqdm import tqdm
from config.config import DATA_ROOT
import pickle
import os
from sklearn.cluster import KMeans


class KMeansGeoGrid(object):
    def __init__(self, dataset, beta=1.0, k=10):
        self.dataset = dataset
        self.location_id2pos_dict = None
        self.pos2location_id_dict = None
        self.beta = beta
        self.k = k
        self.cluster = KMeans(n_clusters=k, random_state=37)
        self._grid()

    def _grid(self):
        locations = self.dataset.train[['location_id', 'longitude', 'latitude']].drop_duplicates(subset='location_id')

        self.cluster.fit(locations[['longitude', 'latitude']].values)
        classes_index = self.cluster.predict(locations[['longitude', 'latitude']].values)
        location_id2pos_dict = {}
        pos2location_id_dict = {}
        for location_id, class_index in zip(locations['location_id'], classes_index):
            location_id2pos_dict[location_id] = class_index
            pos2location_id_dict[class_index] = location_id

        self.location_id2pos_dict = location_id2pos_dict
        self.pos2location_id_dict = pos2location_id_dict

        all_grid = []
        for key in pos2location_id_dict:
            all_grid.append(key)

        num_locations = len(location_id2pos_dict.keys())
        self.poi_influence_area = np.zeros(shape=[num_locations, self.k])
        self.poi_influence_area_gen()
        self.poi_influence_area = np.power(self.poi_influence_area, self.beta)
        mins = np.min(self.poi_influence_area, axis=1)
        maxs = np.max(self.poi_influence_area, axis=1)
        diff = maxs - mins
        self.poi_influence_area = (self.poi_influence_area - mins[:, np.newaxis]) / diff[:, np.newaxis]
        e = np.sum(np.exp(self.poi_influence_area), axis=1)
        self.poi_influence_area /= e[:, np.newaxis]
        self.num_grid = self.k

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
                    grid2 = self.location_id2pos_dict[l2]
                    poi_pair[grid1, grid2] += 1
                    poi_pair[grid2, grid1] += 1

        for location in range(self.poi_influence_area.shape[0]):
            for grid_id in range(self.poi_influence_area.shape[1]):
                location_pos = self.location_id2pos_dict[location]
                self.poi_influence_area[location, grid_id] = poi_pair[location_pos, grid_id]


if __name__ == '__main__':
    from data.dataset import Dataset
    from evaluate.estimator import Estimator
    import pickle
    from config.config import *

    dataset = Dataset(geo=True, sparse=True)
    geo_grid = KMeansGeoGrid(dataset, 0.95, k=100)
    print(geo_grid.poi_influence_area.shape)
    print(geo_grid.poi_influence_area)
    # print(np.sum(np.abs(np.array([[1, 2], [3, 4], [5, 6]]) - np.array([[5, 6]])), axis=1))
