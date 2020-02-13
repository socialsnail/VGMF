# -*- coding: utf-8 -*-
# @Time    : 2019/1/20  14:24
# @Author  : Dyn

import numpy as np
import pandas as pd
import pickle
from data.dataset import Dataset
from config.config import *


class VisualSimilarity(object):
    def __init__(self):
        self.user_features = pickle.load(open(USERS_PHOTOS_FEATURES_2CLASSES, mode='rb'))
        self.location_features = pickle.load(open(LOCATIONS_PHOTOS_FEATURES_2CLASSES, mode='rb'))

    def l2_dis(self):
        self.user_features['features'][self.user_features['cnt'] == 0] = 1.0
        self.location_features['features'][self.location_features['cnt'] == 0] = 1.0
        mask_matrix_row = self.user_features['cnt'] == 0
        mask_matrix_col = self.location_features['cnt'] == 0

        X = self.user_features['features']
        Y = self.location_features['features']

        # X2 = np.sum(np.square(X), axis=1)
        # X2 = np.repeat(X2, Y.shape[0], axis=1)
        # Y2 = np.transpose([np.sum(np.square(Y), axis=1)])
        # Y2 = np.repeat(Y2, X.shape[0], axis=0)
        # print(X2.shape)
        # print(Y2.shape)
        dists = np.sqrt(-2 * np.dot(X, Y.T) + np.sum(np.square(X), axis=1)[:, np.newaxis] +
                        np.sum(np.square(Y), axis=1).T[np.newaxis, :])

        user_features_norm2 = np.sqrt(np.sum(self.user_features['features'] ** 2, axis=1))
        location_features_norm2 = np.sqrt(np.sum(self.location_features['features'] ** 2, axis=1))
        user_features_norm2 = np.repeat(user_features_norm2[:, np.newaxis], location_features_norm2.shape[0], axis=1)
        location_features_norm2 = np.repeat(location_features_norm2[np.newaxis, :], user_features_norm2.shape[0],
                                            axis=0)

        norm = user_features_norm2 * location_features_norm2
        dists /= norm

        dists[mask_matrix_row == True, :] = 0
        dists[:, mask_matrix_col == True] = 0
        dists[mask_matrix_row == True, :] = np.mean(dists[mask_matrix_row == False, :])
        dists[:, mask_matrix_col == True] = np.mean(dists[:, mask_matrix_col == False])
        return dists

    def cosine_similarity(self):
        self.user_features['features'][self.user_features['cnt'] == 0] = 1.0
        self.location_features['features'][self.location_features['cnt'] == 0] = 1.0
        mask_matrix_row = self.user_features['cnt'] == 0
        mask_matrix_col = self.location_features['cnt'] == 0
        cosine_similarity = np.dot(self.user_features['features'], self.location_features['features'].T)
        user_features_norm2 = np.sqrt(np.sum(self.user_features['features'] ** 2, axis=1))
        location_features_norm2 = np.sqrt(np.sum(self.location_features['features'] ** 2, axis=1))
        user_features_norm2 = np.repeat(user_features_norm2[:, np.newaxis], location_features_norm2.shape[0], axis=1)
        location_features_norm2 = np.repeat(location_features_norm2[np.newaxis, :], user_features_norm2.shape[0],
                                            axis=0)
        norm = user_features_norm2 * location_features_norm2
        cosine_similarity /= norm
        cosine_similarity = np.max(cosine_similarity, axis=0) - cosine_similarity
        cosine_similarity /= np.max(cosine_similarity, axis=0) - np.min(cosine_similarity, axis=0)

        cosine_similarity[mask_matrix_row == True, :] = 0
        cosine_similarity[:, mask_matrix_col == True] = 0
        cosine_similarity[mask_matrix_row == True, :] = np.mean(cosine_similarity[mask_matrix_row == False, :])
        cosine_similarity[:, mask_matrix_col == True] = np.mean(cosine_similarity[:, mask_matrix_col == False])
        return cosine_similarity


if __name__ == '__main__':
    vs = VisualSimilarity()
    dataset = Dataset()
    mask_matrix_row = vs.user_features['cnt'] == 1
    mask_matrix_col = vs.location_features['cnt'] == 0

    dists = vs.cosine_similarity()

    pickle.dump(dists, open(VISUAL_WEIGHTS, mode='wb'))

    print(dists.shape)
    print(dists)

    # print(np.sum(cos))
    # print(cos.shape)
    # print(cos[dataset.check_ins_matrix==1])
