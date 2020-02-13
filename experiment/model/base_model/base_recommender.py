# -*- coding: utf-8 -*-
# @Time    : 2018/6/14  14:55
# @Author  : Dyn
from config.config import *


class BaseRecommendationModel(object):
    def __init__(self):
        # dataset
        self.dataset = None
        # total user and item num
        self.user_num = None
        self.item_num = None

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

    def build(self):
        return None

    def recommend(self, k):
        return None

if __name__ == '__main__':
    pass
