# -*- coding: utf-8 -*-
# @Time    : 2018/6/14  21:25
# @Author  : Dyn
import numpy as np


class Estimator(object):
    @staticmethod
    def precision_at_k(rec_list, ground_truth, k):
        assert (len(rec_list) == len(ground_truth))
        right = 0
        total = len(rec_list) * k
        for rec, g in zip(rec_list, ground_truth):
            right += len(set(rec[:k]) & set(g))
        return right / total

    @staticmethod
    def hit_rate_at_k(rec_list, ground_truth, k):
        assert (len(rec_list) == len(ground_truth))
        hit = 0
        total = 0
        for rec, g in zip(rec_list, ground_truth):
            hit += len(set(rec[:k]) & set(g))
            total += len(g)
        return hit / total

    @staticmethod
    def NDCG_at_k(rec_list, ground_truth, k):
        assert (len(rec_list) == len(ground_truth))
        n = len(rec_list)
        sums = 0
        for rec, g in zip(rec_list, ground_truth):
            gg = set(g)
            DCG = 0
            hit = 0
            for i in range(1, k+1):
                if rec[i-1] in gg:
                    rel = 1
                    hit += 1
                else:
                    rel = 0
                DCG += (pow(2, rel) - 1) / np.log2(i + 1)

            IDCG = 0
            for i in range(1, hit+1):
                IDCG += (pow(2, 1) - 1) / np.log2(i + 1)
            if IDCG != 0:
                NDCG = DCG / IDCG
            else:
                NDCG = 0
            sums += NDCG
        return sums / n

    @staticmethod
    def recall_at_k(rec_list, ground_truth, k):
        assert (len(rec_list) == len(ground_truth))
        right = 0
        for rec, g in zip(rec_list, ground_truth):
            right += len(set(rec[:k]) & set(g))
        total = 0
        for each in ground_truth:
            total += len(set(each))
        return right / total

    @staticmethod
    def coverage(rec_list, n):
        item_set = set()
        for items in rec_list:
            for item in items:
                item_set.add(item)
        return len(item_set) / (n * 1.0)


if __name__ == '__main__':
    print(Estimator.NDCG_at_k([[1, 2, 3, 4], [2, 1, 3, 4]], [[1], [1]], 2))
