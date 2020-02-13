# -*- coding: utf-8 -*-
# @Time    : 2018/9/17  10:14
# @Author  : Dyn

from heapq import heapify, heappop, heappush, heapreplace
import numpy as np


class EarlyStopping(object):
    def __init__(self, k):
        """Early stop training process if model loss didn't decease in k epoch
        Args:
            k(int): number of epoch to stop training process
        """
        self.k = k
        self.loss_hp = []
        heapify(self.loss_hp)
        self.min_loss = np.inf

    def loss_compare(self, loss):
        """Return if the loss is the minimum
        Args:
            loss(float): training loss
        """
        if loss < self.min_loss:
            return True
        else:
            return False

    def log(self, loss):
        """Log loss, use max heap to log least k loss,
        and end the training process if loss larger than maximum of the k least loss
        Args:
             loss(float): training loss
        """
        self.min_loss = min(self.min_loss, loss)
        # if heap element less than k, push element
        if len(self.loss_hp) < self.k:
            heappush(self.loss_hp, -loss)
        # update the heap if loss less than the max element of the heap
        elif loss < -self.loss_hp[0]:
            heappop(self.loss_hp)
            heappush(self.loss_hp, -loss)
        else:
            return False
        return True


if __name__ == '__main__':
    es = EarlyStopping(3)
    es.log(1)
    es.log(5)
    es.log(2)
    print(es.loss_hp)
    es.log(3)
    print(es.loss_hp)

