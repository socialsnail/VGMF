# -*- coding: utf-8 -*-
# @Time    : 2018/9/19  10:55
# @Author  : Dyn

from multiprocessing import *
import numpy as np


class ThreadProcess(object):
    def __init__(self, thread_num=cpu_count()):
        """Multi thread function class
        Args:
            thread_num(int), numbers of thread to run function in parallel
        """
        self.thread_num = thread_num if thread_num <= cpu_count() else cpu_count()
        self.pool = Pool(self.thread_num)

    def run(self, f, inputs, split_parts=None):
        """Running function with inputs in parallel
        Args:
            f(function): take inputs and call function in parallel
            inputs(list or numpy.ndarray): dataset feeds to f
            split_parts(int or None): numbers of part to split dataset
        """
        if split_parts is None:
            split_parts = self.thread_num
        r = self.pool.map(f, np.array_split(inputs,split_parts) )
        return r

    @staticmethod
    def contact(results, axis=0):
        """Contact results in given axis
        Args:
            axis(int):axis to contact dataset
        """
        return np.concatenate(results, axis=axis)

    def close(self):
        """Close thread pool
        """
        self.pool.close()
        self.pool.join()


if __name__ == '__main__':
    class A(object):
        def __init__(self):
            self.x = [1, 2, 3, 4, 5]
            self.matrix = np.zeros(shape=[20, 20])

        def r(self, x):
            self.matrix[x[0], x[1]] = x[0]
            return [[x[0], x[1], x[0]]]

    a = A()
    tp = ThreadProcess(10)
    re = tp.run(a.r, [i for i in range(20)], 10)
    re = np.concatenate(re)
    print(re)