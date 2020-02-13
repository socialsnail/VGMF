# -*- coding: utf-8 -*-
# Time: 2018/11/27 16:19
# Author: Dyn
import os
import tensorflow as tf


FLAGS = tf.flags.FLAGS


class Summary(object):
    def __init__(self):
        self.train_cnt = 0
        self.valid_cnt = 0
        self.index_in_num_epoch = 0

    @staticmethod
    def clear_dir(path):
        path_list = path.split(os.sep)[1:]
        path_pre = os.sep
        for d in path_list[:]:
            path_pre = os.path.join(path_pre, d)
            if not os.path.exists(path_pre):
                os.mkdir(path_pre)

        if os.path.exists(path):
            del_dir = [x for x in os.listdir(path)]
            for i in del_dir:
                print(os.sep.join([path, i]))
                os.remove(os.sep.join([path, i]))

    def init_op(self, sess):
        self.clear_dir(FLAGS.train_summary_path)
        self.clear_dir(FLAGS.valid_summary_path)
        self._train_writer = tf.summary.FileWriter(FLAGS.train_summary_path, sess.graph)
        self._valid_writer = tf.summary.FileWriter(FLAGS.valid_summary_path, sess.graph)

    def log(self, summary, train_or_valid='train', epoch_wise=False):
        if train_or_valid == 'train':
            if epoch_wise:
                self._train_writer.add_summary(summary, self.index_in_num_epoch)
            else:
                self.train_cnt += 1
                self._train_writer.add_summary(summary, self.train_cnt)
        else:
            self._valid_writer.add_summary(summary, self.index_in_num_epoch)
            self.index_in_num_epoch += 1


def variable_summaries(var, name):
    s_var = tf.cast(var, tf.float32)

    mean = tf.reduce_mean(s_var)
    tf.summary.scalar('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_sum(tf.square(s_var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)

    tf.summary.histogram(name, var)


def visualize_grads_and_vars(optimizer, loss):
    with tf.variable_scope('training_op', reuse=tf.AUTO_REUSE):
        if FLAGS.gradients_summary:
            grads = optimizer.compute_gradients(loss)
            for grad, var in grads:
                if grad is not None:
                    summary = tf.summary.histogram(var.op.name + '/gradients', grad)
                    tf.add_to_collection('grads_and_vars', summary)

        if FLAGS.variables_summary:
            for var in tf.trainable_variables():
                summary = tf.summary.histogram(var.name, var)
                tf.add_to_collection('grads_and_vars', summary)


if __name__ == '__main__':
    pass