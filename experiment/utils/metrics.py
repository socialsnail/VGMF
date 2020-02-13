# -*- coding: utf-8 -*-
# Time: 2018/11/27 20:45
# Author: Dyn

import tensorflow as tf
import numpy as np


class PrecisionAtK(object):
    def __init__(self, outputs, labels, k, mode='epoch'):
        self.name = 'precision_at_%d' % k
        self.k = k
        self.update = []
        self.mode = mode

        if mode == 'epoch' or mode == 'both':
            self._epoch_metric(outputs, labels)
        elif mode == 'batch' or mode == 'both':
            self._batch_metric(outputs, labels)

        with tf.variable_scope('metric'):
            self._add_summary()

    def _get_metric(self, outputs, labels, name):
        tf_metric, tf_metric_update = tf.metrics.precision_at_k(labels, outputs, self.k)
        # local vars will include isolate variables behind scene (count and total)
        local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name + '_' + self.name)
        local_vars_initializer = tf.variables_initializer(var_list=local_vars)
        return tf_metric, tf_metric_update, local_vars_initializer

    def _epoch_metric(self, outputs, labels):
        self.epoch_tf_metric, epoch_tf_metric_update, epoch_local_vars_initializer = \
            self._get_metric(outputs, labels, 'epoch')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        tf.add_to_collection('metric_epoch_initializer', epoch_local_vars_initializer)
        self.update.append(epoch_tf_metric_update)

    def _batch_metric(self, outputs, labels):
        self.batch_tf_metric, batch_tf_metric_update, batch_local_vars_initializer = \
            self._get_metric(outputs, labels, 'batch')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        tf.add_to_collection('metric_batch_initializer', batch_local_vars_initializer)

        self.update.append(batch_tf_metric_update)

    def _add_summary(self):
        if tf.flags.FLAGS.metric_summary:
            if self.mode == 'epoch' or self.mode == 'both':
                summary = tf.summary.scalar('epoch_' + self.name, self.epoch_tf_metric)
                tf.add_to_collection('epoch_summary', summary)
                tf.add_to_collection('metric_epoch_summary', summary)

            elif self.mode == 'batch' or self.mode == 'both':
                summary = tf.summary.scalar('batch_' + self.name, self.batch_tf_metric)
                tf.add_to_collection('batch_summary', summary)
                tf.add_to_collection('metric_batch_summary', summary)

    def metric(self):
        return tf.group(self.update)


class HitRateAtK(object):
    def __init__(self, values, k, name=None, mode='epoch'):
        self.name = 'hit_rate_at_%d' % k if name is None else name
        self.update = []
        self.mode = mode
        m = tf.gather(values, list(range(k)), axis=-1)
        hit = tf.reduce_sum(m, axis=-1)
        self._epoch_metric(hit)
        self._batch_metric(hit)
        self._add_summary()

    def _get_metric(self, values, name):
        tf_metric, tf_metric_update = tf.metrics.mean(values)
        # local vars will include isolate variables behind scene (count and total)
        local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name + '_' + self.name)
        local_vars_initializer = tf.variables_initializer(var_list=local_vars)
        return tf_metric, tf_metric_update, local_vars_initializer

    def _epoch_metric(self, values):
        self.epoch_tf_metric, epoch_tf_metric_update, epoch_local_vars_initializer = \
            self._get_metric(values, 'epoch')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        self.update.append(epoch_tf_metric_update)

    def _batch_metric(self, values):
        self.batch_tf_metric, batch_tf_metric_update, batch_local_vars_initializer = \
            self._get_metric(values, 'batch')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        self.update.append(batch_tf_metric_update)

    def _add_summary(self):
        if tf.flags.FLAGS.metric_summary:
            if self.mode == 'epoch' or self.mode == 'both':
                summary = tf.summary.scalar('epoch_' + self.name, self.epoch_tf_metric)
                tf.add_to_collection('epoch_summary', summary)
                tf.add_to_collection('metric_epoch_summary', summary)

            elif self.mode == 'batch' or self.mode == 'both':
                summary = tf.summary.scalar('batch_' + self.name, self.batch_tf_metric)
                tf.add_to_collection('batch_summary', summary)
                tf.add_to_collection('metric_batch_summary', summary)

    def metric(self):
        return tf.group(self.update)


class NDCGAtK(object):
    def __init__(self, values, k, name=None, mode='epoch'):
        self.name = 'NDCG_at_%d' % k if name is None else name
        self.update = []
        self.mode = mode
        weights = [1.0 / np.log2(i + 1) for i in range(1, k + 1)]
        values = tf.gather(values, list(range(k)), axis=-1)
        values = tf.pow(2.0, tf.cast(values, tf.float32)) - 1
        DCG = tf.reduce_sum(values * weights, axis=-1)
        sorted_predict_values, sorted_predict_index = tf.math.top_k(values, k)
        IDCG = tf.reduce_sum(sorted_predict_values * weights, axis=-1)
        IDCG = tf.where(tf.equal(IDCG, tf.constant(0.0)), tf.ones_like(IDCG), IDCG)
        NDCG = DCG / IDCG
        self._epoch_metric(NDCG)
        self._batch_metric(NDCG)
        self._add_summary()

    def _get_metric(self, values, name):
        tf_metric, tf_metric_update = tf.metrics.mean(values)
        # local vars will include isolate variables behind scene (count and total)
        local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name + '_' + self.name)
        local_vars_initializer = tf.variables_initializer(var_list=local_vars)
        return tf_metric, tf_metric_update, local_vars_initializer

    def _epoch_metric(self, values):
        self.epoch_tf_metric, epoch_tf_metric_update, epoch_local_vars_initializer = \
            self._get_metric(values, 'epoch')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        self.update.append(epoch_tf_metric_update)

    def _batch_metric(self, values):
        self.batch_tf_metric, batch_tf_metric_update, batch_local_vars_initializer = \
            self._get_metric(values, 'batch')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        self.update.append(batch_tf_metric_update)

    def _add_summary(self):
        if tf.flags.FLAGS.metric_summary:
            if self.mode == 'epoch' or self.mode == 'both':
                summary = tf.summary.scalar('epoch_' + self.name, self.epoch_tf_metric)
                tf.add_to_collection('epoch_summary', summary)
                tf.add_to_collection('metric_epoch_summary', summary)

            elif self.mode == 'batch' or self.mode == 'both':
                summary = tf.summary.scalar('batch_' + self.name, self.batch_tf_metric)
                tf.add_to_collection('batch_summary', summary)
                tf.add_to_collection('metric_batch_summary', summary)

    def metric(self):
        return tf.group(self.update)


class Mean(object):
    def __init__(self, values, name=None, mode='epoch'):
        self.name = 'mean' if name is None else name
        self.update = []
        self.mode = mode
        self._epoch_metric(values)
        self._batch_metric(values)
        self._add_summary()

    def _get_metric(self, values, name):
        tf_metric, tf_metric_update = tf.metrics.mean(values)
        # local vars will include isolate variables behind scene (count and total)
        local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name + '_' + self.name)
        local_vars_initializer = tf.variables_initializer(var_list=local_vars)
        return tf_metric, tf_metric_update, local_vars_initializer

    def _epoch_metric(self, values):
        self.epoch_tf_metric, epoch_tf_metric_update, epoch_local_vars_initializer = \
            self._get_metric(values, 'epoch')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        self.update.append(epoch_tf_metric_update)

    def _batch_metric(self, values):
        self.batch_tf_metric, batch_tf_metric_update, batch_local_vars_initializer = \
            self._get_metric(values, 'batch')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        self.update.append(batch_tf_metric_update)

    def _add_summary(self):
        if tf.flags.FLAGS.metric_summary:
            if self.mode == 'epoch' or self.mode == 'both':
                summary = tf.summary.scalar('epoch_' + self.name, self.epoch_tf_metric)
                tf.add_to_collection('epoch_summary', summary)
                tf.add_to_collection('metric_epoch_summary', summary)

            elif self.mode == 'batch' or self.mode == 'both':
                summary = tf.summary.scalar('batch_' + self.name, self.batch_tf_metric)
                tf.add_to_collection('batch_summary', summary)
                tf.add_to_collection('metric_batch_summary', summary)

    def metric(self):
        return tf.group(self.update)


class Accuracy(object):
    def __init__(self, outputs, labels, name=None):
        self.name = 'accuracy' if name is None else name
        self.update = []
        self._epoch_metric(outputs, labels)
        self._batch_metric(outputs, labels)
        with tf.variable_scope('metric'):
            self._add_summary()

    def _get_metric(self, outputs, labels, name):
        tf_metric, tf_metric_update = tf.metrics.accuracy(tf.argmax(outputs, 1),
                                                          tf.argmax(labels, 1),
                                                          name=name + '_' + self.name)
        # local vars will include isolate variables behind scene (count and total)
        local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name + '_' + self.name)
        local_vars_initializer = tf.variables_initializer(var_list=local_vars)
        return tf_metric, tf_metric_update, local_vars_initializer

    def _epoch_metric(self, outputs, labels):
        self.epoch_tf_metric, epoch_tf_metric_update, epoch_local_vars_initializer = \
            self._get_metric(outputs, labels, 'epoch')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        self.update.append(epoch_tf_metric_update)

    def _batch_metric(self, outputs, labels):
        self.batch_tf_metric, batch_tf_metric_update, batch_local_vars_initializer = \
            self._get_metric(outputs, labels, 'batch')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        self.update.append(batch_tf_metric_update)

    def _add_summary(self):
        if tf.flags.FLAGS.metric_summary:
            summary = tf.summary.scalar('epoch_' + self.name, self.epoch_tf_metric)
            tf.add_to_collection('epoch_summary', summary)
            summary = tf.summary.scalar('batch_' + self.name, self.batch_tf_metric)
            tf.add_to_collection('batch_summary', summary)

    def metric(self):
        return tf.group(self.update)


if __name__ == '__main__':
    pass
