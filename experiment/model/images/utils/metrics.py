# -*- coding: utf-8 -*-
# Time: 2018/11/27 20:45
# Author: Dyn

import tensorflow as tf


class Accuracy(object):
    def __init__(self, outputs, labels):
        self.name = 'accuracy'
        self.update = []
        self._epoch_metric(outputs, labels)
        self._batch_metric(outputs, labels)
        with tf.variable_scope('metric'):
            self._add_summary()

    def _get_metric(self, outputs, labels, name):
        tf_metric, tf_metric_update = tf.metrics.accuracy(tf.argmax(outputs, 1),
                                                          tf.argmax(labels, 1),
                                                          name=name +'_'+self.name)
        # local vars will include isolate variables behind scene (count and total)
        local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name +'_'+self.name)
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