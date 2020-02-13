# -*- coding: utf-8 -*-
# Time: 2018/11/27 20:40
# Author: Dyn
import tensorflow as tf


class LogLoss(object):
    def __init__(self, outputs, labels):
        self.name = 'log_loss'
        self.update = []
        self.losses = self._loss(outputs, labels)
        if tf.flags.FLAGS.total_loss_summary:
            self._epoch_total_loss()
            self._batch_total_loss()
        if tf.flags.FLAGS.main_loss_summary:
            self._epoch_main_loss()
            self._batch_main_loss()
        with tf.variable_scope('loss'):
            self._add_summary()

    def _loss(self, outputs, labels):
        self.main_loss = tf.losses.log_loss(labels, outputs)
        tf.add_to_collection('losses', self.main_loss)

        weights = [tf_var for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name.lower()
                                                                         or "bias" in tf_var.name.lower())]
        if tf.flags.FLAGS.l2_loss:
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=tf.flags.FLAGS.l2_rate)
            l2_loss = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
            tf.add_to_collection('losses', l2_loss)

        if tf.flags.FLAGS.l1_loss:
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=tf.flags.FLAGS.l1_rate)
            l1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            tf.add_to_collection('losses', l1_loss)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss_add')
        return loss

    def _get_loss(self, loss, name):
        tf_loss, tf_loss_update = tf.metrics.mean(loss, name=name + '_' + self.name)
        # local vars will include isolate variables behind scene (count and total)
        local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name + '_' + self.name)
        local_vars_initializer = tf.variables_initializer(var_list=local_vars)
        return tf_loss, tf_loss_update, local_vars_initializer

    def _epoch_total_loss(self):
        self.epoch_total_loss, epoch_tf_loss_update, epoch_local_vars_initializer = \
            self._get_loss(self.losses, 'epoch_total')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        self.update.append(epoch_tf_loss_update)

    def _batch_total_loss(self):
        self.batch_total_loss, batch_tf_loss_update, batch_local_vars_initializer = \
            self._get_loss(self.losses, 'batch_total')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        self.update.append(batch_tf_loss_update)

    def _epoch_main_loss(self):
        self.epoch_main_loss, epoch_tf_loss_update, epoch_local_vars_initializer = \
            self._get_loss(self.main_loss, 'epoch_main')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        self.update.append(epoch_tf_loss_update)

    def _batch_main_loss(self):
        self.batch_main_loss, batch_tf_loss_update, batch_local_vars_initializer = \
            self._get_loss(self.main_loss, 'batch_main')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        self.update.append(batch_tf_loss_update)

    def _add_summary(self):
        if tf.flags.FLAGS.total_loss_summary:
            summary = tf.summary.scalar('epoch_total_' + self.name, self.epoch_total_loss)
            tf.add_to_collection('epoch_summary', summary)
            tf.add_to_collection('loss_epoch_summary', summary)

            summary = tf.summary.scalar('batch_total_' + self.name, self.batch_total_loss)
            tf.add_to_collection('batch_summary', summary)
            tf.add_to_collection('loss_batch_summary', summary)

        if tf.flags.FLAGS.main_loss_summary:
            summary = tf.summary.scalar('epoch_main_' + self.name, self.epoch_main_loss)
            tf.add_to_collection('epoch_summary', summary)
            tf.add_to_collection('loss_epoch_summary', summary)

            summary = tf.summary.scalar('batch_main_' + self.name, self.batch_main_loss)
            tf.add_to_collection('batch_summary', summary)
            tf.add_to_collection('loss_batch_summary', summary)

    def loss(self):
        return tf.group(self.update)


class CrossEntropyLoss(object):
    def __init__(self, outputs, labels):
        self.name = 'cross_entropy_loss'
        self.update = []
        self.losses = self._loss(outputs, labels)
        if tf.flags.FLAGS.total_loss_summary:
            self._epoch_total_loss()
            self._batch_total_loss()
        if tf.flags.FLAGS.main_loss_summary:
            self._epoch_main_loss()
            self._batch_main_loss()
        with tf.variable_scope('loss'):
            self._add_summary()

    def _loss(self, outputs, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs,
                                                                         labels=labels,
                                                                         name=self.name))
        self.main_loss = loss
        tf.add_to_collection('losses', loss)

        weights = [tf_var for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name.lower()
                                                                         or "bias" in tf_var.name.lower())]
        if tf.flags.FLAGS.l2_loss:
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=tf.flags.FLAGS.l2_rate)
            l2_loss = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
            tf.add_to_collection('losses', l2_loss)

        if tf.flags.FLAGS.l1_loss:
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=tf.flags.FLAGS.l1_rate)
            l1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            tf.add_to_collection('losses', l1_loss)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss_add')
        return loss
    
    def _get_loss(self, loss, name):
        tf_loss, tf_loss_update = tf.metrics.mean(loss, name=name + '_' +self.name)
        # local vars will include isolate variables behind scene (count and total)
        local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name +'_'+self.name)
        local_vars_initializer = tf.variables_initializer(var_list=local_vars)
        return tf_loss, tf_loss_update, local_vars_initializer

    def _epoch_total_loss(self):
        self.epoch_total_loss, epoch_tf_loss_update, epoch_local_vars_initializer = \
        self._get_loss(self.losses, 'epoch_total')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        self.update.append(epoch_tf_loss_update)

    def _batch_total_loss(self):
        self.batch_total_loss, batch_tf_loss_update, batch_local_vars_initializer = \
        self._get_loss(self.losses, 'batch_total')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        self.update.append(batch_tf_loss_update)

    def _epoch_main_loss(self):
        self.epoch_main_loss, epoch_tf_loss_update, epoch_local_vars_initializer = \
        self._get_loss(self.main_loss, 'epoch_main')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        self.update.append(epoch_tf_loss_update)

    def _batch_main_loss(self):
        self.batch_main_loss, batch_tf_loss_update, batch_local_vars_initializer = \
        self._get_loss(self.main_loss, 'batch_main')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        self.update.append(batch_tf_loss_update)

    def _add_summary(self):
        if tf.flags.FLAGS.total_loss_summary:
            summary = tf.summary.scalar('epoch_total_' + self.name, self.epoch_total_loss)
            tf.add_to_collection('epoch_summary', summary)
            tf.add_to_collection('loss_epoch_summary', summary)

            summary = tf.summary.scalar('batch_total_' + self.name, self.batch_total_loss)
            tf.add_to_collection('batch_summary', summary)
            tf.add_to_collection('loss_batch_summary', summary)

        if tf.flags.FLAGS.main_loss_summary:
            summary = tf.summary.scalar('epoch_main_' + self.name, self.epoch_main_loss)
            tf.add_to_collection('epoch_summary', summary)
            tf.add_to_collection('loss_epoch_summary', summary)

            summary = tf.summary.scalar('batch_main_' + self.name, self.batch_main_loss)
            tf.add_to_collection('batch_summary', summary)
            tf.add_to_collection('loss_batch_summary', summary)

    def loss(self):
        return tf.group(self.update)


class LogSigmoidLoss(object):
    def __init__(self, p_predict, n_predict):
        self.name = 'log_sigmoid_loss'
        self.update = []
        self.losses = self._loss(p_predict, n_predict)
        if tf.flags.FLAGS.total_loss_summary:
            self._epoch_total_loss()
            self._batch_total_loss()
        if tf.flags.FLAGS.main_loss_summary:
            self._epoch_main_loss()
            self._batch_main_loss()
        with tf.variable_scope('loss'):
            self._add_summary()

    def _main_loss(self, p_predict, n_predict):
        return -tf.reduce_mean(tf.log(tf.sigmoid(p_predict - n_predict)), name=self.name)

    def _loss(self, p_predict, n_predict):
        loss = self._main_loss(p_predict, n_predict)
        self.main_loss = loss

        tf.add_to_collection('losses', loss)

        weights = [tf_var for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name.lower()
                                                                         or "bias" in tf_var.name.lower())]
        if tf.flags.FLAGS.l2_loss:
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=tf.flags.FLAGS.l2_rate)
            l2_loss = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
            tf.add_to_collection('losses', l2_loss)

        if tf.flags.FLAGS.l1_loss:
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=tf.flags.FLAGS.l1_rate)
            l1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            tf.add_to_collection('losses', l1_loss)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss_add')
        return loss

    def _get_loss(self, loss, name):
        tf_loss, tf_loss_update = tf.metrics.mean(loss, name=name + '_' + self.name)
        # local vars will include isolate variables behind scene (count and total)
        local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name + '_' + self.name)
        local_vars_initializer = tf.variables_initializer(var_list=local_vars)
        return tf_loss, tf_loss_update, local_vars_initializer

    def _epoch_total_loss(self):
        self.epoch_total_loss, epoch_tf_loss_update, epoch_local_vars_initializer = \
            self._get_loss(self.losses, 'epoch_total')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        self.update.append(epoch_tf_loss_update)

    def _batch_total_loss(self):
        self.batch_total_loss, batch_tf_loss_update, batch_local_vars_initializer = \
            self._get_loss(self.losses, 'batch_total')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        self.update.append(batch_tf_loss_update)

    def _epoch_main_loss(self):
        self.epoch_main_loss, epoch_tf_loss_update, epoch_local_vars_initializer = \
            self._get_loss(self.main_loss, 'epoch_main')
        tf.add_to_collection('epoch_initializer', epoch_local_vars_initializer)
        tf.add_to_collection('loss_epoch_initializer', epoch_local_vars_initializer)

        self.update.append(epoch_tf_loss_update)

    def _batch_main_loss(self):
        self.batch_main_loss, batch_tf_loss_update, batch_local_vars_initializer = \
            self._get_loss(self.main_loss, 'batch_main')
        tf.add_to_collection('batch_initializer', batch_local_vars_initializer)
        tf.add_to_collection('loss_batch_initializer', batch_local_vars_initializer)

        self.update.append(batch_tf_loss_update)

    def _add_summary(self):
        if tf.flags.FLAGS.total_loss_summary:
            summary = tf.summary.scalar('epoch_total_' + self.name, self.epoch_total_loss)
            tf.add_to_collection('epoch_summary', summary)
            tf.add_to_collection('loss_epoch_summary', summary)

            summary = tf.summary.scalar('batch_total_' + self.name, self.batch_total_loss)
            tf.add_to_collection('batch_summary', summary)
            tf.add_to_collection('loss_batch_summary', summary)

        if tf.flags.FLAGS.main_loss_summary:
            summary = tf.summary.scalar('epoch_main_' + self.name, self.epoch_main_loss)
            tf.add_to_collection('epoch_summary', summary)
            tf.add_to_collection('loss_epoch_summary', summary)

            summary = tf.summary.scalar('batch_main_' + self.name, self.batch_main_loss)
            tf.add_to_collection('batch_summary', summary)
            tf.add_to_collection('loss_batch_summary', summary)

    def loss(self):
        return tf.group(self.update)


def loss_function(outputs, labels, loss_type, name=None, loss_collection_name='losses'):
    if loss_type == 'cross_entropy':
        with tf.variable_scope('losses', reuse=tf.AUTO_REUSE):
            name = loss_type + '_loss' if name is None else name
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs,
                                                                             labels=labels,
                                                                             name=name))
            tf.add_to_collection(loss_collection_name, loss)
            if tf.flags.FLAGS.main_loss_summary:
                summary = tf.summary.scalar('main_loss', loss)
                tf.add_to_collection('batch_summary', summary)
            loss = merge_loss(loss_collection_name)
        return loss


def merge_loss(loss_collection_name='losses'):
    weights = [tf_var for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name.lower()
                                                                     or "bias" in tf_var.name.lower())]
    if tf.flags.FLAGS.l2_loss:
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=tf.flags.FLAGS.l2_rate)
        l2_loss = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
        tf.add_to_collection('losses', l2_loss)

    if tf.flags.FLAGS.l1_loss:
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=tf.flags.FLAGS.l1_rate)
        l1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        tf.add_to_collection('losses', l1_loss)

    loss = tf.add_n(tf.get_collection(loss_collection_name), name='total_loss_add')
    if tf.flags.FLAGS.summary:
        summary = tf.summary.scalar('total_loss', loss)
        tf.add_to_collection('batch_summary', summary)
    return loss


if __name__ == '__main__':
    place = tf.placeholder(tf.float32)
    tf.add_to_collection('losses', place)
    tf.get_default_graph().clear_collection('losses')
    print(tf.get_collection('losses'))
