# -*- coding: utf-8 -*-
# Time: 2018/11/27 15:43
# Author: Dyn

import tensorflow as tf


def optimizer():
    if tf.app.flags.FLAGS.optimizer == 'Momentum':
        return tf.train.MomentumOptimizer(learning_rate=tf.app.flags.FLAGS.learning_rate,
                                          momentum=0.9)
    elif tf.app.flags.FLAGS.optimizer == 'Adam':
        return tf.train.AdamOptimizer(learning_rate=tf.app.flags.FLAGS.learning_rate)
    elif tf.app.flags.FLAGS.optimizer == 'RMSProp':
        return tf.train.RMSPropOptimizer(learning_rate=tf.app.flags.FLAGS.learning_rate)
    elif tf.app.flags.FLAGS.optimizer == 'Adagrad':
        return tf.train.AdagradOptimizer(learning_rate=tf.app.flags.FLAGS.learning_rate)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate=tf.app.flags.FLAGS.learning_rate)

if __name__ == '__main__':
    pass