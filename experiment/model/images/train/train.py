# -*- coding: utf-8 -*-
# @Time    : 2018/12/10  13:52
# @Author  : Dyn


if __name__ == '__main__':
    import os
    import tensorflow as tf
    from model.images.arc.vgg import VGG16
    from model.images.data.dataset import ClassificationDataset
    from config.config import *
    import warnings
    from model.images.utils import Optimizer, Summary, Accuracy, visualize_grads_and_vars, CrossEntropyLoss
    from model.images.utils.bars import Bar
    warnings.filterwarnings('ignore')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    flag = tf.app.flags

    # dataset
    flag.DEFINE_integer('num_classes', 2, 'num classes of place')
    flag.DEFINE_integer('seed', 253, 'random seeds')
    flag.DEFINE_string('train_path', CATEGORIES_CLASSIFICATION_TRAIN2_FILE, 'path to save train file')
    flag.DEFINE_string('valid_path', CATEGORIES_CLASSIFICATION_VALID2_FILE, 'path to save valid file')
    flag.DEFINE_integer('num_cpus', 40, 'cpu to load record')
    flag.DEFINE_integer('height', 224, 'height of the image to the network')
    flag.DEFINE_integer('width', 224, 'weight of the image to the network')

    # dataset config
    flag.DEFINE_integer('prefetch_buffer_size', 2, 'buffer size for preload train features')
    flag.DEFINE_integer('shuffle_buffer_size', 10000, 'buffer size for shuffle train features')

    # net config
    flag.DEFINE_float('drop_rate', 0.25, 'dropout rate')
    flag.DEFINE_float('l2_rate', 0.001, 'l2 regularization rate')
    flag.DEFINE_bool('l2_loss', True, 'using l2 loss or not')
    flag.DEFINE_float('l1_rate', 0.001, 'l1 regularization rate')
    flag.DEFINE_bool('l1_loss', False, 'using l1 loss or not')

    # training schedule
    flag.DEFINE_float('learning_rate', 0.001, 'learning rate for updating parameters')
    flag.DEFINE_integer('batch_size', 64, 'train images batch size')
    flag.DEFINE_integer('num_epochs', 50, 'training epoch')
    flag.DEFINE_string('optimizer', 'Momentum', 'learning algorithms {Momentum, Adam, RMSProp, Adagrad}')

    # summary config
    flag.DEFINE_string('train_summary_path', '/mnt/xk/experiment/log/photos/train', 'path to save train summary')
    flag.DEFINE_string('valid_summary_path', '/mnt/xk/experiment/log/photos/valid', 'path to save valid summary')
    flag.DEFINE_bool('gradients_summary', False, 'whether to summary gradients')
    flag.DEFINE_bool('variables_summary', False, 'whether to summary variables distribution')
    flag.DEFINE_bool('metric_summary', True, 'whether to summary metric')
    flag.DEFINE_bool('total_loss_summary', True, 'whether to summary total loss')
    flag.DEFINE_bool('main_loss_summary', False, 'whether to summary main loss')
    flag.DEFINE_bool('summary', True, 'whether to summary')

    # save path
    flag.DEFINE_string('model_path', MODEL_PATH, 'path to save model')

    FLAGS = flag.FLAGS

    # dataset
    images, labels, train_init_op, valid_init_op = ClassificationDataset().data()

    with tf.name_scope("net_config"):
        images_placeholder = tf.placeholder(tf.float32, shape=(None, FLAGS.height, FLAGS.width, 3),
                                            name='images')
        labels_placeholder = tf.placeholder(tf.float32, shape=(None, FLAGS.num_classes), name='labels')

    # net
    nets = VGG16(weights_path=os.sep.join([MODEL_PATH, 'vgg16.npy']))
    # inference
    logits = nets.inference(images_placeholder)

    # loss
    loss = CrossEntropyLoss(logits, labels_placeholder)
    loss_update = loss.loss()

    # optimizer for training network
    optimizer = Optimizer()
    train_op = optimizer.minimize(loss.losses)

    # define metric
    metrics = Accuracy(logits, labels_placeholder)
    metric_update = metrics.metric()

    # batch_update
    update = tf.group([loss_update, metric_update], name='metrics_and_batch_to_updated')

    # visualize grads and variable distribution with summary
    visualize_grads_and_vars(optimizer, loss.losses)

    # summary
    summary_writer = Summary()
    if FLAGS.gradients_summary or FLAGS.variables_summary:
        grads_and_vars_summary = tf.summary.merge(tf.get_collection('grads_and_vars'))
    else:
        grads_and_vars_summary = None

    batch_summary = tf.summary.merge(tf.get_collection('batch_summary'))
    epoch_summary = tf.summary.merge(tf.get_collection('epoch_summary'))

    saver = tf.train.Saver(max_to_keep=None)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer.init_op(sess)
        bar_training = Bar('[INFO] training')
        bar_validating = Bar('[INFO] validating')
        for index_in_epoch in range(FLAGS.num_epochs):
            sess.run(train_init_op)
            # reset local vars
            sess.run(tf.get_collection('epoch_initializer'))
            index_in_batch = 0
            while True:
                try:
                    # reinitialize for new batch reset local parameters
                    sess.run(tf.get_collection('batch_initializer'))
                    _images, _labels = sess.run([images, labels])
                    feed_dict = {images_placeholder: _images, labels_placeholder: _labels}
                    # run training process
                    # log gradient and variables distribution
                    if grads_and_vars_summary is not None:
                        _, __, _summary = sess.run([train_op, update, grads_and_vars_summary],
                                                   feed_dict=feed_dict)
                        summary_writer.log(_summary, 'train', epoch_wise=False)
                    else:
                        _, __, = sess.run([train_op, update], feed_dict=feed_dict)
                    # summary batch metrics and loss
                    _batch_summary = sess.run(batch_summary)
                    summary_writer.log(_batch_summary, 'train', epoch_wise=False)
                    bar_training.update()
                except tf.errors.OutOfRangeError:
                    bar_training.clear()
                    # summary epoch metrics and loss
                    _epoch_summary = sess.run(epoch_summary)
                    summary_writer.log(_epoch_summary, 'train', epoch_wise=True)
                    break

            # valid
            sess.run(valid_init_op)
            sess.run(tf.get_collection('epoch_initializer'))
            while True:
                try:
                    _images, _labels = sess.run([images, labels])
                    feed_dict = {images_placeholder: _images, labels_placeholder: _labels}
                    sess.run([loss_update, metric_update], feed_dict=feed_dict)
                    bar_validating.update()
                except tf.errors.OutOfRangeError:
                    bar_validating.clear()
                    _epoch_summary = sess.run(epoch_summary)
                    summary_writer.log(_epoch_summary, 'valid', epoch_wise=True)
                    saver.save(sess, os.path.join(FLAGS.model_path, "photos_2classes"), index_in_epoch)
                    break