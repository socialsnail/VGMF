# -*- coding: utf-8 -*-
# @Time    : 2019/1/18  15:33
# @Author  : Dyn


if __name__ == '__main__':
    import os
    import tensorflow as tf
    from model.images.arc.vgg import VGG16
    from model.images.data.dataset import UserLocationPhotosDataset
    from config.config import *
    import warnings
    import numpy as np
    import pickle
    warnings.filterwarnings('ignore')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    flag = tf.app.flags

    # dataset
    flag.DEFINE_integer('num_classes', 2, 'num classes of place')
    flag.DEFINE_integer('seed', 253, 'random seeds')
    flag.DEFINE_string('train_path', LOCATIONS_REC_PHOTOS, 'path to save train file')
    # flag.DEFINE_string('valid_path', CATEGORIES_CLASSIFICATION_VALID2_FILE, 'path to save valid file')
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
    flag.DEFINE_bool('metric_summary', False, 'whether to summary metric')
    flag.DEFINE_bool('total_loss_summary', False, 'whether to summary total loss')
    flag.DEFINE_bool('main_loss_summary', False, 'whether to summary main loss')
    flag.DEFINE_bool('summary', False, 'whether to summary')

    # save path
    flag.DEFINE_string('model_path', MODEL_PATH, 'path to save model')

    FLAGS = flag.FLAGS

    num_latent_features = 512
    num_users_or_locations = 2621 # 2337


    # dataset
    images, user_location_id, train_init_op = UserLocationPhotosDataset().data()

    with tf.name_scope("net_config"):
        images_placeholder = tf.placeholder(tf.float32, shape=(None, FLAGS.height, FLAGS.width, 3),
                                            name='images')

    # net
    nets = VGG16(weights_path=os.sep.join([MODEL_PATH, 'vgg16.npy']))
    # inference
    logits = nets.inference(images_placeholder)
    features = nets.get_features()

    saver = tf.train.Saver(max_to_keep=None)

    features_dict = {'features': np.zeros(shape=[num_users_or_locations, num_latent_features]),
                     'cnt': np.zeros(shape=[num_users_or_locations])}

    def log(features, ids):
        for feature,  id_ in zip(features, ids):
            features_dict['cnt'][id_] += 1
            features_dict['features'][id_] += feature


    with tf.Session() as sess:
        saver.restore(sess, FLAGS.model_path + '/photos_2classes-%d' % (13))

        sess.run(train_init_op)
        # reset local vars
        index_in_epoch = 0
        while True:
            try:
                # reinitialize for new batch reset local parameters
                _images, _user_or_location_id = sess.run([images, user_location_id])
                feed_dict = {images_placeholder: _images}
                _features = sess.run(features, feed_dict=feed_dict)
                log(_features, _user_or_location_id)
                index_in_epoch += 1
                print('Finished %d photos' % (index_in_epoch * FLAGS.batch_size))
            except tf.errors.OutOfRangeError:
                break

    for index, cnt in enumerate(features_dict['cnt']):
        if cnt > 0:
            features_dict['features'][index] /= cnt

    pickle.dump(features_dict, open(LOCATIONS_PHOTOS_FEATURES_2CLASSES, mode='wb'))