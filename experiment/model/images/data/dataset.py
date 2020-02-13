# -*- coding: utf-8 -*-
# @Time    : 2018/12/7  22:06
# @Author  : Dyn
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class ClassificationDataset(object):
    def __init__(self):
        self.image, self.label, self.train_init_op, self.valid_init_op = self._dataset_iterator()
        self.label = tf.one_hot(self.label, FLAGS.num_classes)

    def parse(self, example):
        feature = {'image_path': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}
        parsed = tf.parse_single_example(serialized=example, features=feature)
        return {'image_path': parsed['image_path'], 'label': parsed['label']}

    @staticmethod
    def _read_image(x):
        # read image
        image_string = tf.read_file(x['image_path'])
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize_images(image, [FLAGS.height, FLAGS.width])
        return {'image': image, 'label': x['label']}

    @staticmethod
    def _image_aug(x):
        image = x['image']
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.adjust_brightness(image, delta=0.2)
        return {'image': image, 'label': x['label']}

    def _make_dataset(self, path, train_or_valid='train'):
        dataset = tf.data.TFRecordDataset([path])
        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, seed=FLAGS.seed)
        dataset = dataset.map(self.parse, num_parallel_calls=FLAGS.num_cpus)
        dataset = dataset.map(self._read_image, num_parallel_calls=FLAGS.num_cpus)
        # if train_or_valid == 'train':
        #     dataset = dataset.map(self._image_aug, num_parallel_calls=FLAGS.num_cpus)
        dataset = dataset.batch(FLAGS.batch_size)
        dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
        return dataset

    def _dataset_iterator(self):
        train_dataset = self._make_dataset(FLAGS.train_path, 'train')
        valid_dataset = self._make_dataset(FLAGS.valid_path, 'valid')
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)
        next_element = iterator.get_next()

        train_init_op = iterator.make_initializer(train_dataset)
        valid_init_op = iterator.make_initializer(valid_dataset)
        return next_element['image'], next_element['label'], train_init_op, valid_init_op

    def data(self):
        return self.image, self.label, self.train_init_op, self.valid_init_op


class UserLocationPhotosDataset(object):
    def __init__(self):
        self.image, self.id_, self.train_init_op,  = self._dataset_iterator()

    def parse(self, example):
        feature = {'image_path': tf.FixedLenFeature([], tf.string),
                   'id': tf.FixedLenFeature([], tf.int64)}
        parsed = tf.parse_single_example(serialized=example, features=feature)
        return {'image_path': parsed['image_path'], 'id': parsed['id']}

    @staticmethod
    def _read_image(x):
        # read image
        image_string = tf.read_file(x['image_path'])
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize_images(image, [FLAGS.height, FLAGS.width])
        return {'image': image, 'id': x['id']}

    def _make_dataset(self, path):
        dataset = tf.data.TFRecordDataset([path])
        dataset = dataset.map(self.parse, num_parallel_calls=FLAGS.num_cpus)
        dataset = dataset.map(self._read_image, num_parallel_calls=FLAGS.num_cpus)
        dataset = dataset.batch(FLAGS.batch_size)
        dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
        return dataset

    def _dataset_iterator(self):
        train_dataset = self._make_dataset(FLAGS.train_path)
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)
        next_element = iterator.get_next()

        train_init_op = iterator.make_initializer(train_dataset)
        return next_element['image'], next_element['id'], train_init_op,

    def data(self):
        return self.image, self.id_, self.train_init_op


if __name__ == '__main__':
    import os
    from config.config import *

    flag = tf.app.flags
    # flag.DEFINE_string('train_path', os.path.join(DATA_ROOT, 'location_images_train.tfrecords'),
    #                    'path to save train file')
    flag.DEFINE_string('valid_path', os.path.join(DATA_ROOT, 'location_images_valid.tfrecords'),
                       'path to save valid file')
    flag.DEFINE_integer('shuffle_buffer_size', 1000, 'num of buffer')
    flag.DEFINE_integer('batch_size', 256, 'num of buffer')
    flag.DEFINE_integer('seed', 435, 'num of buffer')
    flag.DEFINE_integer('num_cpus', 40, 'num of buffer')
    flag.DEFINE_integer('crop_height', 256, 'num of buffer')
    flag.DEFINE_integer('crop_width', 256, 'num of buffer')
    flag.DEFINE_integer('height', 224, 'num of buffer')
    flag.DEFINE_integer('width', 224, 'num of buffer')
    flag.DEFINE_integer('num_classes', 8, 'num classes of place')
    flag.DEFINE_string('train_path', LOCATIONS_REC_PHOTOS, 'path to save train file')
    flag.DEFINE_integer('prefetch_buffer_size', 2, 'buffer size for preload train features')


    dataset = UserLocationPhotosDataset()
    images, ids, init_op1 = dataset.data()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init_op1)
        while True:
            try:
                x = sess.run(images)[0]
                print(x[0])
            except Exception as e:
                print(e)