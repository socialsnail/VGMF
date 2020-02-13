# -*- coding: utf-8 -*-
# Time: 2018/12/7 12:32
# Author: Dyn

import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature_list(values):
    return tf.train.Feature(int64_list=tf.train.FloatList(value=values))


class SequenceWriter(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __enter__(self):
        self.writer = tf.python_io.TFRecordWriter(self.file_name)
        return self

    def write(self, sequence, label):
        example = self.make_example(sequence, label)
        self.writer.write(example.SerializeToString())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
        print('[INFO] writing to %s' % self.file_name)

    def make_example(self, sequence, label):
        example = tf.train.SequenceExample()
        sequence_length = len(sequence)
        example.context.feature['length'].int64_list.value.append(sequence_length)
        example.context.feature['label'].int64_list.value.append(label)

        words_list = example.feature_lists.feature_list['words']
        for word in sequence:
            words_list.feature.add().int64_list.value.append(word)
        return example


class ImagesPathWriter(object):
    def __init__(self, file_name=None):
        self.file_name = file_name

    def __enter__(self):
        self.writer = tf.python_io.TFRecordWriter(self.file_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
        print('[INFO] writing to %s' % self.file_name)

    def write(self, sequence, label):
        example = self.make_example(sequence, label)
        self.writer.write(example.SerializeToString())

    def make_example(self, path, label):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image_path': _bytes_feature(tf.compat.as_bytes(path)),
                'label': _int64_feature(int(label)),
            }
        ))
        return example


class UserLocPathWriter(object):
    def __init__(self, file_name=None):
        self.file_name = file_name

    def __enter__(self):
        self.writer = tf.python_io.TFRecordWriter(self.file_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
        print('[INFO] writing to %s' % self.file_name)

    def write(self, path, user_or_loc_id):
        example = self.make_example(path, user_or_loc_id)
        self.writer.write(example.SerializeToString())

    def make_example(self, path, id_):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image_path': _bytes_feature(tf.compat.as_bytes(path)),
                'id': _int64_feature(id_),
            }
        ))
        return example

class ImagesWriter(object):
    def __init__(self, file_name=None):
        self.file_name = file_name
        self.writer = tf.python_io.TFRecordWriter(file_name)

    def __enter__(self):
        self.writer = tf.python_io.TFRecordWriter(self.file_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
        print('[INFO] writing to %s' % self.file_name)

    def write(self, sequence, label):
        example = self.make_example(sequence, label)
        self.writer.write(example.SerializeToString())

    def make_example(self, image, label):
        rows = image.shape[0]
        cols = image.shape[1]
        depth = image.shape[2]
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(label)),
                'image_raw': _bytes_feature(tf.compat.as_bytes(image.tostring()))
            }
        ))
        return example


if __name__ == '__main__':
    with ImagesPathWriter('test.tfrecords') as writer:
        writer.write(tf.compat.as_bytes('now'), 0)


    def parse(ex):
        feature = {'image_path': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}
        parsed = tf.parse_single_example(serialized=ex, features=feature)
        return {'image_path': parsed['image_path'], 'label': parsed['label']}


    dataset = tf.data.TFRecordDataset('test.tfrecords')
    dataset = dataset.map(parse)
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    init_op = iterator.make_initializer(dataset)
    next = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init_op)
        print(sess.run(next))
