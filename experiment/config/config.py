# -*- coding: utf-8 -*-
# @Time    : 2018/6/14  16:02
# @Author  : Dyn

DATA_PATH = '/mnt/xk/experiment/checkin_NY.csv'
FILTER_NUM = 10
OUTPUT_PATH = '/mnt/xk/experiment/log'
DATA_ROOT = '/mnt/xk/experiment/dataset'
CONFIG = '/mnt/xk/experiment/config'
MODEL_PATH = '/mnt/xk/experiment/model'
PHOTO_PATH = '/mnt/xk/fousquareData'

suffix = ''

TRAIN = '/mnt/xk/experiment/dataset/train_2000%s.csv' % suffix
VALID = '/mnt/xk/experiment/dataset/valid_2000%s.csv' % suffix
TEST = '/mnt/xk/experiment/dataset/test_2000%s.csv' % suffix
SPARSE_DATA = '/mnt/xk/experiment/dataset/sparse_2000%s.pkl' % suffix
VALID_GROUND_TRUTH = '/mnt/xk/experiment/dataset/valid_ground_truth_2000%s.pkl' % suffix
TEST_GROUND_TRUTH = '/mnt/xk/experiment/dataset/test_ground_truth_2000%s.pkl' % suffix
LOCATIONS = '/mnt/xk/experiment/dataset/locations_2000%s.pkl' % suffix
USERS = '/mnt/xk/experiment/dataset/users_2000%s.pkl' % suffix
LOCATIONS_PHOTOS_PATH = '/mnt/xk/experiment/dataset/locations_photos_path_2000%s.pkl' % suffix
USERS_PHOTOS_PATH = '/mnt/xk/experiment/dataset/users_photos_path_2000%s.pkl' % suffix

LOCATION_PHOTOS = '/mnt/xk/experiment/dataset/location_photos_2000%s.pkl' % suffix
CATEGORIES = '/mnt/xk/experiment/data/categories.json'
NO_PHOTOS_LOCATIONS = '/mnt/xk/experiment/dataset/no_photos_locations_2000%s.pkl' % suffix
CATEGORIES_CLASSIFICATION_TRAIN_FILE = '/mnt/xk/experiment/dataset/category_classification_train_2000%s.tfrecords' % suffix
CATEGORIES_CLASSIFICATION_VALID_FILE = '/mnt/xk/experiment/dataset/category_classification_valid_2000%s.tfrecords' % suffix

CATEGORIES_CLASSIFICATION_TRAIN5_FILE = '/mnt/xk/experiment/dataset/category_classification_train5_2000%s.tfrecords' % suffix
CATEGORIES_CLASSIFICATION_VALID5_FILE = '/mnt/xk/experiment/dataset/category_classification_valid5_2000%s.tfrecords' % suffix

CATEGORIES_CLASSIFICATION_TRAIN2_FILE = '/mnt/xk/experiment/dataset/category_classification_train2_2000%s.tfrecords' % suffix
CATEGORIES_CLASSIFICATION_VALID2_FILE = '/mnt/xk/experiment/dataset/category_classification_valid2_2000%s.tfrecords' % suffix

LOCATIONS_REC_PHOTOS = '/mnt/xk/experiment/dataset/location_rec_photos_2000%s.tfrecords' % suffix
USERS_REC_PHOTOS = '/mnt/xk/experiment/dataset/user_rec_photos_2000%s.tfrecords' % suffix

LOCATIONS_PHOTOS_FEATURES_2CLASSES = '/mnt/xk/experiment/dataset/location_photos_features_2classes_2000%s.tfrecords' % suffix
USERS_PHOTOS_FEATURES_2CLASSES = '/mnt/xk/experiment/dataset/user_photos_features_2classes_2000%s.tfrecords' % suffix


DISTANCE_EMBEDDING_TRAIN = '/mnt/xk/experiment/dataset/train_direc_dis_2000%s.tfrecords' % suffix
DISTANCE_EMBEDDING_TRAIN2 = '/mnt/xk/experiment/dataset/train_direc_dis2_2000%s.tfrecords' % suffix
DISTANCE_EMBEDDING_TEST = '/mnt/xk/experiment/dataset/test_direc_dis_2000%s.tfrecords' % suffix

DISTANCE = '/mnt/xk/experiment/dataset/distance_2000%s.pkl' % suffix
DEVIATION = '/mnt/xk/experiment/dataset/deviation_2000%s.pkl' % suffix
MEAN = '/mnt/xk/experiment/dataset/mean_2000%s.pkl' % suffix
PROBABILITY = '/mnt/xk/experiment/dataset/probability_2000%s.pkl' % suffix

PROBABILITY_10 = '/mnt/xk/experiment/dataset/probability_2000_10%s.pkl' % suffix
PROBABILITY_25 = '/mnt/xk/experiment/dataset/probability_2000_25%s.pkl' % suffix
PROBABILITY_50 = '/mnt/xk/experiment/dataset/probability_2000_50%s.pkl' % suffix
PROBABILITY_100 = '/mnt/xk/experiment/dataset/probability_2000_100%s.pkl' % suffix
PROBABILITY_500 = '/mnt/xk/experiment/dataset/probability_2000_500%s.pkl' % suffix
PROBABILITY_TOTAL = '/mnt/xk/experiment/dataset/probability_2000_total%s.pkl' % suffix
PROBABILITY_TOTAL2 = '/mnt/xk/experiment/dataset/probability_2000_total2%s.pkl' % suffix

GRID = '/mnt/xk/experiment/dataset/grid_2000%s.pkl' % suffix

NCF_DISTANCE_TRAIN = '/mnt/xk/experiment/dataset/ncf_train_2000%s.pkl' % suffix
NCF_DISTANCE_VALID = '/mnt/xk/experiment/dataset/ncf_valid_2000%s.pkl' % suffix
NCF_DISTANCE_TEST = '/mnt/xk/experiment/dataset/ncf_test_2000%s.pkl' % suffix
NCF_DISTANCE_SAMPLE_USER_ID ='/mnt/xk/experiment/dataset/ncf_sample_user_id_2000%s.pkl' % suffix

KDE_PREDICT = '/mnt/xk/experiment/dataset/kde_predict_2000%s.pkl' % suffix
MF_PREDICT = '/mnt/xk/experiment/dataset/mf_predict_2000%s.pkl' % suffix

MF_MODEL = '/mnt/xk/experiment/model/mf_2000%s.pkl' % suffix

VISUAL_WEIGHTS = '/mnt/xk/experiment/dataset/visual_weights_2000%s.pkl' % suffix


IGSLR_UCF = '/mnt/xk/experiment/dataset/IGSLR_UCF_predict_2000%s.pkl' % suffix
IGSLR_KDE_PREDICT = '/mnt/xk/experiment/dataset/probability_2000_total%s.pkl' % suffix
VPOI_USER_IMAGES = '/mnt/xk/experiment/dataset/vpoi_user_images_2000%s.tfrecords' % suffix
VPOI_USER_VGG_FEATURES = '/mnt/xk/experiment/dataset/vpoi_user_vgg_features_2000%s.pkl' % suffix
VPOI_LOCATION_VGG_FEATURES = '/mnt/xk/experiment/dataset/vpoi_location_vgg_features_2000%s.pkl' % suffix
VPOI_LOCATION_IMAGES = '/mnt/xk/experiment/dataset/vpoi_location_images_2000%s.tfrecords' % suffix
VPOI_MATRIX = '/mnt/xk/experiment/dataset/vpoi_matrix_2000%s.pkl' % suffix

if __name__ == '__main__':
    pass
