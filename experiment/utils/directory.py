# -*- coding: utf-8 -*-
# Time: 2018/11/29 10:01
# Author: Dyn
import os


def clear_dir(path):
    path_list = path.split(os.sep)[1:]
    path_pre = os.sep
    for d in path_list[:-1]:
        path_pre = os.path.join(path_pre, d)
        if not os.path.exists(path_pre):
            os.mkdir(path_pre)
    if os.path.exists(path):
        os.remove(path)

# if not os.path.exists(train_summary_path):
#     print(['[INFO] create dir {}'.format(train_summary_path)])
#     os.mkdir(train_summary_path)
# else:
#     del_dir = [x for x in os.listdir(train_summary_path)]
#     for i in del_dir:
#         print(os.sep.join([train_summary_path, i]))
#         os.remove(os.sep.join([train_summary_path, i]))
#
# if not os.path.exists(valid_summary_path):
#     print(['[INFO] create dir {}'.format(valid_summary_path)])
#     os.mkdir(valid_summary_path)
# else:
#     del_dir = [x for x in os.listdir(valid_summary_path)]
#     for i in del_dir:
#         print(os.sep.join([valid_summary_path, i]))
#         os.remove(os.sep.join([valid_summary_path, i]))

if __name__ == '__main__':
    clear_dir('/mnt/xk/nlp/ls/aa/fe/w/a.log')