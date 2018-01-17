import tensorflow as tf
import os
import random
import numpy as np
import cv2
import face_alignment
import pickle
import platform
import re


# 得到当前路径
def get_patch_path(patch_name):
    if platform.system() == 'Windows':
        patch_path = os.path.join('./DataSet/', patch_name)
    if platform.system() == 'Linux':
        userRoot = os.environ['HOME']
        patch_path = os.path.join(userRoot + '/DataSet/', patch_name)
    return patch_path


# 更新数据集信息
def update_dataset_imformation(patch_name):
    if not os.path.exists('data'):
        os.mkdir('data')
    patch_path = get_patch_path(patch_name)
    if not os.path.exists(patch_path):
        os.mkdir(patch_path)
    people_list = os.listdir(patch_path)
    #people_list = list(enumerate(people_list))
    people_num = len(people_list)
    people_num_for_deepid = people_num * 4 // 5  # 用于训练deepid的人数
    people_num_for_verification_model = people_num - people_num_for_deepid  # 用于验证模型的人数
    people_list_for_deepid = people_list[0:people_num_for_deepid]
    people_list_for_deepid=list(enumerate(people_list_for_deepid))
    people_list_for_verification_model = people_list[people_num_for_deepid:]
    people_list_for_verification_model=list(enumerate(people_list_for_verification_model))
    return people_list, \
           people_list_for_deepid, \
           people_list_for_verification_model, \
           people_num, \
           people_num_for_deepid, \
           people_num_for_verification_model


def generate_patch(patch_name):
    def getRois(img, *landmarks):
        if not len(landmarks) in [1, 2]:
            raise Exception('para num error')
        elif len(landmarks) == 1:
            landmark = landmarks[0]
            roi_size_half = min(landmark[0],
                                landmark[1],
                                img.shape[0] - landmark[1],
                                img.shape[1] - landmark[0],
                                img.shape[0] // 4
                                )
            img = img[
                  landmark[1] - roi_size_half:landmark[1] + roi_size_half,
                  landmark[0] - roi_size_half:landmark[0] + roi_size_half
                  ]
            return img

        elif len(landmarks) == 2:
            # 正方形区域的半边长
            left_landmark = landmarks[0]
            right_landmark = landmarks[1]
            roi_size_half = min(left_landmark[0],
                                left_landmark[1],
                                img.shape[0] - left_landmark[1],
                                img.shape[1] - left_landmark[0],
                                right_landmark[0],
                                right_landmark[1],
                                img.shape[0] - right_landmark[1],
                                img.shape[1] - right_landmark[0]
                                )
            left_img = img[
                       left_landmark[1] - roi_size_half:left_landmark[1] + roi_size_half,
                       left_landmark[0] - roi_size_half:left_landmark[0] + roi_size_half
                       ]
            right_img = img[
                        right_landmark[1] - roi_size_half:right_landmark[1] + roi_size_half,
                        right_landmark[0] - roi_size_half:right_landmark[0] + roi_size_half
                        ]
            return left_img, right_img

    if patch_name == 'images':
        raise Exception("you can't generate origin dataset")

    people_count = 0
    people_list_for_deepid = update_dataset_imformation('images')[1]
    people_num = update_dataset_imformation('images')[4]
    dataset_path = get_patch_path('images')
    for people in people_list_for_deepid:
        people = people[1]
        people_count += 1
        print('Generating...', people_count, '/', people_num)
        # 原图改变尺寸
        if patch_name == 'patch_1':
            patch_path = get_patch_path(patch_name)
            if not os.path.exists(patch_path):
                os.mkdir(patch_path)
            for img_name in os.listdir(os.path.join(dataset_path, people)):
                origin_img_path = os.path.join(dataset_path, people, img_name)
                dst_img_path = os.path.join(patch_path, people, img_name)
                img = cv2.imread(origin_img_path)
                img = cv2.resize(img, (31, 31))
                cv2.imwrite(dst_img_path, img)

        # 两只眼睛
        if patch_name in ['patch_2', 'patch_3']:
            patch_2_path = get_patch_path('patch_2')
            patch_3_path = get_patch_path('patch_3')
            if not os.path.exists(os.path.join(patch_2_path, people)):
                os.mkdir(os.path.join(patch_2_path, people))
            if not os.path.exists(os.path.join(patch_3_path, people)):
                os.mkdir(os.path.join(patch_3_path, people))
            for img_name in os.listdir(os.path.join(dataset_path, people)):
                origin_img_path = os.path.join(dataset_path, people, img_name)
                left_dst_img_path = os.path.join(patch_2_path, people, img_name)
                right_dst_img_path = os.path.join(patch_3_path, people, img_name)
                img = cv2.imread(origin_img_path)
                left_landmark = face_alignment.alignment(img)[0]
                right_landmark = face_alignment.alignment(img)[1]
                try:
                    left_img, right_img = getRois(img, left_landmark, right_landmark)
                    left_img = cv2.resize(left_img, (31, 31))
                    right_img = cv2.resize(right_img, (31, 31))
                    cv2.imwrite(left_dst_img_path, left_img)
                    cv2.imwrite(right_dst_img_path, right_img)
                except:
                    print('Something wrong happens ...')

        # 鼻尖
        if patch_name == 'patch_4':
            patch_4_path = get_patch_path('patch_4')
            if not os.path.exists(os.path.join(patch_4_path, people)):
                os.mkdir(os.path.join(patch_4_path, people))
            for img_name in os.listdir(os.path.join(dataset_path, people)):
                origin_img_path = os.path.join(dataset_path, people, img_name)
                dst_img_path = os.path.join(patch_4_path, people, img_name)
                img = cv2.imread(origin_img_path)
                landmark = face_alignment.alignment(img)[2]
                try:
                    img = getRois(img, landmark)
                    img = cv2.resize(img, (31, 31))
                    cv2.imwrite(dst_img_path, img)
                except:
                    print('Something wrong happens ...')

        # 两个嘴角
        if patch_name in ['patch_5', 'patch_6']:
            patch_5_path = get_patch_path('patch_5')
            patch_6_path = get_patch_path('patch_6')
            if not os.path.exists(os.path.join(patch_5_path, people)):
                os.mkdir(os.path.join(patch_5_path, people))
            if not os.path.exists(os.path.join(patch_6_path, people)):
                os.mkdir(os.path.join(patch_6_path, people))
            for img_name in os.listdir(os.path.join(dataset_path, people)):
                origin_img_path = os.path.join(dataset_path, people, img_name)
                left_dst_img_path = os.path.join(patch_5_path, people, img_name)
                right_dst_img_path = os.path.join(patch_6_path, people, img_name)
                img = cv2.imread(origin_img_path)
                left_landmark = face_alignment.alignment(img)[3]
                right_landmark = face_alignment.alignment(img)[4]
                try:
                    left_img, right_img = getRois(img, left_landmark, right_landmark)
                    left_img = cv2.resize(left_img, (31, 31))
                    right_img = cv2.resize(right_img, (31, 31))
                    cv2.imwrite(left_dst_img_path, left_img)
                    cv2.imwrite(right_dst_img_path, right_img)
                except:
                    print('Something wrong happens ...')

        # 生成对应的灰度patch
        dst_patch_name_list = ['patch_' + str(num) for num in range(7, 13)]
        src_patch_name_list = ['patch_' + str(num) for num in range(1, 7)]
        patch_dict = dict(list(zip(dst_patch_name_list, src_patch_name_list)))
        if patch_name in dst_patch_name_list:
            patch_path = get_patch_path(patch_name)
            if not os.path.exists(os.path.join(patch_path, people)):
                os.mkdir(os.path.join(patch_path, people))
            src_patch_name = patch_dict[patch_name]
            src_patch_path = get_patch_path(src_patch_name)
            for img_name in os.listdir(os.path.join(src_patch_path, people)):
                origin_img_path = os.path.join(src_patch_path, people, img_name)
                dst_img_path = os.path.join(patch_path, people, img_name)
                img = cv2.imread(origin_img_path, 0)
                cv2.imwrite(dst_img_path, img)


# 获取一个人的所有图片（文件路径）列表 乱序之后输出该列表
def get_img_path_for_one_people(people_dir, *truncation):
    if people_dir[-1] != ('/' or '\\'):
        people_dir += '/'
    people_img_path = []
    for img in os.listdir(people_dir):
        img_path = people_dir + img
        if truncation[0] == True:  # 截断 方便生成训练分类模型的数据
            p = re.compile('\w*.\d+.jpg')  # 用正则表达式匹配 进行截断
            img_path = p.search(img_path).group()  # 截断之后的数据 只剩下人名和图像编号
        people_img_path.append(img_path)
    random.shuffle(people_img_path)
    return people_img_path


# 构建用于训练deepid的csv
def generate_csv_for_deepid(patch_name):
    people_list_for_deepid = update_dataset_imformation(patch_name)[1]
    dataset_for_deepid_train = []
    dataset_for_deepid_valid = []
    for label, people in people_list_for_deepid:
        patch_path = get_patch_path(patch_name)
        people_img_path = get_img_path_for_one_people(os.path.join(patch_path, people))
        people_imgs_for_train = people_img_path[0:9 * len(people_img_path) // 10]  # 取其中9/10作为训练集
        people_imgs_for_valid = people_img_path[9 * len(people_img_path) // 10:]  # 其中1/10作为验证（防止过拟合）
        dataset_for_deepid_train += zip(people_imgs_for_train, [str(label)] * len(people_imgs_for_train))
        dataset_for_deepid_valid += zip(people_imgs_for_valid, [str(label)] * len(people_imgs_for_valid))

    random.shuffle(dataset_for_deepid_train)
    random.shuffle(dataset_for_deepid_valid)
    csv_for_deepid_train = 'data/' + patch_name + '_for_deepid_train.csv'
    csv_for_deepid_valid = 'data/' + patch_name + '_for_deepid_valid.csv'
    with open(csv_for_deepid_train, 'w') as f:
        for item in dataset_for_deepid_train:
            print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
    with open(csv_for_deepid_valid, 'w') as f:
        for item in dataset_for_deepid_valid:
            print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)


# 构建用于训练分类模型的csv
def generate_csv_for_verification_model():
    people_list_for_verification_model = update_dataset_imformation('images')[2]
    dataset_for_verification_model = []
    for label, people in people_list_for_verification_model:
        path = get_patch_path('images')
        people_img_path = get_img_path_for_one_people(os.path.join(path, people), True)
        dataset_for_verification_model += zip(people_img_path, [str(label)] * len(people_img_path))
    random.shuffle(dataset_for_verification_model)
    csv_for_verification_model = 'data/verification_model.csv'
    with open(csv_for_verification_model, 'w') as f:
        for item in dataset_for_verification_model:
            print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)


# 读取csv文件
def read_csv(csv_path):
    img_pathList = []
    labelList = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    count = 0
    line_num = len(lines)
    for line in lines:
        count += 1
        print('Loading...', count, '/', line_num)
        line = line.split(' ')
        img_path = line[0].strip(' ' and '\n')
        img_pathList.append(img_path)
        label = line[-1].strip(' ' and '\n')
        labelList.append(label)
    return img_pathList, labelList


# 生成用于训练deepid的tfrecords
def generate_recorder_for_deepid(patch_name):
    def generate_recorder(TFwriter, pairList):
        count = 0
        num = len(pairList)
        for img_path, label in pairList:
            count += 1
            print('recording...', count, '/', num)
            img = cv2.imread(img_path, -1)
            if img.shape == (31, 31):
                img.resize(31, 31, 1)
            imgRaw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgRaw]))
            }))
            TFwriter.write(example.SerializeToString())
        TFwriter.close()

        TFwriter = tf.python_io.TFRecordWriter('data/' + patch_name + '_train.tfrecords')
        img_pathList, labelList = read_csv('data/' + patch_name + '_for_deepid_train.csv')
        pairList = list(zip(img_pathList, labelList))
        generate_recorder(TFwriter, pairList)
        TFwriter = tf.python_io.TFRecordWriter('data/' + patch_name + '_valid.tfrecords')
        img_pathList, labelList = read_csv('data/' + patch_name + '_for_deepid_valid.csv')
        pairList = list(zip(img_pathList, labelList))
        generate_recorder(TFwriter, pairList)


# 生成用于训练deepid的pickle（和tfrecorder 二选一）
def generate_pickle_for_deepid(patch_name):
    csv_for_deepid_train = 'data/' + patch_name + '_for_deepid_train.csv'
    csv_for_deepid_valid = 'data/' + patch_name + '_for_deepid_valid.csv'
    pkl_train = 'data/' + patch_name + '_train.pkl'
    pkl_valid = 'data/' + patch_name + '_valid.pkl'
    # 生成train集的pickle
    img_pathList, labelList = read_csv(csv_for_deepid_train)
    count = 0
    num = len(img_pathList)
    imgList = []
    for img_path in img_pathList:
        count += 1
        print('reading...', count, '/', num)
        img = cv2.imread(img_path, -1)
        if img.shape == (31, 31):
            img.resize(31, 31, 1)
        imgList.append(img)
    imgArr = np.asarray(imgList, dtype='uint8')
    labelArr = np.asarray(labelList, dtype='float32')
    with open(pkl_train, 'wb') as f:
        pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)

    # 生成valid集的pickle
    img_pathList, labelList = read_csv(csv_for_deepid_valid)
    count = 0
    num = len(img_pathList)
    imgList = []
    for img_path in img_pathList:
        count += 1
        print('reading...', count, '/', num)
        img = cv2.imread(img_path, -1)
        if img.shape == (31, 31):
            img.resize(31, 31, 1)
        imgList.append(img)
    imgArr = np.asarray(imgList, dtype='uint8')
    labelArr = np.asarray(labelList, dtype='float32')
    with open(pkl_valid, 'wb') as f:
        pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)


# 判断输入图像是否有且只有一张人脸 （毕竟数据集里混进了很多奇怪的东西 输入之前要筛选一下）
def is_face(img):
    face_detector = face_alignment.detector
    if type(img) != np.ndarray:
        return False
    if img.ndim != 3:
        return False
    faces = face_detector(img, 1)
    if (len(faces) == 1):
        return True
    else:
        return False


# 清洗(原始)数据 去掉混进来奇怪的东西（非人脸 非rgb）
def wash_data():
    dataset_path = get_patch_path('image')
    people_count = 0
    img_count = 0
    people_num = len(os.listdir(dataset_path))
    for people in os.listdir(dataset_path):
        people_count += 1
        for img_name in os.listdir(os.path.join(dataset_path, people)):
            img_count += 1
            print('Washing...', people_count, '/', people_num, ':', img_count)
            img_path = os.path.join(dataset_path, people, img_name)
            img = cv2.imread(img_path)
            if not is_face(img):
                os.remove(img_path)
                print('delete..............................')


if __name__ == '__main__':
    patch_name = 'patch_12'
    generate_csv_for_verification_model()
    # generate_patch(patch_name)
    # generate_csv_for_deepid(patch_name)
    # generate_pickle_for_deepid(patch_name)
