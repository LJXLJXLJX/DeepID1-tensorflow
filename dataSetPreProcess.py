'''
数据集预处理 分割 打标签
'''
import tensorflow as tf
import os
import random
import numpy as np
import cv2
import face_alignment
import pickle
import platform

# 如果当前调试系统是windows（本地），切换到相应的路径
if platform.system() == 'Windows':
    dataset_path = './DataSet/images/'
    patch_1_path = './DataSet/patch_1/'
    patch_2_path = './DataSet/patch_2/'
    patch_3_path = './DataSet/patch_3/'
    patch_4_path = './DataSet/patch_4/'
    patch_5_path = './DataSet/patch_5/'
    patch_6_path = './DataSet/patch_6/'
    patch_7_path = './DataSet/patch_7/'
    patch_8_path = './DataSet/patch_8/'
    patch_9_path = './DataSet/patch_9/'
    patch_10_path = './DataSet/patch_10/'
    patch_11_path = './DataSet/patch_11/'
    patch_12_path = './DataSet/patch_12/'

# 如果当前调试系统是Linux（服务器），切换到相应路径
elif platform.system() == 'Linux':
    userRoot = os.environ['HOME']
    dataset_path = userRoot + '/DataSet/images/'
    patch_1_path = userRoot + '/DataSet/patch_1/'
    patch_2_path = userRoot + '/DataSet/patch_2/'
    patch_3_path = userRoot + '/DataSet/patch_3/'
    patch_4_path = userRoot + '/DataSet/patch_4/'
    patch_5_path = userRoot + '/DataSet/patch_5/'
    patch_6_path = userRoot + '/DataSet/patch_6/'
    patch_7_path = userRoot + '/DataSet/patch_7/'
    patch_8_path = userRoot + '/DataSet/patch_8/'
    patch_9_path = userRoot + '/DataSet/patch_9/'
    patch_10_path = userRoot + '/DataSet/patch_10/'
    patch_11_path = userRoot + '/DataSet/patch_11/'
    patch_12_path = userRoot + '/DataSet/patch_12/'


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


class DataSetPreProcessor():
    def __init__(self, patchID):
        if patchID not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            raise Exception('patch号错误')
        self.patch_to_process = patchID
        self.update_dataset_imformation()

    # 更新数据集信息
    def update_dataset_imformation(self):
        if not os.path.exists('data'):
            os.mkdir('data')
        if not os.path.exists(patch_1_path):
            os.mkdir(patch_1_path)
        if not os.path.exists(patch_2_path):
            os.mkdir(patch_2_path)
        if not os.path.exists(patch_3_path):
            os.mkdir(patch_3_path)
        if not os.path.exists(patch_4_path):
            os.mkdir(patch_4_path)
        if not os.path.exists(patch_5_path):
            os.mkdir(patch_5_path)
        if not os.path.exists(patch_6_path):
            os.mkdir(patch_6_path)
        if not os.path.exists(patch_6_path):
            os.mkdir(patch_6_path)
        if not os.path.exists(patch_6_path):
            os.mkdir(patch_7_path)
        if not os.path.exists(patch_6_path):
            os.mkdir(patch_8_path)
        if not os.path.exists(patch_6_path):
            os.mkdir(patch_9_path)
        if not os.path.exists(patch_6_path):
            os.mkdir(patch_10_path)
        if not os.path.exists(patch_6_path):
            os.mkdir(patch_11_path)
        if not os.path.exists(patch_6_path):
            os.mkdir(patch_12_path)

        if self.patch_to_process == 0:
            self.people_list = os.listdir(dataset_path)  # 所有人的列表
        elif self.patch_to_process == 1:
            self.people_list = os.listdir(patch_1_path)
        elif self.patch_to_process == 2:
            self.people_list = os.listdir(patch_2_path)
        elif self.patch_to_process == 3:
            self.people_list = os.listdir(patch_3_path)
        elif self.patch_to_process == 4:
            self.people_list = os.listdir(patch_4_path)
        elif self.patch_to_process == 5:
            self.people_list = os.listdir(patch_5_path)
        elif self.patch_to_process == 6:
            self.people_list = os.listdir(patch_6_path)
        elif self.patch_to_process == 7:
            self.people_list = os.listdir(patch_7_path)
        elif self.patch_to_process == 8:
            self.people_list = os.listdir(patch_8_path)
        elif self.patch_to_process == 9:
            self.people_list = os.listdir(patch_9_path)
        elif self.patch_to_process == 10:
            self.people_list = os.listdir(patch_10_path)
        elif self.patch_to_process == 11:
            self.people_list = os.listdir(patch_11_path)
        elif self.patch_to_process == 12:
            self.people_list = os.listdir(patch_12_path)

        # 以下list的每个元素为tuple
        self.people_list = list(enumerate(self.people_list))  # 每个元素转成tuple 相当于打标签
        self.people_num = len(self.people_list)  # 人的数量
        self.people_num_for_deepid = self.people_num * 4 // 5  # 用于训练deepid的人数
        self.people_num_for_verification_model = self.people_num - self.people_num_for_deepid  # 用于验证模型的人数
        self.people_list_for_deepid = self.people_list[0:self.people_num_for_deepid]  # 用于训练deepid的人
        self.people_list_for_verification_model = self.people_list[self.people_num_for_deepid:]  # 用于训练验证模型的人

    # 获取一个人的所有图片（文件路径）列表 乱序之后输出该列表
    def get_pics_for_one_people(self, people_dir):
        if people_dir[-1] != ('/' or '\\'):
            people_dir += '/'
        people_imgs = []
        for img in os.listdir(people_dir):
            img_path = people_dir + img
            people_imgs.append(img_path)
        random.shuffle(people_imgs)
        return people_imgs

    def generate_patch(self):
        # 鼻尖一点
        def getRoi(img, landmark):
            # 正方形区域的半边长5
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

        # 眼两点，嘴两点
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

        if self.patch_to_process == 0:
            raise Exception("you can't generate origin dataset")
        temp_0_processor = DataSetPreProcessor(0)

        people_count = 0
        people_num = temp_0_processor.people_num_for_deepid
        for people in temp_0_processor.people_list_for_deepid:
            people = people[1]
            people_count += 1
            print('Generating...', people_count, '/', people_num)
            if self.patch_to_process == 1:
                if not os.path.exists(patch_1_path + people):
                    os.mkdir(patch_1_path + people)
                for img_name in os.listdir(dataset_path + people):
                    origin_img_path = os.path.join(dataset_path + people, img_name)
                    dst_img_path = os.path.join(patch_1_path + people, img_name)
                    img = cv2.imread(origin_img_path)
                    img = cv2.resize(img, (31, 31))
                    cv2.imwrite(dst_img_path, img)
            # 两只眼睛
            if self.patch_to_process in [2, 3]:
                if not os.path.exists(patch_2_path + people):
                    os.mkdir(patch_2_path + people)
                if not os.path.exists(patch_3_path + people):
                    os.mkdir(patch_3_path + people)
                for img_name in os.listdir(dataset_path + people):
                    origin_img_path = os.path.join(dataset_path + people, img_name)
                    left_dst_img_path = os.path.join(patch_2_path + people, img_name)
                    right_dst_img_path = os.path.join(patch_3_path + people, img_name)
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
            if self.patch_to_process == 4:
                if not os.path.exists(patch_4_path + people):
                    os.mkdir(patch_4_path + people)
                for img_name in os.listdir(dataset_path + people):
                    origin_img_path = os.path.join(dataset_path + people, img_name)
                    dst_img_path = os.path.join(patch_4_path + people, img_name)
                    img = cv2.imread(origin_img_path)
                    landmark = face_alignment.alignment(img)[2]
                    try:
                        img = getRois(img, landmark)
                        img = cv2.resize(img, (31, 31))
                        cv2.imwrite(dst_img_path, img)
                    except:
                        print('Something wrong happens ...')
            # 两个嘴角
            if self.patch_to_process in [5, 6]:
                if not os.path.exists(patch_5_path + people):
                    os.mkdir(patch_5_path + people)
                if not os.path.exists(patch_6_path + people):
                    os.mkdir(patch_6_path + people)
                for img_name in os.listdir(dataset_path + people):
                    origin_img_path = os.path.join(dataset_path + people, img_name)
                    left_dst_img_path = os.path.join(patch_5_path + people, img_name)
                    right_dst_img_path = os.path.join(patch_6_path + people, img_name)
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

    # 构建用于训练deepid的csv
    def generate_csv_for_deepid(self):
        self.update_dataset_imformation()
        dataset_for_deepid_train = []
        dataset_for_deepid_valid = []
        for label, people in self.people_list_for_deepid:
            if self.patch_to_process == 0:
                people_imgs = self.get_pics_for_one_people(dataset_path + people)
            elif self.patch_to_process == 1:
                people_imgs = self.get_pics_for_one_people(patch_1_path + people)
            elif self.patch_to_process == 2:
                people_imgs = self.get_pics_for_one_people(patch_2_path + people)
            elif self.patch_to_process == 3:
                people_imgs = self.get_pics_for_one_people(patch_3_path + people)
            elif self.patch_to_process == 4:
                people_imgs = self.get_pics_for_one_people(patch_4_path + people)
            elif self.patch_to_process == 5:
                people_imgs = self.get_pics_for_one_people(patch_5_path + people)
            elif self.patch_to_process == 6:
                people_imgs = self.get_pics_for_one_people(patch_6_path + people)
            elif self.patch_to_process == 7:
                people_imgs = self.get_pics_for_one_people(patch_7_path + people)
            elif self.patch_to_process == 8:
                people_imgs = self.get_pics_for_one_people(patch_8_path + people)
            elif self.patch_to_process == 9:
                people_imgs = self.get_pics_for_one_people(patch_9_path + people)
            elif self.patch_to_process == 10:
                people_imgs = self.get_pics_for_one_people(patch_10_path + people)
            elif self.patch_to_process == 11:
                people_imgs = self.get_pics_for_one_people(patch_11_path + people)
            elif self.patch_to_process == 12:
                people_imgs = self.get_pics_for_one_people(patch_12_path + people)

            people_imgs_for_train = people_imgs[0:9 * len(people_imgs) // 10]  # 取其中9/10作为训练集
            people_imgs_for_valid = people_imgs[9 * len(people_imgs) // 10:]  # 其中1/10作为验证（防止过拟合）
            dataset_for_deepid_train += zip(people_imgs_for_train, [str(label)] * len(people_imgs_for_train))
            dataset_for_deepid_valid += zip(people_imgs_for_valid, [str(label)] * len(people_imgs_for_valid))

        random.shuffle(dataset_for_deepid_train)
        random.shuffle(dataset_for_deepid_valid)
        if self.patch_to_process == 0:
            with open('data/dataset_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)

            with open('data/dataset_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)

        elif self.patch_to_process == 1:
            with open('data/patch_1_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_1_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 2:
            with open('data/patch_2_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_2_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 3:
            with open('data/patch_3_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_3_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 4:
            with open('data/patch_4_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_4_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 5:
            with open('data/patch_5_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_5_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 6:
            with open('data/patch_6_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_6_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 7:
            with open('data/patch_7_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_7_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 8:
            with open('data/patch_8_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_8_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 9:
            with open('data/patch_9_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_9_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 10:
            with open('data/patch_10_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_10_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 11:
            with open('data/patch_11_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_11_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 12:
            with open('data/patch_12_for_deepid_train.csv', 'w') as f:
                for item in dataset_for_deepid_train:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
            with open('data/patch_12_for_deepid_valid.csv', 'w') as f:
                for item in dataset_for_deepid_valid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)

        self.update_dataset_imformation()

    # 构建用于训练验证模型的csv  （问题很大）要改
    def generate_csv_for_verification_model(self):
        self.update_dataset_imformation()
        dataset_for_verification_model = []
        for label, people in self.people_list_for_verification_model:
            if self.patch_to_process == 0:
                people_imgs = self.get_pics_for_one_people(dataset_path + people)
            elif self.patch_to_process == 1:
                people_imgs = self.get_pics_for_one_people(patch_1_path + people)
            elif self.patch_to_process == 2:
                people_imgs = self.get_pics_for_one_people(patch_2_path + people)
            elif self.patch_to_process == 3:
                people_imgs = self.get_pics_for_one_people(patch_3_path + people)
            elif self.patch_to_process == 4:
                people_imgs = self.get_pics_for_one_people(patch_4_path + people)
            elif self.patch_to_process == 5:
                people_imgs = self.get_pics_for_one_people(patch_5_path + people)
            elif self.patch_to_process == 6:
                people_imgs = self.get_pics_for_one_people(patch_6_path + people)
            elif self.patch_to_process == 7:
                people_imgs = self.get_pics_for_one_people(patch_7_path + people)
            elif self.patch_to_process == 8:
                people_imgs = self.get_pics_for_one_people(patch_8_path + people)
            elif self.patch_to_process == 9:
                people_imgs = self.get_pics_for_one_people(patch_9_path + people)
            elif self.patch_to_process == 10:
                people_imgs = self.get_pics_for_one_people(patch_10_path + people)
            elif self.patch_to_process == 11:
                people_imgs = self.get_pics_for_one_people(patch_11_path + people)
            elif self.patch_to_process == 12:
                people_imgs = self.get_pics_for_one_people(patch_12_path + people)
            dataset_for_verification_model += zip(people_imgs, [str(label)] * len(people_imgs))

        random.shuffle(dataset_for_verification_model)
        if self.patch_to_process == 0:
            with open('data/dataset_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 1:
            with open('data/patch_1_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 2:
            with open('data/patch_2_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 3:
            with open('data/patch_3_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 4:
            with open('data/patch_4_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 5:
            with open('data/patch_5_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 6:
            with open('data/patch_6_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 7:
            with open('data/patch_7_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 8:
            with open('data/patch_8_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 9:
            with open('data/patch_9_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 10:
            with open('data/patch_10_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 11:
            with open('data/patch_11_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 12:
            with open('data/patch_12_for_verification_model.csv', 'w') as f:
                for item in dataset_for_verification_model:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)

        self.update_dataset_imformation()

    # 生成用于训练deepid的tfrecords
    def generate_recorder_for_deepid(self):
        def generate_recorder_for_a_patch(TFwriter, pairList):
            count = 0
            num = len(pairList)
            for img_path, label in pairList:
                count += 1
                print('recording...', count, '/', num)
                img = cv2.imread(img_path)
                imgRaw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                    'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgRaw]))
                }))
                TFwriter.write(example.SerializeToString())
            TFwriter.close()

        if self.patch_to_process not in range(1, 7):
            raise Exception('The patch should be 1 to 6')
        if self.patch_to_process == 1:
            TFwriter = tf.python_io.TFRecordWriter("data/patch_1_train.tfrecords")
            img_pathList, labelList = read_csv('data/patch_1_for_deepid_train.csv')
            pairList = list(zip(img_pathList, labelList))
            generate_recorder_for_a_patch(TFwriter, pairList)
            TFwriter = tf.python_io.TFRecordWriter("data/patch_1_valid.tfrecords")
            img_pathList, labelList = read_csv('data/patch_1_for_deepid_valid.csv')
            pairList = list(zip(img_pathList, labelList))
            generate_recorder_for_a_patch(TFwriter, pairList)

    # 生成用于训练deepid的pickle（和tfrecorder 二选一）
    def generate_pickle_for_deepid(self):
        # 生成train集的pickle
        img_pathList = []
        labelList = []
        if self.patch_to_process == 1:
            img_pathList, labelList = read_csv('data/patch_1_for_deepid_train.csv')
        elif self.patch_to_process == 2:
            img_pathList, labelList = read_csv('data/patch_2_for_deepid_train.csv')
        elif self.patch_to_process == 3:
            img_pathList, labelList = read_csv('data/patch_3_for_deepid_train.csv')
        elif self.patch_to_process == 4:
            img_pathList, labelList = read_csv('data/patch_4_for_deepid_train.csv')
        elif self.patch_to_process == 5:
            img_pathList, labelList = read_csv('data/patch_5_for_deepid_train.csv')
        elif self.patch_to_process == 6:
            img_pathList, labelList = read_csv('data/patch_6_for_deepid_train.csv')
        elif self.patch_to_process == 7:
            img_pathList, labelList = read_csv('data/patch_7_for_deepid_train.csv')
        elif self.patch_to_process == 8:
            img_pathList, labelList = read_csv('data/patch_8_for_deepid_train.csv')
        elif self.patch_to_process == 9:
            img_pathList, labelList = read_csv('data/patch_9_for_deepid_train.csv')
        elif self.patch_to_process == 10:
            img_pathList, labelList = read_csv('data/patch_10_for_deepid_train.csv')
        elif self.patch_to_process == 11:
            img_pathList, labelList = read_csv('data/patch_11_for_deepid_train.csv')
        elif self.patch_to_process == 12:
            img_pathList, labelList = read_csv('data/patch_12_for_deepid_train.csv')

        count = 0
        num = len(img_pathList)
        imgList = []
        for img_path in img_pathList:
            count += 1
            print('reading...', count, '/', num)
            img = cv2.imread(img_path, -1)
            img.resize(31, 31, 1)
            imgList.append(img)
        imgArr = np.asarray(imgList, dtype='uint8')
        labelArr = np.asarray(labelList, dtype='float32')

        if self.patch_to_process == 1:
            with open('data/patch_1_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 2:
            with open('data/patch_2_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 3:
            with open('data/patch_3_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 4:
            with open('data/patch_4_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 5:
            with open('data/patch_5_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 6:
            with open('data/patch_6_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 7:
            with open('data/patch_7_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 8:
            with open('data/patch_8_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 9:
            with open('data/patch_9_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 10:
            with open('data/patch_10_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 11:
            with open('data/patch_11_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 12:
            with open('data/patch_12_train.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)

        # 生成valid集的pickle
        if self.patch_to_process == 1:
            img_pathList, labelList = read_csv('data/patch_1_for_deepid_valid.csv')
        elif self.patch_to_process == 2:
            img_pathList, labelList = read_csv('data/patch_2_for_deepid_valid.csv')
        elif self.patch_to_process == 3:
            img_pathList, labelList = read_csv('data/patch_3_for_deepid_valid.csv')
        elif self.patch_to_process == 4:
            img_pathList, labelList = read_csv('data/patch_4_for_deepid_valid.csv')
        elif self.patch_to_process == 5:
            img_pathList, labelList = read_csv('data/patch_5_for_deepid_valid.csv')
        elif self.patch_to_process == 6:
            img_pathList, labelList = read_csv('data/patch_6_for_deepid_valid.csv')
        elif self.patch_to_process == 7:
            img_pathList, labelList = read_csv('data/patch_7_for_deepid_valid.csv')
        elif self.patch_to_process == 8:
            img_pathList, labelList = read_csv('data/patch_8_for_deepid_valid.csv')
        elif self.patch_to_process == 9:
            img_pathList, labelList = read_csv('data/patch_9_for_deepid_valid.csv')
        elif self.patch_to_process == 10:
            img_pathList, labelList = read_csv('data/patch_10_for_deepid_valid.csv')
        elif self.patch_to_process == 11:
            img_pathList, labelList = read_csv('data/patch_11_for_deepid_valid.csv')
        elif self.patch_to_process == 12:
            img_pathList, labelList = read_csv('data/patch_12_for_deepid_valid.csv')

        count = 0
        num = len(img_pathList)

        imgList = []
        for img_path in img_pathList:
            count += 1
            print('reading...', count, '/', num)
            img = cv2.imread(img_path, -1)
            img.resize(31, 31, 1)
            imgList.append(img)
        imgArr = np.asarray(imgList, dtype='uint8')
        labelArr = np.asarray(labelList, dtype='float32')

        if self.patch_to_process == 1:
            with open('data/patch_1_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 2:
            with open('data/patch_2_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 3:
            with open('data/patch_3_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 4:
            with open('data/patch_4_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 5:
            with open('data/patch_5_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 6:
            with open('data/patch_6_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 7:
            with open('data/patch_7_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 8:
            with open('data/patch_8_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 9:
            with open('data/patch_9_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 10:
            with open('data/patch_10_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 11:
            with open('data/patch_5_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)
        elif self.patch_to_process == 12:
            with open('data/patch_6_valid.pkl', 'wb') as f:
                pickle.dump(imgArr, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelArr, f, pickle.HIGHEST_PROTOCOL)

    # 判断输入图像是否有且只有一张人脸 （毕竟数据集里混进了很多奇怪的东西 输入之前要筛选一下）
    def is_face(self, img):
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

    # 清洗数据 去掉混进来奇怪的东西（非人脸 非rgb）
    def wash_data(self):
        people_count = 0
        img_count = 0
        people_num = len(os.listdir(dataset_path))
        if self.patch_to_process != 0:
            raise Exception('Only origin data can be washed')
        for people in os.listdir(dataset_path):
            people_count += 1
            for img_name in os.listdir(os.path.join(dataset_path, people)):
                img_count += 1
                print('Washing...', people_count, '/', people_num, ':', img_count)

                img_path = os.path.join(dataset_path, people, img_name)
                img = cv2.imread(img_path)
                if not self.is_face(img):
                    os.remove(img_path)
                    print('delete..............................')

        self.generate_csv_for_deepid()
        self.generate_csv_for_verification_model()


if __name__ == '__main__':
    processor = DataSetPreProcessor(7)
    # processor.generate_patch()
    processor.generate_csv_for_deepid()
    processor.generate_csv_for_verification_model()
    # processor.generate_recorder_for_deepid()
    # processor.wash_data()
    processor.generate_pickle_for_deepid()
