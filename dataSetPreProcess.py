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
        if patchID not in [0, 1, 2, 3, 4, 5, 6]:
            raise Exception('patch号错误')
        self.patch_to_process = patchID
        self.update_dataset_imformation()

    # 更新数据集信息
    def update_dataset_imformation(self):
        if not os.path.exists('data'):
            os.mkdir('data')
        if not os.path.exists(patch_1_path):
            os.mkdir(patch_1_path)
        if not os.path.exists(patch_1_path):
            os.mkdir(patch_2_path)
        if not os.path.exists(patch_1_path):
            os.mkdir(patch_3_path)
        if not os.path.exists(patch_1_path):
            os.mkdir(patch_4_path)
        if not os.path.exists(patch_1_path):
            os.mkdir(patch_5_path)
        if not os.path.exists(patch_1_path):
            os.mkdir(patch_6_path)

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
        if self.patch_to_process == 0:
            raise Exception("you can't generate origin dataset")
        temp_0_processor = DataSetPreProcessor(0)
        if self.patch_to_process == 1:
            people_count = 0
            people_num = temp_0_processor.people_num_for_deepid
            for people in temp_0_processor.people_list_for_deepid:
                people = people[1]
                people_count += 1
                print('Generating...', people_count, '/', people_num)
                if not os.path.exists(patch_1_path + people):
                    os.mkdir(patch_1_path + people)
                    for img_name in os.listdir(dataset_path + people):
                        origin_img_path = os.path.join(dataset_path + people, img_name)
                        dst_img_path = os.path.join(patch_1_path + people, img_name)
                        img = cv2.imread(origin_img_path)
                        img = cv2.resize(img, (31, 31))
                        cv2.imwrite(dst_img_path, img)

    # 构建用于训练deepid的csv
    def generate_csv_for_deepid(self):
        self.update_dataset_imformation()
        dataset_for_deepid = []
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

            dataset_for_deepid += zip(people_imgs, [str(label)] * len(people_imgs))
        random.shuffle(dataset_for_deepid)
        if self.patch_to_process == 0:
            with open('data/dataset_for_deepid.csv', 'w') as f:
                for item in dataset_for_deepid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 1:
            with open('data/patch_1_for_deepid.csv', 'w') as f:
                for item in dataset_for_deepid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 2:
            with open('data/patch_2_for_deepid.csv', 'w') as f:
                for item in dataset_for_deepid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 3:
            with open('data/patch_3_for_deepid.csv', 'w') as f:
                for item in dataset_for_deepid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 4:
            with open('data/patch_4_for_deepid.csv', 'w') as f:
                for item in dataset_for_deepid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 5:
            with open('data/patch_5_for_deepid.csv', 'w') as f:
                for item in dataset_for_deepid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        elif self.patch_to_process == 6:
            with open('data/patch_6_for_deepid.csv', 'w') as f:
                for item in dataset_for_deepid:
                    print(str(item[0]).strip(), ' ', str(item[1]).strip(), file=f)
        self.update_dataset_imformation()

    # 构建用于训练验证模型的csv
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
            TFwriter = tf.python_io.TFRecordWriter("data/patch_1.tfrecords")
            img_pathList, labelList = read_csv('data/patch_1_for_deepid.csv')
            pairList = list(zip(img_pathList, labelList))
            generate_recorder_for_a_patch(TFwriter, pairList)

    # 生成用于训练deepid的pickle（和tfrecorder 二选一）
    def generate_pickle_for_deepid(self):
        imgList = []
        if self.patch_to_process == 1:
            img_pathList, labelList = read_csv('data/patch_1_for_deepid.csv')
            for img_path in img_pathList:
                img = cv2.imread(img_path)
                imgList.append(img)
            with open('data/patch_1.pkl', 'wb') as f:
                pickle.dump(imgList, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(labelList, f, pickle.HIGHEST_PROTOCOL)

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

    # 清洗数据 去掉混进来奇怪的东西（非脸 非rgb）
    def wash_data(self):
        delete_count = 0
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
                    delete_count += 1
                    print('delete..............................')

        self.generate_csv_for_deepid()
        self.generate_csv_for_verification_model()


if __name__ == '__main__':
    processor = DataSetPreProcessor(1)
    # processor.generate_patch()
    processor.generate_csv_for_deepid()
    processor.generate_csv_for_verification_model()
    processor.generate_recorder_for_deepid()
    # processor.wash_data()
