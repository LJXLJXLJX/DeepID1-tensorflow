'''
CNN通用模版 有十个子类 分别对应十个patch的输入
这十个子类又分别有三个子类 分别对应三种尺度
这三个子类又有
'''

import numpy as np
import tensorflow as tf
import time
import os
import cv2
import dataSetPreProcess



# 高斯截断初始化w
def weight_variable(shape):
    with tf.name_scope('weights'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


# 全零初始化b
def bias_variable(shape):
    with tf.name_scope('biases'):
        return tf.Variable(tf.zeros(shape))


# 做Wx+b运算
def Wx_plus_b(weights, x, biases):
    with tf.name_scope('Wx_plus_b'):
        return tf.matmul(x, weights) + biases


# 全连接层
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = Wx_plus_b(weights, input_tensor, biases)
        if act != None:
            # 激活
            activations = act(preactivate, name='activation')
            return activations
        else:
            # 未激活
            return preactivate


# 卷积池化层
def conv_pool_layer(x, w_shape, b_shape, layer_name, act=tf.nn.relu, only_conv=False):
    with tf.name_scope(layer_name):
        W = weight_variable(w_shape)
        b = bias_variable(b_shape)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='conv2d')
        h = conv + b
        relu = act(h, name='relu')
        if only_conv == True:
            return relu
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max-pooling')
        return pool


# 准确率表示
def accuracy(y, y_):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # 每个batch1024 y y_都是 1024*classnum的
            # equal每次比一个得到（1，0，1，0，...，1）这种 1024维向量 均值即为该batch　的 accuracy
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy


# 训练 AdamOptimizer
def train_step(loss):
    with tf.name_scope('train'):
        return tf.train.AdamOptimizer(1e-4).minimize(loss)


# 读取tfrecords文件
def read_single_sample_from_tfrecords(tfrecords_path):
    fileNameQue = tf.train.string_input_producer([tfrecords_path])
    reader = tf.TFRecordReader()
    key, value = reader.read(fileNameQue)
    features = tf.parse_single_example(value, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img': tf.FixedLenFeature([], tf.string)
    })
    img = tf.decode_raw(features['img'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    img=tf.reshape(img,[31,31,3])
    label=tf.one_hot(label,depth=class_num)

    return img, label

# 得到一次训练的batch
# def get_batch(start, step):
#     img_batch = []
#     label_batch = []
#     with open('data/dataset_for_deepid.csv', 'r') as f:
#         lines = f.readlines()
#         line_num = len(lines)
#     end = (start + step) % line_num
#     count = 0
#     # 圈内
#     if start < end:
#         sub_lines = lines[start:end]
#         sub_len = len(sub_lines)
#         for line in sub_lines:
#             count += 1
#             print(' get_batch:', count, '/', sub_len)
#             line = line.split(' ')
#             img_path = line[0]
#             label = int(line[-1])
#             label = (np.arange(class_num) == label).astype(np.float32)  # 转成onehot
#             img = cv2.imread(img_path)
#             img_batch.append(img)
#             label_batch.append(label)
#
#         return np.asarray(img_batch, dtype='uint8'), np.asarray(label_batch, dtype='float32'), end
#     # 圈外
#     else:
#         sub_lines = lines[start:] + lines[0:end]
#         sub_len = len(sub_lines)
#         for line in sub_lines:
#             count += 1
#             print(' get_batch:', count, '/', sub_len)
#             line = line.split(' ')
#             img_path = line[0]
#             label = int(line[-1])
#             label = (np.arange(class_num) == label).astype(np.float32)  # 转成onehot
#             img = cv2.imread(img_path)
#             img_batch.append(img)
#             label_batch.append(label)
#         return np.asarray(img_batch, dtype='float32'), np.asarray(label_batch, dtype='float32'), end


# 构建输入为彩色图像网络结构

class CNN():
    def __init__(self):
        with tf.name_scope('input'):
            self.h0 = tf.placeholder(tf.float32, [None, 31, 31, 3], name='x')  # 输入
            self.y_ = tf.placeholder(tf.float32, [None, class_num], name='y')  # 分类结果  onehot码

        self.h1 = conv_pool_layer(self.h0, [4, 4, 3, 20], [20], 'Conv_layer_1')
        self.h2 = conv_pool_layer(self.h1, [3, 3, 20, 40], [40], 'Conv_layer_2')
        self.h3 = conv_pool_layer(self.h2, [3, 3, 40, 60], [60], 'Conv_layer_3')
        self.h4 = conv_pool_layer(self.h3, [2, 2, 60, 80], [80], 'Conv_layer_4', only_conv=True)

        # Deepid层，与最后两个卷积层相连接
        with tf.name_scope('DeepID1'):
            self.h3r = tf.reshape(self.h3, [-1, 2 * 2 * 60])
            self.h4r = tf.reshape(self.h4, [-1, 1 * 1 * 80])
            self.W1 = weight_variable([2 * 2 * 60, 160])
            self.W2 = weight_variable([1 * 1 * 80, 160])
            self.b = bias_variable([160])
            self.h = tf.matmul(self.h3r, self.W1) + tf.matmul(self.h4r, self.W2) + self.b
            self.h5 = tf.nn.relu(self.h)

        # loss softmax 分类器
        with tf.name_scope('loss'):
            self.y = nn_layer(self.h5, 160, class_num, 'nn_layer', act=None)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
            tf.summary.scalar('loss', self.loss)

        self.accuracy = accuracy(self.y, self.y_)
        self.train_step = train_step(self.loss)

        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    # 以下训练函数分别训练每个patch的正反 以及其三个不同尺度
    # 此处并未完全按照论文取patch 嫌麻烦 少取了四个rectangle的 翻转输入也省了
    # 前一个数字为选取部分（1，2，3，4，5，6分别对应整张脸 右眼 左眼 鼻尖 左嘴角 右嘴角）
    # 后一个数字表示三种尺度 分别是原始区域 缩小至3/4 1/2 文章没给出具体数字 就这样吧
    def train_patch(self,patch_name):
        if type(patch_name)!=str:
            raise Exception('patch_name should be a str')
        logdir = 'log'

        # 加上以下语句会删除上次的log
        # if tf.gfile.Exists(logdir):
        #     tf.gfile.DeleteRecursively(logdir)
        # tf.gfile.MakeDirs(logdir)

        img, label = read_single_sample_from_tfrecords('data/'+patch_name+'.tfrecords')


        with tf.Session() as sess:
            imgs, labels = tf.train.shuffle_batch([img, label],
                                                            batch_size=1024,
                                                            capacity=20000,
                                                            min_after_dequeue=10000,
                                                            num_threads=10)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            #恢复上次的训练进度
            self.saver.restore(sess,'checkpoint/patch_1_05000.ckpt')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            train_writer = tf.summary.FileWriter(logdir + '/'+patch_name, sess.graph)
            for i in range(50001):
                print('training...',i, '/50000')

                img_batch,label_batch=sess.run([imgs,labels])



                # label_batch=(np.arange(class_num)==label_batch[:,None]).astype(np.float32)
                # label = (np.arange(class_num) == np.asarray([label])).astype(np.float32)

                summary, _ = sess.run([self.merged, self.train_step], {self.h0: img_batch, self.y_: label_batch})

                train_writer.add_summary(summary, i)
                if i % 500 == 0 and i != 0:
                    self.saver.save(sess, 'checkpoint/'+patch_name)
            coord.request_stop()
            coord.join(threads)

# 构建输入为灰度图像的网络结构（继承自CNN）
class grayCNN(CNN):
    def __int__(self):
        super().__init__()
        # 灰度图像的输入深度和第一个卷积层要改 其余的结构与超类彩色一致
        with tf.name_scope('input'):
            self.h0 = tf.placeholder(tf.float32, [None, 31, 31, 1], name='x')  # 输入
            self.y_ = tf.placeholder(tf.float32, [None, class_num], name='y')  # 分类结果  onehot码

        self.h1 = conv_pool_layer(self.h0, [4, 4, 1, 20], [20], 'Conv_layer_1')


if __name__ == '__main__':
    processor = dataSetPreProcess.DataSetPreProcessor(0)
    class_num = processor.people_num_for_deepid
    cnn = CNN()
    cnn.train_patch('patch_1')
