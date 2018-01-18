import tensorflow as tf
import numpy as np
import cv2
import face_alignment
import Net
import time
from scipy.spatial.distance import cosine
from sklearn import preprocessing


def feature_extract(img):
    landmarkList = face_alignment.alignment(img)
    patch1 = cv2.resize(img, (31, 31))
    patch2, patch3 = face_alignment.getRois(img, landmarkList[0], landmarkList[1])

    patch2 = cv2.resize(patch2, (31, 31))
    patch3 = cv2.resize(patch3, (31, 31))
    patch4 = face_alignment.getRois(img, landmarkList[2])
    patch4 = cv2.resize(patch4, (31, 31))
    patch5, patch6 = face_alignment.getRois(img, landmarkList[3], landmarkList[4])
    patch5 = cv2.resize(patch5, (31, 31))
    patch6 = cv2.resize(patch6, (31, 31))
    patch7 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
    patch8 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)
    patch9 = cv2.cvtColor(patch3, cv2.COLOR_BGR2GRAY)
    patch10 = cv2.cvtColor(patch4, cv2.COLOR_BGR2GRAY)
    patch11 = cv2.cvtColor(patch5, cv2.COLOR_BGR2GRAY)
    patch12 = cv2.cvtColor(patch6, cv2.COLOR_BGR2GRAY)
    patch1 = patch1.reshape(1, 31, 31, 3)
    patch2 = patch2.reshape(1, 31, 31, 3)
    patch3 = patch3.reshape(1, 31, 31, 3)
    patch4 = patch4.reshape(1, 31, 31, 3)
    patch5 = patch5.reshape(1, 31, 31, 3)
    patch6 = patch6.reshape(1, 31, 31, 3)
    patch7 = patch7.reshape(1, 31, 31, 1)
    patch8 = patch8.reshape(1, 31, 31, 1)
    patch9 = patch9.reshape(1, 31, 31, 1)
    patch10 = patch10.reshape(1, 31, 31, 1)
    patch11 = patch11.reshape(1, 31, 31, 1)
    patch12 = patch12.reshape(1, 31, 31, 1)
    tf.reset_default_graph()
    net = Net.CNN('rgb')
    feature1 = net.output_deepid(patch1, 'checkpoint/patch_1.ckpt')
    feature2 = net.output_deepid(patch2, 'checkpoint/patch_2.ckpt')
    feature3 = net.output_deepid(patch3, 'checkpoint/patch_3.ckpt')
    feature4 = net.output_deepid(patch4, 'checkpoint/patch_4.ckpt')
    feature5 = net.output_deepid(patch5, 'checkpoint/patch_5.ckpt')
    feature6 = net.output_deepid(patch6, 'checkpoint/patch_6.ckpt')
    tf.reset_default_graph()
    net = Net.CNN('gray')
    feature7 = net.output_deepid(patch7, 'checkpoint/patch_7.ckpt')
    feature8 = net.output_deepid(patch8, 'checkpoint/patch_8.ckpt')
    feature9 = net.output_deepid(patch9, 'checkpoint/patch_9.ckpt')
    feature10 = net.output_deepid(patch10, 'checkpoint/patch_10.ckpt')
    feature11 = net.output_deepid(patch11, 'checkpoint/patch_11.ckpt')
    feature12 = net.output_deepid(patch12, 'checkpoint/patch_12.ckpt')

    final_feature = np.concatenate((feature1, feature2, feature3, feature4, feature5, feature6,
                                    feature7, feature8, feature9, feature10, feature11, feature12),
                                   axis=0)
    final_feature = preprocessing.scale(final_feature)  # 归一化
    return final_feature


def getSim(img1, img2):
    if type(img1) == str:
        img1 = cv2.imread(img1)
    if type(img2) == str:
        img2 = cv2.imread(img2)
    feature1 = feature_extract(img1)
    feature2 = feature_extract(img2)
    sim = 1 - cosine(feature1, feature2)
    return sim


if __name__ == '__main__':
    img1='F:/Project/LBPface/224pics_nolens/69A1.jpg'
    img2='F:/Project/LBPface/224pics_nolens/69A4.jpg'
    sim=getSim(img1,img2)
    print(sim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()