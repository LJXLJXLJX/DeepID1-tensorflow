import tensorflow as tf
import numpy as np
import cv2
import face_alignment
import Net
import time
from scipy.spatial.distance import cosine
from sklearn import preprocessing
import threading
import queue


def feature_extract(img):
    landmarkList = face_alignment.alignment(img)
    q = queue.Queue()

    def thread_get_patch_1_7():
        patch1 = cv2.resize(img, (31, 31))
        patch7 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
        patch1 = patch1.reshape(1, 31, 31, 3)
        patch7 = patch7.reshape(1, 31, 31, 1)
        q.put((patch1, patch7, '17'))

    def thread_get_patch_2_3_8_9():
        patch2, patch3 = face_alignment.getRois(img, landmarkList[0], landmarkList[1])
        patch2 = cv2.resize(patch2, (31, 31))
        patch3 = cv2.resize(patch3, (31, 31))
        patch8 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)
        patch9 = cv2.cvtColor(patch3, cv2.COLOR_BGR2GRAY)
        patch2 = patch2.reshape(1, 31, 31, 3)
        patch3 = patch3.reshape(1, 31, 31, 3)
        patch8 = patch8.reshape(1, 31, 31, 1)
        patch9 = patch9.reshape(1, 31, 31, 1)
        q.put((patch2, patch3, patch8, patch9, '2389'))

    def thread_get_patch_4_10():
        patch4 = face_alignment.getRois(img, landmarkList[2])
        patch4 = cv2.resize(patch4, (31, 31))
        patch10 = cv2.cvtColor(patch4, cv2.COLOR_BGR2GRAY)
        patch4 = patch4.reshape(1, 31, 31, 3)
        patch10 = patch10.reshape(1, 31, 31, 1)
        q.put((patch4, patch10, '410'))

    def thread_get_patch_5_6_11_12():
        patch5, patch6 = face_alignment.getRois(img, landmarkList[3], landmarkList[4])
        patch5 = cv2.resize(patch5, (31, 31))
        patch6 = cv2.resize(patch6, (31, 31))
        patch11 = cv2.cvtColor(patch5, cv2.COLOR_BGR2GRAY)
        patch12 = cv2.cvtColor(patch6, cv2.COLOR_BGR2GRAY)
        patch5 = patch5.reshape(1, 31, 31, 3)
        patch6 = patch6.reshape(1, 31, 31, 3)
        patch11 = patch11.reshape(1, 31, 31, 1)
        patch12 = patch12.reshape(1, 31, 31, 1)
        q.put((patch5, patch6, patch11, patch12, '561112'))

    t1 = threading.Thread(target=thread_get_patch_1_7())
    t2 = threading.Thread(target=thread_get_patch_2_3_8_9())
    t3 = threading.Thread(target=thread_get_patch_4_10())
    t4 = threading.Thread(target=thread_get_patch_5_6_11_12())

    for t in [t1, t2, t3, t4]:
        t.start()
    for t in [t1, t2, t3, t4]:
        t.join()
    result = []
    while not q.empty():
        result.append(q.get())
    for item in result:
        if item[-1] == '17':
            patch1 = item[0]
            patch7 = item[1]
        if item[-1] == '2389':
            patch2 = item[0]
            patch3 = item[1]
            patch8 = item[2]
            patch9 = item[3]
        if item[-1] == '410':
            patch4 = item[0]
            patch10 = item[1]
        if item[-1] == '561112':
            patch5 = item[0]
            patch6 = item[1]
            patch11 = item[2]
            patch12 = item[3]

    del t1, t2, t3, t4, q

    q = queue.Queue()

    def thread_rgb():
        tf.reset_default_graph()
        net = Net.CNN('rgb')
        feature1 = net.output_deepid(patch1, 'checkpoint/patch_1.ckpt')
        feature2 = net.output_deepid(patch2, 'checkpoint/patch_2.ckpt')
        feature3 = net.output_deepid(patch3, 'checkpoint/patch_3.ckpt')
        feature4 = net.output_deepid(patch4, 'checkpoint/patch_4.ckpt')
        feature5 = net.output_deepid(patch5, 'checkpoint/patch_5.ckpt')
        feature6 = net.output_deepid(patch6, 'checkpoint/patch_6.ckpt')
        q.put((feature1, feature2, feature3, feature4, feature5, feature6, 'rgb'))

    def thread_gray():
        tf.reset_default_graph()
        net = Net.CNN('gray')
        feature7 = net.output_deepid(patch7, 'checkpoint/patch_7.ckpt')
        feature8 = net.output_deepid(patch8, 'checkpoint/patch_8.ckpt')
        feature9 = net.output_deepid(patch9, 'checkpoint/patch_9.ckpt')
        feature10 = net.output_deepid(patch10, 'checkpoint/patch_10.ckpt')
        feature11 = net.output_deepid(patch11, 'checkpoint/patch_11.ckpt')
        feature12 = net.output_deepid(patch12, 'checkpoint/patch_12.ckpt')
        q.put((feature7, feature8, feature9, feature10, feature11, feature12, 'gray'))

    t1 = threading.Thread(target=thread_rgb())
    t2 = threading.Thread(target=thread_gray())
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    result = []
    while not q.empty():
        result.append(q.get())
    for item in result:
        if item[-1] == 'rgb':
            feature1 = item[0]
            feature2 = item[1]
            feature3 = item[2]
            feature4 = item[3]
            feature5 = item[4]
            feature6 = item[5]
        if item[-1] == 'gray':
            feature7 = item[0]
            feature8 = item[1]
            feature9 = item[2]
            feature10 = item[3]
            feature11 = item[4]
            feature12 = item[5]
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
    q = queue.Queue()

    def extract_1():
        feature = feature_extract(img1)
        q.put((feature))

    def extract_2():
        feature = feature_extract(img2)
        q.put((feature))

    t1=threading.Thread(target=extract_1)
    t2=threading.Thread(target=extract_2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    result = []
    while not q.empty():
        result.append(q.get())
    feature1 = result[0]
    feature2 = result[1]

    sim = 1 - cosine(feature1, feature2)
    return sim


if __name__ == '__main__':
    img1 = 'F:/Project/TheTrulyDeepID/DeepID1/DataSet/images/Aamir_Khan/3.jpg'
    img2 = 'F:/Project/TheTrulyDeepID/DeepID1/DataSet/images/Aaron_Staton/1.jpg'
    start = time.time()
    sim = getSim(img1, img2)
    end = time.time()
    print(sim, end - start)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
