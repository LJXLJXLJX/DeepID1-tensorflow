import dlib
import cv2

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# 得到五个特征点
def alignment(img, *crop):
    faces = detector(img, 1)
    if len(faces) >= 1:
        max_face = faces[0]
        for i in range(1, len(faces)):
            if (faces[i].right() - faces[i].left()) > (max_face.right() - max_face.left()):
                max_face = faces[i]
        shape = landmark_predictor(img, max_face)
        left_eye = []
        right_eye = []
        nose_tip = []
        mouse_left = []
        mouse_right = []
        left_eye.append(int((shape.part(36).x + shape.part(39).x) // 2))
        left_eye.append(int((shape.part(36).y + shape.part(39).y) // 2))
        right_eye.append(int((shape.part(42).x + shape.part(45).x) // 2))
        right_eye.append(int((shape.part(42).y + shape.part(45).y) // 2))
        nose_tip.append(int(shape.part(30).x))
        nose_tip.append(int(shape.part(30).y))
        mouse_left.append(int(shape.part(48).x))
        mouse_left.append(int(shape.part(48).y))
        mouse_right.append(int(shape.part(54).x))
        mouse_right.append(int(shape.part(54).y))

        top=max(max_face.top(),0)
        bottom=min(max_face.bottom(),img.shape[0])
        left=max(max_face.left(),0)
        right=min(max_face.right(),img.shape[1])

        img = img[top:bottom, left:right]
        landmarkList = [left_eye, right_eye, nose_tip, mouse_left, mouse_right]
        if len(crop)>0 :
            if crop[0] == True:
                return landmarkList, img
        return landmarkList


# 得到五个landmark对应的区域
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


if __name__ == '__main__':
    img = cv2.imread('1.jpg')
    landmarkList,img = alignment(img,True)
    img1,img2=getRois(img,landmarkList[0],landmarkList[1])
    img3=getRois(img,landmarkList[2])
    img4,img5=getRois(img,landmarkList[3],landmarkList[4])
    cv2.namedWindow('1',flags=cv2.WINDOW_NORMAL)
    cv2.imshow('1', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
