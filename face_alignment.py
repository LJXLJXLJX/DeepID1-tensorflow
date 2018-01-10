

import dlib



detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def alignment(img):
    faces = detector(img, 1)
    if (len(faces) == 1):
        shape = landmark_predictor(img, faces[0])
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

        landmarkList=[left_eye,right_eye,nose_tip,mouse_left,mouse_right]
        return landmarkList
        # for landmark in landmarkList:
        #     cv2.circle(img, (landmark[0],landmark[1]), 5, (255, 0, 0), -1, 8)


