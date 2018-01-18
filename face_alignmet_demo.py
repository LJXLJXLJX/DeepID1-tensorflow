import cv2
import dlib
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
img = cv2.imread('F:/Project/TheTrulyDeepID/DeepID1/DataSet/images/A.J._Buckley/1.jpg')
img=cv2.resize(img,(500,500))
faces = detector(img,1)
if (len(faces) > 0):
    for k,d in enumerate(faces):
        cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
        shape = landmark_predictor(img,d)
        for i in range(68):
            cv2.circle(img, (shape.part(i).x, shape.part(i).y),5,(0,255,0), -1, 8)
            cv2.putText(img,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
cv2.imshow('Frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()