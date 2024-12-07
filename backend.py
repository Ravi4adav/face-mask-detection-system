import cv2
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import datetime


facemodel=cv2.CascadeClassifier("./Face_Training_Data/face.xml")
# facemodel return coordinates of frame where faces are detect i.e. (x, y, length, width)

# loading mask detection model
mask_model=load_model('mask.h5')

# cv2.VideoCapture("IP Camera Link") # Used to connect IP Camera via IP camera link
vid=cv2.VideoCapture('https://192.168.1.8:8080/video')     # 0 --> will connect device webcam
# vid=cv2.VideoCapture('./Media/mask.mp4')     # 0 --> will connect device webcam

# ===================================================================================================================

# img=cv2.imread('./Data/data/train/with_mask/13-with-mask.jpg')
# img=cv2.imread('./Data/data/train/without_mask/18.jpg')
# img=cv2.imread('face-mask-study.jpg')


# faces=facemodel.detectMultiScale(img)
# # faces=facemodel.detectMultiScale('./Data/data/train/without_mask/3.jpg')

# for (x,y,l,w) in faces:
#     crop_face=img[y:y+w,x:x+l]
#     cv2.imwrite('temp.jpg',crop_face)
#     crop_face=load_img('temp.jpg',target_size=(150,150,3))
#     crop_face=img_to_array(crop_face)
#     crop_face=np.expand_dims(crop_face,axis=0)

#     pred=mask_model.predict(crop_face)[0][0]
#     if pred==0:
#         cv2.rectangle(img,(x,y),(x+l,y+w), (2,240,66), 3)
#     else:
#         cv2.rectangle(img,(x,y),(x+l,y+w), (0,0,255), 3)

# cv2.namedWindow("Camera Window",cv2.WINDOW_NORMAL)
# cv2.imshow("Camera Window",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ===================================================================================================================


while (vid.isOpened()):
    # Reading individual frames from video
    flag,frame=vid.read()
    if (flag):
        faces=facemodel.detectMultiScale(frame) # returns (x,y, length, width)
        for (x,y,l,w) in faces:

            crop_face_image=frame[y:y+w,x:x+l]
            crop_face=cv2.imwrite('temp.jpg',crop_face_image)
            crop_face=load_img('temp.jpg',target_size=(150,150,3))
            crop_face=img_to_array(crop_face)
            crop_face=np.expand_dims(crop_face,axis=0)

            pred=mask_model(crop_face)[0][0]

            if pred==0:
                # creating green rectangle around face while face mask detect
                cv2.rectangle(frame,(x,y),(x+l, y+w),(2, 240, 66),3)
            else:
                cv2.rectangle(frame,(x,y),(x+l, y+w), (0,0,255), 3)
                
                # Saving cropped image with no mask face for future training.
                current_time=datetime.datetime.now()
                current_time=current_time.strftime('%d%m%y_%H%M%S')

                path="Data/train/unmask/"+str(current_time)+".jpg"
                cv2.imwrite(path,crop_face_image)



        cv2.namedWindow("Camera Window", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera Window", frame)
        # Because our video is frames per second which moves very fast, so to handle this we will freeze a frame for 20 miliseconds.
        # using "cv2.waitKey()" function
        key=cv2.waitKey(14)
        # cv2.waitKey() also takes the value of key pressed by the user in the keyboard. So this can be helpful stopping our endless
        # video streaming.

        # Check if the window was closed manually using window close button.
        if cv2.getWindowProperty('Camera Window', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Check if key press 'x' to quit window.
        if (key==ord('x')):
            break
    else:
        break


vid.release()
cv2.destroyAllWindows()