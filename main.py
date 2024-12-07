import os
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import load_img, img_to_array
import tempfile

facemodel=cv2.CascadeClassifier("./Face_Training_Data/face.xml")
# facemodel return coordinates of frame where faces are detect i.e. (x, y, length, width)

# loading mask detection model
mask_model=load_model('mask.h5')

st.title("Face Mask Detection System")
choice=st.sidebar.selectbox("",("Home","Image","Video","Camera"))

if choice=="Home":
    # st.header("Welcome to Face Mask Detection System Application")
    st.image("./Media/home.png",width=550)

elif choice=="Image":
    file=st.file_uploader("Upload Image file")
    if file:
        # changing file into binary format
        b=file.getvalue()
        # creating buffer for binary format file
        d=np.frombuffer(b,np.uint8)
        # Changing binary file to viewable format to represent in frontend window.
        img=cv2.imdecode(d, cv2.IMREAD_COLOR)

        
        faces=facemodel.detectMultiScale(img)

        for (x,y,l,w) in faces:
            crop_face=img[y:y+w,x:x+l]
            cv2.imwrite('temp.jpg',crop_face)
            crop_face=load_img('temp.jpg',target_size=(150,150,3))
            crop_face=img_to_array(crop_face)
            crop_face=np.expand_dims(crop_face,axis=0)

            pred=mask_model.predict(crop_face)[0][0]
            if pred==0:
                cv2.rectangle(img,(x,y),(x+l,y+w), (2,240,66), 3)
            else:
                cv2.rectangle(img,(x,y),(x+l,y+w), (0,0,255), 3)

        st.image(img,channels='BGR',width=400)


elif choice=="Video":
    file=st.file_uploader("Upload Video")
    # creating window for showing the frames of video.
    window=st.empty()
    if file:
        tfile=tempfile.NamedTemporaryFile()
        tfile.write(file.read())
        vid=cv2.VideoCapture(tfile.name)
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
                    
                window.image(frame, channels='BGR')

elif choice=="Camera":
    link=st.text_input("Enter the IP camera link or value 0 for device webcam opening")
    button=st.button("Start")
    if link=='0':
        link=0
    else:
        link=link+'/video'
    # creating window for showing the frames of video.
    window=st.empty()
    if button:
        vid=cv2.VideoCapture(link)
        stop_btn=st.button("Stop")
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
                    
                window.image(frame, channels='BGR')
            
            if stop_btn:
                break
                st.rerun()