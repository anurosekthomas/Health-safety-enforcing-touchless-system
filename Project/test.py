from itertools import count
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import speech_recognition as spr
from playsound import playsound
import threading
from tkinter import *
import tkinter as tki
from PIL import Image as Img
from PIL import ImageTk
import numpy as np
import imutils
import os
import cv2
import time
import sqlite3
from datetime import datetime

counter_no_mask = 0
counter = 0

#load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
vs = VideoStream(src=0).start()

class SpeechRecognizer(threading.Thread):
    
    def __init__(self):
        super(SpeechRecognizer, self).__init__()
        self.setDaemon(True)
        self.question_text = ""
        self.answer_text = ""
        self.start_face_detection = False
        self.delay = 0

       
    
    def check_in_database(self, collected_data):
        name = '%' + collected_data[0] + '%'
        phone = collected_data[3].replace(' ', '')

         # sqllite connection.
        govt_database_file = "GovtData.db"
        conn = sqlite3.connect(govt_database_file)
        db_cursor = conn.cursor()
        db_cursor.execute(
            "SELECT 1 FROM Person WHERE Name LIKE '{}' AND Phone LIKE '{}'".format(name, phone))
        query_result = db_cursor.fetchall()
        if len(query_result) > 0:
            return 1
        return 0

    def save_to_db(self,collected_data):
        name  = collected_data[0]
        place = collected_data[1]
        is_vaccinated = 0
        if (collected_data[2]=="yes"):
            is_vaccinated = 1
        phone = collected_data[3].replace(' ', '')
        today = datetime.now()

        # sqllite connection.
        customer_db = "customer.db"
        conn = sqlite3.connect(customer_db)
        db_cursor = conn.cursor()
        db_cursor.execute("INSERT INTO Customer (Name, Place, Phone, IsVaccinated, VisitTime) VALUES ('{}','{}','{}','{}','{}')".format(name, place, phone, is_vaccinated, today))
        conn.commit()
        conn.close()



    def run(self):
        while True:
            if self.start_face_detection == False:
                if self.delay==1:
                    self.question_text = "You are Welcome"
                    time.sleep(3)
                    self.delay=0
                recognition = spr.Recognizer()
                mc = spr.Microphone(device_index=0)

                collected_data = ["", "", "", ""]
                self.question_text = "Say Hello to start"
                playsound("hello.mp3")
                while True:
                    with mc as source:
                        try:
                            audio = recognition.listen(source)
                            rec_text = recognition.recognize_google(audio).lower()
                        except:
                            continue
                        if (rec_text.find('hello') > -1):
                            self.answer_text = 'Your Answer: ' + "Hello"
                            time.sleep(2)
                            self.answer_text = ""
                            break

                questions = ["What is your name: ","Where are you from: ","Have you been vaccinated: ","What is your phone number: "]
                audios = ["name.mp3","place.mp3","vaccination.mp3","phone.mp3"]

                with mc as source:
                    for i in range (0, 4):
                        self.question_text = questions[i]
                        playsound(audios[i])
                        text = ""
                        while(text==""):
                            try:
                                audio = recognition.listen(source)
                                text = recognition.recognize_google(audio)
                                
                                # Adding collected data into list.
                                collected_data[i] = text 
                                self.answer_text = 'Your Answer: ' + text
                            except:
                                print("Please Tell Something")
                        time.sleep(3)
                        self.answer_text = ""
                self.save_to_db(collected_data)
                if(self.check_in_database(collected_data) == 1):
                    self.question_text = "You were recommended to be at quarantine"
                    playsound('alert.mp3')
                    time.sleep(3)
                else:
                    self.question_text = "Please wait until we detect your mask properly."
                    time.sleep(3)
                    self.start_face_detection = True
                
            else:
                time.sleep(1)

            


recognizer = SpeechRecognizer()
recognizer.start()

class App(object):

    def __init__(self,root):
        self.root = root
        self.root.geometry('700x600')
        self.root.title('Covid Demo')
        self.num = 1
        self.start_face_detection = False


        # Question label.
        self.question_label = tki.Label(self.root, text="",bd='10',fg = "black",bg = "orange",width=60, height=2,font = "Helvetica 16 bold")
        self.question_label.pack(side=TOP) # Center alligned
        # self.question_label.configure(text="Your Answer: {}".format("Nirmal"))

        # Answer label.
        self.answer_label = tki.Label(self.root, text="",bd='10',fg = "black",bg = "cyan",width=60, height=2,font = "Helvetica 16 bold")
        self.answer_label.pack(side=TOP) # Center alligned
        # self.answer_label.configure(text="Your Answer: {}".format("Nirmal"))

        

        # Create a label in the frame
        self.lmain = tki.Label(self.root)
        self.lmain.pack()

        # Show camera function.
        self.video_stream()
        
    def detect_and_predict_mask(self,frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()
        # print(detections.shape)

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)


    def update_recognized_text(self):
        self.question_label.configure(text=recognizer.question_text)
        self.answer_label.configure(text=recognizer.answer_text)
        self.start_face_detection = recognizer.start_face_detection
    
    # Function for video streaming
    def video_stream(self):
        self.update_recognized_text()
        global counter_no_mask
        global counter

        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        if (self.start_face_detection == True):
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                if label == 'Mask':
                    counter+=1
                else:
                    counter_no_mask+=1
                if counter_no_mask == 50:
                    playsound("warning.mp3")
                    counter_no_mask=0

                if counter== 40:
                    counter_no_mask = 0
                    counter = 0
                    playsound("welcome.mp3")
                    recognizer.start_face_detection = False
                    recognizer.delay=1
                    # time.sleep(3)
                
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
            
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.img = Img.fromarray(self.cv2image)
        self.imgtk = ImageTk.PhotoImage(image=self.img)
        self.lmain.imgtk = self.imgtk
        self.lmain.configure(image=self.imgtk)
        self.lmain.after(1, self.video_stream) 
    

root = tki.Tk()
app = App(root)
root.mainloop()