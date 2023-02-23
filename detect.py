import os
import numpy as np
import cv2
import pickle
import uuid

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
#https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf   -> haarcascades


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("tariner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
tot=0

while(True):
    
	ret, frame = cap.read()
	cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
	gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x, y, w, h) in faces:
		print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w] 
		roi_color = frame[y:y+h, x:x+w]
		id_, conf = recognizer.predict(roi_gray)
		if conf>=75:
			font = cv2.FONT_HERSHEY_PLAIN
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 2, color, stroke, cv2.LINE_AA)

		
		# if not in database then deal with this part of code
		elif conf<75:

			
			font = cv2.FONT_HERSHEY_PLAIN
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, "Unknown", (x,y), font, 3, color, stroke, cv2.LINE_AA)

			
			f_path = './unknown_people'
			p_id = uuid.uuid1()
			u_id = str(p_id)
			cnt = 15
			ori = "unknown_person_.png"
			res = ori[ : cnt] + u_id + ori[cnt : ]
			# img_item = "unknown_person_.png"	
			if tot%70==0: 
					cv2.imwrite(os.path.join(f_path, res), roi_color) 
			
			tot+=1


		color = (255, 0, 0) 
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    	
	cv2.imshow('frame',frame) #https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()