import cv2
import numpy as np
import os
from datetime import datetime

# Predefined library for face_recognition using opencv
import face_recognition

# Get images
def getImageClassNames():
	path = "images"
	images = []
	classNames = []

	# Lists files in given path
	myList = os.listdir(path)

	# Adding images to the lists
	for image in myList:
		curImg = cv2.imread(f'{path}/{image}')
		images.append(curImg)
		classNames.append(os.path.splitext(image)[0])

	return [images, classNames]


# Get encodings for all images
def findEncodings(imagesList):
	encodings = []

	for img in imagesList:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodings.append(encode)

	return encodings


# Add attendance in CSV file
def markAttendance(name):
	with open("attendance.csv", 'r+') as f:
		origAttendance = f.readlines()
		names = [line.split(',')[0] for line in origAttendance]

		if name not in names:
			now = datetime.now()
			dtString = now.strftime("%H:%M:%S")
			f.writelines(f'\n{name},{dtString}')


# Main Function
if __name__ == "__main__":
	# Get images
	images, classNames = getImageClassNames()

	# Encodings for known images
	encodeListKnown = findEncodings(images)
	print("Encodings complete ...")

	# Getting images from webcam
	cap = cv2.VideoCapture(0)

	while True:
		success, img = cap.read()
		imgResize = cv2.resize(img, (0,0), None, 0.25, 0.25)
		imgResize = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)

		faceCurFrame = face_recognition.face_locations(imgResize)
		encodesCurFrame = face_recognition.face_encodings(imgResize, faceCurFrame)

		for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
			matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
			faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
			# print(faceDist)

			matchIdx = np.argmin(faceDist)

			if matches[matchIdx]:
				name = classNames[matchIdx]
				print(name)

				y1, x2, y2, x1 = faceLoc
				y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

				cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
				cv2.rectangle(img, (x1,y2-35), (x2,y2), (255,255,255), cv2.FILLED)
				cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

				markAttendance(name)

		cv2.imshow("Webcam", img)
		cv2.waitKey(1)














