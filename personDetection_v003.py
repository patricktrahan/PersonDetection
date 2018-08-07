# real time object detection PC

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import numpy as np
import argparse
import imutils
import time
import cv2
import smtplib
import datetime
import os

def email():
		fromaddr = 'pdt2157@gmail.com'	# from email address
		toaddrs  = 'pdt2157@gmail.com'	# to email address
		username = 'pdt2157@gmail.com' # username
		password = 'KlismGds2931'	# password
		msg = "Person Detected"
		server = smtplib.SMTP('smtp.gmail.com')
		server.ehlo()
		server.starttls()
		server.login(username,password)
		server.sendmail(fromaddr, toaddrs, msg)
		server.quit()

def emailImage():
		fromaddr = ""	# from email address
		toaddrs  = ""	# to email address
		username = "" # username
		password = ""	# password
		msg = MIMEMultipart()
		msg["From"] = fromaddr
		msg["To"] = toaddrs
		msg["Subject"] = "Person detected."
		body = "A person has been detected."
		msg.attach(MIMEText(body, "plain"))
		filename = str(imgName)
		print(filename)
		attachment = open(("# Location of attachment/%s" % filename + ".jpg"), "rb")
		part = MIMEBase("application", "octet-stream")
		part.set_payload((attachment).read())
		encoders.encode_base64(part)
		part.add_header("Content-Disposition", "attachement; filename= %s" % filename)
		msg.attach(part)
		server = smtplib.SMTP('smtp.gmail.com', 587)
		server.ehlo()
		server.starttls()
		server.login(username,password)
		server.sendmail(fromaddr, toaddrs, msg)
		server.quit()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# varibales for detecting a person and sending a notification
detectedClass = None
counter = 0
counterThreshold = 10
resetCounter = 0
resetCounterMax = 10
resetBool = False

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# init date and time for video output file
timestamp = time.strftime("%Y%m%d%H%M%S")

# video capture init and save
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=480)
	ret, frame = cap.read()
	
	# counter increments for resetting video capture to a new file
	resetCounter += 1

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			detectedClass = CLASSES[idx]

			# if person detected, send an email notification
			if detectedClass == "person":
					# increment initial counter and return resetCounter to 0
					counter += 1
					resetCounter = 0
					# allows the counter to reach the counterThreshold before
					# notification is sent
					if counter == counterThreshold:
							# notification if person is detected when the counter
							# reaches counterThreshold and sends email
							resetBool = True
							print("person detected")
							# init video and image output file with start time stamp
							img = True
							timestamp = time.strftime("%Y%m%d%H%M%S")
							imgName = timestamp
							out = cv2.VideoWriter((timestamp + ".avi"), fourcc, 15.0, (640,480))
							img = cv2.imwrite((timestamp + ".jpg"), frame)
							#email()
							#emailImage()
					if counter >= counterThreshold:
							# write video to file
							out.write(frame)
		if resetCounter >= resetCounterMax:
				# resets counter after resetCounterMax is reached
				if resetBool == True:
						counter = 0
						out.release()
						print("out released")
						resetBool = False
				resetCounter = 0
				print('counter reset')

	# show the output frame
	cv2.imshow("Motion Detector", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
vs.stop()
cap.release()
out.release()
cv2.destroyAllWindows()