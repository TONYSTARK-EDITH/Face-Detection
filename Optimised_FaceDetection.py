import cv2
from PIL import Image


def optimisedDetector():
	# noinspection PyShadowingNames
	def detectFaceOpenCVDnn(net, frame):
		# Making an copy of the frame which is given as argument
		frameOpencvDnn = frame.copy()
		# Determining the Height and width of the frame
		frameHeight = frameOpencvDnn.shape[0]
		frameWidth = frameOpencvDnn.shape[1]
		# Determining the blob from the frame
		blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
		
		# net is the model and the config file of the Neural Network
		net.setInput(blob)
		detections = net.forward()
		boxes = []
		for i in range(detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > conf_threshold:
				x1 = int(detections[0, 0, i, 3] * frameWidth)
				y1 = int(detections[0, 0, i, 4] * frameHeight)
				x2 = int(detections[0, 0, i, 5] * frameWidth)
				y2 = int(detections[0, 0, i, 6] * frameHeight)
				# appending the face locations in the list named boxes
				boxes.append([x1, y1, x2, y2])
		return frameOpencvDnn, boxes
	
	# OpenCV DNN supports 2 networks.
	# 1. FP16 version of the original caffe implementation ( 5.4 MB )
	# 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
	
	DNN = "CAFFE"
	if DNN == "CAFFE":
		# Providing the models for the neural network to detect the accurate face
		modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
		configFile = "models/deploy.prototxt"
		
		net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
	else:
		# Providing the models for the neural network to detect the accurate face
		modelFile = "models/opencv_face_detector_uint8.pb"
		configFile = "models/opencv_face_detector.pbtxt"
		net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
	conf_threshold = 0.7
	cap = cv2.VideoCapture(0)
	# Iteration needed when face is not detected properly
	while 1:
		# Capturing the First Frame of the sys camera
		hasFrame, frame = cap.read()
		if not hasFrame:
			break
		# Determining the face location using DNN algorithms in OpenCv2
		outOpencvDnn, boxes = detectFaceOpenCVDnn(net, frame)
		cv2.imwrite("face.jpg", outOpencvDnn)
		outOpencvDn = Image.open("face.jpg")
		# Iterating over the face locations provided by the Function detectFaceOpenCVDnn
		for x, y, w, h in boxes:
			# Cropping the Image Over the Face locations Using Pillow Image
			outOpencvDnn = outOpencvDn.crop((x, y, w, h))
			outOpencvDnn.save("face.jpg")
			# break is needed when there is more than one face in the captured frame it should take the first face it captured
			break
		# noinspection PyUnboundLocalVariable
		break
optimisedDetector()
