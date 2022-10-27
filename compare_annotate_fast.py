import os
import cv2
import json
import time

path = "/home/nathan/Desktop/run_1665783080/"
saveName = "ear"
modelName = "/home/nathan/Desktop/ear_finder/ear_tflite/corn_ear_oct13.tflite"

if modelName != "":
	from tflite_support.task import core
	from tflite_support.task import processor
	from tflite_support.task import vision


class Annotator:
	def __init__(self, path, saveType, modelName=""):
		self.saveType = saveType
		self.mouseDown = False
		self.dragged = False
		self.openRects = []
		self.currentRectStart = [-1,-1]
		self.possibleRects = []
		self.path = path
		self.firstCallback = True
		self.allRects = {}
		self.auto = False
		self.cursorLocation = (0,0)
		self.autoShape = [100,100] # h,w
		self.autoRect = [0,0, 0,0, 0, "0"]

		dirs = os.listdir(path)
		if "labels.json" in dirs:
			with open(path + '/labels.json') as json_file:
				data = json.load(json_file)
			print(data)
			self.allRects = data
		self.useDetector = False
		if modelName != "":
			self.useDetector = True
			base_options = core.BaseOptions(
			  file_name=modelName, use_coral=False, num_threads=4)
			detection_options = processor.DetectionOptions(
			  max_results=6, score_threshold=0.1)
			options = vision.ObjectDetectorOptions(
			  base_options=base_options, detection_options=detection_options)
			self.detector = vision.ObjectDetector.create_from_options(options)



	def clickAndMove(self, event, x, y, flags, param):
		# print(event, x, y, flags, param)

		self.cursorLocation = (x,y)
		if event==1:
			print("click")
			self.mouseDown = True
			self.openRects += [[x,y,x,y,0]]
			self.dragged = False

		elif event==4:
			print("release")
			self.mouseDown = False
			if abs(self.openRects[-1][0]-self.openRects[-1][2]) < 2 and abs(self.openRects[-1][1]-self.openRects[-1][3]) < 2:
				self.openRects = self.openRects[0:-1]

			if self.dragged == False:


				j=0
				while j<len(self.possibleRects):
					i = self.possibleRects[j]
					if x > min((i[0], i[2])) and x < max((i[0], i[2])):
						if y > min((i[1], i[3])) and y < max((i[1], i[3])):
							i+=[0]
							self.openRects += [i]
							self.possibleRects = self.possibleRects[0:j] + self.possibleRects[j+1::]
							j-=1
					j+=1

				j=0
				while j<len(self.openRects):
					i = self.openRects[j]
					self.openRects[j][4] = 0
					if x > min((i[0], i[2])) and x < max((i[0], i[2])):
						if y > min((i[1], i[3])) and y < max((i[1], i[3])):
							print("selected")
							print("give number")
							k = cv2.waitKey(0)
							if k == 8:
								print("delete")
								self.openRects = self.openRects[0:-1]
							else:
								alphabet = "abcdefghijklmnopqrstuvwxyz"
								if (k-97) >= 0 and (k-97) < len(alphabet):
									print(alphabet[k-97])
									if len(self.openRects[j]) < 6:
										self.openRects[j] += ["_"]
									self.openRects[j][5] = alphabet[k-97]
									print(self.openRects[j])

							self.openRects[j][4] = 1
					j+=1

				self.dispImg()

			else:
				if abs(self.openRects[-1][0]-self.openRects[-1][2]) < 3 or abs(self.openRects[-1][1]-self.openRects[-1][3]) < 3:
					self.openRects=self.openRects[0:-1]
					print("rectangle too small")
				else:
					print("give number")
					k = cv2.waitKey(0)
					if k == 8:
						print("delete")
						self.openRects = self.openRects[0:-1]
					else:
						alphabet = "abcdefghijklmnopqrstuvwxyz"
						if (k-97) >= 0 and (k-97) < len(alphabet):
							print(alphabet[k-97])
							self.openRects[-1] += [alphabet[k-97]]
				self.dispImg()



		elif self.mouseDown:
			self.dragged = True
			h, w, _ = self.ogImg.shape
			if x>w:
				x=w-1
			elif y>h:
				y=h-1
			if x<0:
				x=0
			elif y<0:
				y=0
			self.openRects[-1][2] = x
			self.openRects[-1][3] = y
			# print(x, y)
			self.dispImg()

		elif self.auto:
			self.checkAuto(x,y)
							
			

			
			self.dispImg()


	def checkAuto(self, x, y):
		shapeChanged = False
		for i in self.possibleRects:
			if x > min(i[0], i[2]) and x < max(i[0], i[2]):
				if y > min(i[1], i[3]) and y < max(i[1], i[3]):
					self.autoShape = (abs(i[2]-i[0]), abs(i[3]-i[1]))
					shapeChanged = True
					self.cursorLocation = (int((i[0]+i[2])/2), int((i[1]+i[3])/2))
					print("found possible rect")

					break

		if shapeChanged == False:
			for i in self.openRects:
				if x > min(i[0], i[2]) and x < max(i[0], i[2]):
					if y > min(i[1], i[3]) and y < max(i[1], i[3]):
						self.autoShape = (abs(i[2]-i[0]), abs(i[3]-i[1]))

		x = self.cursorLocation[0]
		y = self.cursorLocation[1]
		w = self.autoShape[0]
		h = self.autoShape[1]			
		self.autoRect = [int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2), 0, self.autoRect[5]]
			

	def dispImg(self):
		img = self.ogImg.copy()

		for i in self.openRects:
			

			if i[4] == 1:
				img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (0,255,0), 2)
			elif i[4] == 0:
				img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (255,0,0), 2)
			if len(i) == 6:
				label = i[5]# str(int(i.categories[0].score*1000)/1000) #  "ear: " + str(j) + " " +
				img = cv2.putText(img, label, (int((i[0]+i[2])/2),int((i[1]+i[3])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)


		for i in self.possibleRects:
			img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (0,255,255), 1)

		if self.auto:
			x = self.cursorLocation[0]
			y = self.cursorLocation[1]
			w = self.autoShape[0]
			h = self.autoShape[1]
			img = cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0,0,255), 3)
			img = cv2.putText(img, self.autoRect[5], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)


			

		cv2.imshow("img", img)


		if self.firstCallback:
			cv2.setMouseCallback("img", self.clickAndMove)
			# cv2.createButton("img",None,cv2.QT_PUSH_BUTTON,1)
			self.firstCallback = False
			# cv2.createButton("img", self.buttonPressed, None,cv2.QT_PUSH_BUTTON,100)
			switch=0
			# cv2.createTrackbar("yo", 'img',0,1, self.buttonPressed)


	# def buttonPressed(self, val):
	# 	print("yo", val)


	def detectPossible(self):
		rgb_image = cv2.cvtColor(self.ogImg, cv2.COLOR_BGR2RGB)

		# Create a TensorImage object from the RGB image.
		input_tensor = vision.TensorImage.create_from_array(rgb_image)

		# Run object detection estimation using the model.
		detection_result = self.detector.detect(input_tensor)
		
		for i in detection_result.detections:
			x = i.bounding_box.origin_x
			y = i.bounding_box.origin_y
			w = i.bounding_box.width
			h = i.bounding_box.height
			if w*h>0:
				self.possibleRects += [[x,y, x+w,y+h]]
		

	def goThroughFiles(self):

		startFrame = 109
		
		file = "color.avi"
		cap = cv2.VideoCapture(path + "/" + file)
		ret = True
		k = 0
		j=0
		jStart = 0
		frameChange = True
		if file not in self.allRects:
			self.allRects[file] = {}


		while j<startFrame:
			cap.read()
			j+=1

		while cap.isOpened() and ret:

			analyzedBefore = False
			if str(j) in self.allRects[file]:
				self.openRects = self.allRects[file][str(j)]
				analyzedBefore = True
			print("open rect", self.openRects)
			jStart = j
			# print("frame?", cap.get(cv2.CAP_PROP_POS_FRAMES))
			if frameChange:
				ret, self.ogImg = cap.read()
				print("reading frame",j)
			if not ret:
				print("done reading this image")
				break
			frameChange = False

			self.possibleRects = []
			if self.useDetector: # and not analyzedBefore:
				self.detectPossible()
			
			if self.auto:
				self.checkAuto(self.cursorLocation[0], self.cursorLocation[1])


			self.dispImg()

			if self.auto:
				time.sleep(0.5)
				k = cv2.waitKey(1)
				if k == 32:
					print("stopping,")
					self.auto = False
			else:
				k = cv2.waitKey(0)
			print(k)

			if k == 27:
				break
			elif k == 44:
				print("going to beginning of video")
				cap = cv2.VideoCapture(path + "/" + file)
				j=0
				while j<startFrame:
					cap.read()
					j+=1
				frameChange = True
			elif k == 13:
				print("about to do auto. Give letter")
				k = cv2.waitKey(0)
				alphabet = "abcdefghijklmnopqrstuvwxyz"
				if (k-97) >= 0 and (k-97) < len(alphabet):
					print("auto")
					self.auto = True
					print(self.autoRect)
					self.autoRect[5] = alphabet[k-97]

				else:
					print("auto cancelled")
		
			elif k == 81 or k == 106: # left
				print("back 1 frame (watch out slow)")

				cap = cv2.VideoCapture(path + "/" + file)
				m = 0
				if j == 0:
					i-=2
					break

				while m < j-1: # watch out: this is pretty inefficient. Use sparingly
					# print(m)
					cap.read()
					m += 1
				j-=1
				frameChange = True


			elif k == 115:
				self.allRects[file][str(j)] = self.openRects[:]
				self.saveDataJSON()

			elif k == 108:
				print("skip 5")
				m = 0
				j+=1
				for m in range(0,5):
					cap.read()
					j+=1
				frameChange = True

			elif k == 8: # delete
				l=0
				deleted = 0
				while l<len(self.openRects):
					if self.openRects[l][4] == 1:
						self.openRects = self.openRects[0:l] + self.openRects[l+1::]
						l-=1
						deleted += 1
					l+=1
				if deleted == 0 and len(self.openRects) > 0:
					self.openRects = self.openRects[0:-1]

			else:
				frameChange = True
				j+=1


			if self.auto:
				print("adding to openrects")
				self.openRects += [self.autoRect]

			if self.openRects[:] != []:
				if self.mouseDown == False:
					m = 0
					while m<len(self.openRects):
						orr = self.openRects[m]
						if abs(orr[0] - orr[2]) < 3 or abs(orr[1] - orr[3]) < 3:
							self.openRects = self.openRects[0:m] + self.openRects[m+1::]
							m-=1
							print("removed tiny rectangle")
						m+=1
				self.allRects[file][str(jStart)] = self.openRects[:]
				self.openRects = []



		if self.openRects[:] != []:
			self.allRects[file][str(jStart)] = self.openRects[:]
			self.openRects = []

	



	def reduceJSON(self):
		k = list(self.allRects.keys())
		reduced = {}
		numEars = 0
		numImgs = 0
		for i in k:
			l = list(self.allRects[i].keys())
			dictToAdd = {}
			for j in l:
				if self.allRects[i][j] != []:
					dictToAdd[j] = self.allRects[i][j]

					numEars += len(self.allRects[i][j])
					numImgs += 1

			if len(list(dictToAdd.keys())) > 0:
				reduced[i] = dictToAdd 
		print("reduced", reduced)
		print("recorded", numEars, "ears from", numImgs, "images")
		return reduced


	def saveDataJSON(self):
		reduced = self.reduceJSON()
		
		
		jsonStr = json.dumps(reduced)
		with open(self.path + "/labels.json", 'w') as fileHandle:
			fileHandle.write(str(jsonStr))
			fileHandle.close()


ann = Annotator(path, saveName, modelName)
ann.goThroughFiles()