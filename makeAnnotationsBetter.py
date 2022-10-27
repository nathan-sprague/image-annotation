import os
import cv2
import json
import numpy as np

saveName = "wholePlant"
saveName = "beeGroup"
largerRectName = ""



if saveName == "ear":
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/ear_videos/ear_rgb_training_videos"
	modelName = "/home/nathan/Desktop/ear_stalk_detection/ear_finder/ear_tflite/corn_ear_oct16.tflite"

elif saveName == "wholePlant":
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/stalk_videos/stalk_rgb_training_videos"
	modelName = ""

elif saveName == "node":
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/stalk_videos/stalk_rgb_training_videos/wholePlant_images"
	modelName = ""

elif saveName == "node": # with subrects
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/stalk_videos/stalk_rgb_training_videos"
	modelName = ""
	largerRectName = "wholePlant"

elif saveName == "beeGroup":
	videosPath = "/home/nathan/Desktop/lab/beeVideos_use"
	modelName = "/home/nathan/Desktop/lab/bee_model.tflite"

elif saveName == "bee":
	videosPath = "/home/nathan/Desktop/lab/beeVideos_use/beeGroup_images"
	modelName = "/home/nathan/Desktop/lab/bee_model.tflite"

elif saveName == "queenBee":
	videosPath = "/home/nathan/Desktop/lab/beeVideos_use"
	modelName = ""


if modelName != "":
	from tflite_support.task import core
	from tflite_support.task import processor
	from tflite_support.task import vision


class Annotator:
	def __init__(self, path, labelName, modelName="", largerRectName = ""):
		self.currentRectangles = {"drawnRectangles": [], "tempRectangle": [[]], "possibleRectangles": [], "selectedRect": []}
		self.firstCallback = True
		self.path = path
		self.labelName = labelName
		self.savableRects = {}
		self.unsavedChanges = False

		self.useSubRects = False
		if largerRectName != "":
			self.useSubRects = True
			self.largerRectName = largerRectName
		self.selectedRectInd = -1

		self.scale = (1,1)
		self.mouseDown = False

		self.falseKey = -1


		self.buttons = {"  delete": [(20, 20), (140, 80), 8], "    prev label": [(180, 20), (360, 80), 91], " prev": [(380, 20), (460, 80), 106], " next": [(480, 20), (560, 80), 108],
						"    next label": [(580, 20), (760, 80), 93], " exit": [(780, 20), (860, 80), 27]}

		self.useDetector = False
		if modelName != "":
			self.useDetector = True
			base_options = core.BaseOptions(
			  file_name=modelName, use_coral=False, num_threads=4)
			detection_options = processor.DetectionOptions(
				max_results=5, score_threshold=0.1)
			options = vision.ObjectDetectorOptions(
				base_options=base_options, detection_options=detection_options)
			self.detector = vision.ObjectDetector.create_from_options(options)


	def mouseEvent(self, event, x, y, flags, param):
		if y < 100:
			if event == 4 and self.currentRectangles["tempRectangle"] == [[]]:
				self.mouseDown = False
				for b in self.buttons:
					if x>self.buttons[b][0][0] and x<self.buttons[b][1][0]:
						if y>self.buttons[b][0][1] and y<self.buttons[b][1][1]:
							print("buttoned")
							self.falseKey = self.buttons[b][2]
							return
				
			elif event == 1:
				return
			y = 100


		y-=100
		y=max(0, y)
		if event == 6:
			self.falseKey = 108 # skip 5 frames if mouse wheel clicked


		x = int(x/self.scale)
		y = int(y/self.scale)

		if event==1:
			self.selectedRectInd = -1
			self.currentRectangles["selectedRect"] = []
			self.mouseDown = True
			self.currentRectangles["tempRectangle"][0] = [x,y,x,y]

		elif event==4:
			self.mouseDown = False
			tempRect = self.currentRectangles["tempRectangle"][0]
			if len(tempRect) > 0:
				if abs(tempRect[0] - tempRect[2]) > 5 and abs(tempRect[1]-tempRect[3]) > 5:
					self.currentRectangles["drawnRectangles"] += [
					[min(tempRect[0], tempRect[2]), min(tempRect[1], tempRect[3]), max(tempRect[0], tempRect[2]), max(tempRect[1], tempRect[3])]]
					print("Made rectangle")

					self.unsavedChanges = True

				else:#if tempRect[0] == tempRect[2] and tempRect[1] == tempRect[3]:
					ind = 0

					for rect in self.currentRectangles["drawnRectangles"]:
						if tempRect[0] > rect[0] and tempRect[0] < rect[2]:
							if tempRect[1] > rect[1] and tempRect[1] < rect[3]:
								self.selectedRectInd = ind
								print("selected", self.selectedRectInd)
								self.currentRectangles["selectedRect"] = [rect]
								break

						ind += 1

					if self.selectedRectInd == -1:
						ind = 0
						for rect in self.currentRectangles["possibleRectangles"]:
							if tempRect[0] > rect[0] and tempRect[0] < rect[2]:
								if tempRect[1] > rect[1] and tempRect[1] < rect[3]:
									self.selectedRectInd = len(self.currentRectangles["drawnRectangles"])
									self.currentRectangles["drawnRectangles"] += [rect]
									print("picked ML predicted")
									self.currentRectangles["selectedRect"] = [rect]
									break

							ind += 1

					if self.selectedRectInd == -1:
						print("didn't select anything")

				# else:
				# 	print("Rectangle too small")
				self.currentRectangles["tempRectangle"] = [[]]

				self.dispImg()

			else:
				print("no Rectangle to be made")

			self.currentRectangles["tempRectangle"][0] = []

		elif event == 0 and self.mouseDown:
			
			h, w = self.openImg.shape[0:2]
			self.currentRectangles["tempRectangle"][0][2] = min(max(x,0),w)
			self.currentRectangles["tempRectangle"][0][3] = min(max(y,0),h)
			self.dispImg()
			

	def detectPossible(self):
		rgb_image = cv2.cvtColor(self.openImg, cv2.COLOR_BGR2RGB)

		# Create a TensorImage object from the RGB image.
		input_tensor = vision.TensorImage.create_from_array(rgb_image)

		# Run object detection estimation using the model.
		detection_result = self.detector.detect(input_tensor)
		possibleRects = []
		
		for i in detection_result.detections:
			x = i.bounding_box.origin_x
			y = i.bounding_box.origin_y
			w = i.bounding_box.width
			h = i.bounding_box.height
			if h > 3 and w > 3:
				possibleRects += [[x,y, x+w,y+h]]
		print("found", len(possibleRects), "possible")
		self.currentRectangles["possibleRectangles"] = possibleRects




	def dispImg(self):
		drawImg = self.openImg.copy()
		# (720, 1280, 3)
		h, w = drawImg.shape[0:2]
		self.scale = 1
		if h != 720:
			self.scale = 720/h
			drawImg = cv2.resize(drawImg, (int(self.scale * w), int(self.scale*h)), interpolation = cv2.INTER_AREA)


		for rectType in self.currentRectangles:
			thicknesses = {"drawnRectangles": 5, "tempRectangle": 1, "possibleRectangles": 1, "selectedRect": 5}
			colors = {"drawnRectangles": (255,0,0), "tempRectangle": (255,0,0), "possibleRectangles": (0,255,255), "selectedRect": (0,255,0)}


			for rect in self.currentRectangles[rectType]:
				if len(rect) > 0:
					drawImg = cv2.rectangle(drawImg, (int(rect[0]*self.scale), int(rect[1]*self.scale)), (int(rect[2]*self.scale), int(rect[3]*self.scale)), colors[rectType], thicknesses[rectType])

		h, w = drawImg.shape[0:2]
		zeros = np.zeros((h+100,w,3), dtype=np.uint8)
		zeros[100::, :] = drawImg
		drawImg = zeros
		for b in self.buttons:
			button = self.buttons[b]
			drawImg = cv2.rectangle(drawImg, button[0], button[1], (255,255,255), -1)
			drawImg = cv2.rectangle(drawImg, button[0], button[1], (255,255,0), 3)
			center = (int((button[0][0]+button[1][0])/2-10*len(b)), int((button[0][1]+button[1][1])/2))
			drawImg = cv2.putText(drawImg, b, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

		cv2.imshow("img", drawImg)


		if self.firstCallback:
			cv2.setMouseCallback("img", self.mouseEvent)

			self.firstCallback = False


	def manageKeyResponses(self):
		jumpAmount = 0
		while jumpAmount == 0:

			self.falseKey = -1
			k = -1
			while self.falseKey == -1 and k == -1:
				k = cv2.waitKey(10)

			if self.falseKey != -1:
				k =  self.falseKey

			print(k)

			if k == 27: # esc
				if self.mouseDown:
					self.mouseDown = False
					self.currentRectangles["tempRectangle"] = [[]]
					self.dispImg()
				else:
					self.saveDataJSON()
					print("done")

					exit()

			elif k == 108: # l
				jumpAmount = 5

			elif k == 106: # j
				if self.useSubRects:
					jumpAmount = -1
				else:
					jumpAmount = -5

			elif k == 47: # / skip to next video
				jumpAmount = self.maxFrameNum + 2

			elif k == 46: # . skip to last video
				jumpAmount = -self.maxFrameNum - 2

			elif k == 91: # [ skip to closest smaller identified rectangle
				framesToGo = np.array([eval(i) for i in list(self.savableRects.keys())])
				if len(framesToGo) > 0:
					if self.frameNum > framesToGo.min():
						closest = framesToGo[framesToGo < self.frameNum].max()
						jumpAmount = closest - self.frameNum
					else:
						print("unable to skip, none before")
				else:
					print("no identified rectangles")

			elif k == 93: # ] skip to closest larger identified rectangle
				framesToGo = np.array([eval(i) for i in list(self.savableRects.keys())])
				if len(framesToGo) > 0:
					if self.frameNum < framesToGo.max():
						closest = framesToGo[framesToGo > self.frameNum].min()
						jumpAmount = closest - self.frameNum
					else:
						print("unable to skip, none above")
				else:
					print("no identified rectangles")

			elif k == 92: # \ skip to last identified rectangle
				framesToGo = np.array([eval(i) for i in list(self.savableRects.keys())])
				if len(framesToGo) > 0:
					jumpAmount = framesToGo.max() - self.frameNum
				else:
					print("no identified rectangles")

			elif k == 115: # s
				print("Saving JSON")
				self.saveDataJSON()

			elif k == 112: # p
				print("exporting images")
				if self.useSubRects:
					self.saveNormalDataXML()
				else:
					self.saveNormalDataXML()

			elif k == 111: # o
				print("exporting just images")
				self.saveImagesData()

			elif k == 8: # d
				print("delete", self.selectedRectInd)
				if self.selectedRectInd >= 0:
					self.currentRectangles["drawnRectangles"] = self.currentRectangles["drawnRectangles"][0:self.selectedRectInd] + self.currentRectangles["drawnRectangles"][self.selectedRectInd+1::]
					print("deleted selected rectangle")
					self.currentRectangles["selectedRect"] = []
					self.selectedRectInd = -1
					self.unsavedChanges = True
					self.dispImg()

				elif len(self.currentRectangles["drawnRectangles"]) > 0:
					self.currentRectangles["drawnRectangles"] = self.currentRectangles["drawnRectangles"][0:-1]
					print("deleted last rectangle")
					self.unsavedChanges = True
					self.dispImg()

				else:
					print("no rectangles to delete")

			elif k == 99: # c
				print("clear")
				self.unsavedChanges = True
				self.currentRectangles = {"drawnRectangles": [], "tempRectangle": [[]], "possibleRectangles": [], "selectedRect": []}
				self.dispImg()

			else:
				jumpAmount += 1


		rects = self.currentRectangles["drawnRectangles"]
		self.currentRectangles = {"drawnRectangles": [], "tempRectangle": [[]], "possibleRectangles": [], "selectedRect": []}
		print(len(rects), "labels made")
		return jumpAmount, rects





	def goThroughDir(self):
		print("starting")
		self.allFiles = os.listdir(self.path)
		self.dirs = []
		for i in self.allFiles:
			if i[-4::] in [".avi", ".png", ".jpg", ".mov"]:
				self.dirs += [i]

		self.dirs.sort()

		self.dirInd = 0

		self.subRectInd = 0
		self.maxSubRects = 0
		self.prelabeledInd = 0

		self.frameNumInd = 0
		self.desiredFrameNum = 0
		self.maxFrameNum = 0


		while self.dirInd < len(self.dirs):
			self.desiredFrameNum = 0

			fileName = self.dirs[self.dirInd]
			print("reading", fileName, "file number", self.dirInd+1, "out of", len(self.dirs))

			self.jsonName = fileName[0:-4] + self.labelName + ".json"
			if self.jsonName in self.allFiles:
				with open(self.path + '/' + self.jsonName) as json_file:
					data = json.load(json_file)
				print("loaded in", data)
				self.savableRects = data
			else:
				self.savableRects = {}
				print("no annotation data found")
			self.unsavedChanges = False

			if self.useSubRects:
				self.largerRectNameJson = fileName[0:-4] + self.largerRectName + ".json"
				if self.largerRectNameJson in self.allFiles:
					with open(self.path + '/' + self.largerRectNameJson) as json_file:
						data = json.load(json_file)
					self.largerRects = data
					self.rectFrames = [eval(i) for i in list(data.keys())] 
					self.desiredFrameNum = int(self.rectFrames[0])-1

				else:
					print("no larger rectangles for", fileName, "called", self.largerRectNameJson)
					print(self.allFiles)
					self.dirInd+=1
					continue


			if fileName[-4::] in [".avi", ".mov"]:
				cap = cv2.VideoCapture(self.path + "/" + fileName)
				self.maxFrameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
				if self.maxFrameNum == 0:
					print("number of frames not known. May need to move through manually")
					self.maxFrameNum = 0
					

				self.frameNum = 0
				while cap.isOpened():

					
					cap.set(1,self.desiredFrameNum)

					ret, self.openImg = cap.read()

					

					if not ret:
						print("failed to read frame")
						self.dirInd += 1
						break

					oldFrameNum = self.frameNum
					self.frameNum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
					print(oldFrameNum, self.frameNum)
					if oldFrameNum == self.frameNum:
						print("the video is strange, attempting to fix")
						cap.release()
						self.fixVideo(self.path + "/" + fileName)
						cap = cv2.VideoCapture(self.path + "/" + fileName)
						cap.set(1,self.desiredFrameNum)
						ret, self.openImg = cap.read()
						self.frameNum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
						self.maxFrameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

						print("back in business!")

					

					

					print("reading frame", self.frameNum, "out of", self.maxFrameNum)


					if self.useSubRects:
						if str(self.frameNum) not in self.savableRects:
							self.savableRects[str(self.frameNum)] = {}
							print("no premade rectangles")
						else:
							print("frame rects", self.savableRects[str(self.frameNum)])

						print("there are", len(self.rectFrames), "frames with rectangles")
						self.subRectInd = 0
						self.maxSubRects = len(self.largerRects[str(self.frameNum)])
						ogImg = self.openImg.copy()
						while self.subRectInd < self.maxSubRects and self.subRectInd >= 0:
							if str(self.subRectInd) in self.savableRects[str(self.frameNum)]:
								self.currentRectangles["drawnRectangles"] = self.savableRects[str(self.frameNum)][str(self.subRectInd)]
								print("current rectangles", self.currentRectangles["drawnRectangles"])
							else:
								print("no rectangles found at", self.subRectInd, "in", self.savableRects[str(self.frameNum)])

							print("reading rectangle", self.subRectInd+1, "out of", self.maxSubRects)
							rect = self.largerRects[str(self.frameNum)][self.subRectInd]
							self.openImg = ogImg[rect[1]:rect[3], rect[0]:rect[2]]
							self.dispImg()
							change, rectangles = self.manageKeyResponses()
							self.subRectInd += change
							if rectangles != []:
								if str(self.frameNum) not in self.savableRects:
									self.savableRects[str(self.frameNum)] = {}
								self.savableRects[str(self.frameNum)][str(self.subRectInd)] = rectangles
								print("added rectangles")


						frameInd = self.rectFrames.index(self.frameNum)
						if self.subRectInd < 0:
							frameInd -= 1
							if frameInd < 0 and self.dirInd > 0:
								self.dirInd += 1
								print("going back a video", self.dirInd)
								break
							elif frameInd < 0:
								print("dirind", self.dirInd)
								print("can't go back any more")
								frameInd += 1


						else:
							frameInd += 1
							if frameInd >= len(self.rectFrames):
								self.dirInd += 1
								if self.dirInd >= len(self.dirs):
									print("no more videos")
									self.dirInd -= 1
								else:
									print("going to next video")
									break

						self.desiredFrameNum = self.rectFrames[frameInd]-1

					else:
						if str(self.frameNum) in self.savableRects:
							self.currentRectangles["drawnRectangles"] = self.savableRects[str(self.frameNum)][:]
						elif self.useDetector:
							self.detectPossible()

						self.dispImg()
					
						change, rectangles = self.manageKeyResponses()

						if len(rectangles) > 0:
							self.savableRects[str(self.frameNum)] = rectangles
						elif str(self.frameNum) in self.savableRects:
							self.savableRects.pop(str(self.frameNum))

						self.desiredFrameNum += change

						if self.desiredFrameNum >= self.maxFrameNum and self.maxFrameNum != 0:
							print("video ended")
							if self.dirInd < len(self.dirs):
								self.dirInd += 1
							break
						elif self.desiredFrameNum < 0:
							print("previous video")
							if self.dirInd > 0:
								self.dirInd -= 1

							break

				cap.release()




			elif fileName[-4::] in [".png", ".jpg"]:
				self.useSubRects = True
				self.openImg = cv2.imread(self.path + "/" + fileName)

				self.jsonName = fileName[0:-4] + self.labelName + ".json"
				if self.jsonName in self.allFiles:
					with open(self.path + '/' + self.jsonName) as json_file:
						data = json.load(json_file)
					print("loaded in", data)
					self.savableRects = data
				else:
					self.savableRects = {}
					print("no annotation data found --- ")
				self.unsavedChanges = False

				self.frameNum = 1
				if str(1) in self.savableRects:
					self.currentRectangles["drawnRectangles"] = self.savableRects[str(1)][:]
				elif self.useDetector:
					self.detectPossible()

				print("displaying image")
				self.dispImg()


				change, rectangles = self.manageKeyResponses()

				if rectangles != []:
					self.savableRects["1"] = rectangles

				self.dirInd += change
				self.useSubRects = False

			else:
				print("unknown file type")
				self.dirInd += 1

			self.saveDataJSON()

			if self.dirInd == len(self.dirs):
				print("no more videos!")
				self.dirInd -= 1


		img = np.zeros((300,300,3), dtype=np.uint8)

		cv2.imshow("img", img)
		k = 0
		while k != 27:
			k = cv2.waitKey(0)
			print("press escape")
		print("done")


	def saveDataJSON(self):
		if self.unsavedChanges:
			if self.savableRects != {}:
				jsonStr = json.dumps(self.savableRects)
				with open(self.path + "/" + self.jsonName, 'w') as fileHandle:
					fileHandle.write(str(jsonStr))
					fileHandle.close()
				print("Saving", jsonStr)
				print("saved", self.jsonName)

			elif self.jsonName in self.allFiles:
				os.remove(self.path + "/" + self.jsonName)
				print("had saved something, nothing left so deleted it")
			else:
				print("nothing was saved, nothing to save")
		else:
			print("no changes to save")
		self.unsavedChanges = False


	def saveImagesData(self):
		# save just the cropped images
		dirName = self.path + "/" + self.labelName + "_images"
		if not os.path.exists(dirName):
			os.makedirs(dirName)

		for fileName in self.dirs:
			jsonName = fileName[0:-4] + self.labelName + ".json"
			if jsonName in self.allFiles:

				with open(self.path + '/' + jsonName) as json_file:
					data = json.load(json_file)
				print("found data for", fileName)
				frameNums = [eval(i) for i in list(data.keys())]
				cap = cv2.VideoCapture(self.path + "/" + fileName)

				for frameNum in frameNums:
					cap.set(1,frameNum-1)
					ret, img = cap.read()
					if not ret:
						print("no frame found at frame number", frameNum)
						break

					
					i = 0
					for rectNum in data[str(frameNum)]:
						i+=1
						if rectNum[2] > rectNum[0] and rectNum[3] > rectNum[1]:

							saveName = fileName[0:-4] + "_f" + str(frameNum) + "_r" + str(i) + ".jpg"
							cv2.imwrite(dirName + "/" + saveName, img[rectNum[1]:rectNum[3], rectNum[0]:rectNum[2]])
							if True:
								cv2.imshow("saving", img[rectNum[1]:rectNum[3], rectNum[0]:rectNum[2]])
								cv2.waitKey(1)

		cv2.destroyWindow('saving')
		print("done saving")



	def saveNormalDataXML(self):
		self.saveDataJSON()

		print("saving xml files")
		trainDirName = self.path + "/" + self.labelName + "_train"
		if not os.path.exists(trainDirName):
			os.makedirs(trainDirName)
		validDirName = self.path + "/" + self.labelName + "_validate"
		if not os.path.exists(validDirName):
			os.makedirs(validDirName)

		showImgs = True
		imgCount = 0
		labelCount = 0
		for fileName in self.dirs:
			print(fileName)

			jsonName = fileName[0:-4] + self.labelName + ".json"
			if jsonName in self.allFiles:

				with open(self.path + '/' + jsonName) as json_file:
					data = json.load(json_file)
				print("found data for", fileName)

				if fileName[-4::] in [".mov", ".avi"]:
					frameNums = [eval(i) for i in list(data.keys())]
					cap = cv2.VideoCapture(self.path + "/" + fileName)

					for frameNum in frameNums:
						if imgCount%10 == 0:
							dirName = validDirName
						else:
							dirName = trainDirName
						imgCount += 1


						saveName = fileName[0:-4] + "_f" + str(frameNum) + ".jpg"

						cap.set(1,frameNum-1)
						ret, img = cap.read()
						if not ret:
							print("no frame found at frame number", frameNum)
							break

						if showImgs:
							if imgCount%10 == 0:
								cv2.imshow("saving", img)
								cv2.waitKey(1)

						cv2.imwrite(dirName + "/" + saveName, img)

						rects = data[str(frameNum)]
						labelCount += len(rects)

						saveTypes = [self.labelName] * len(rects)
						self.saveXML(dirName, saveName, rects, saveTypes, img.shape[0:2])

				elif fileName[-4::] == ".jpg":

					if "1" in data:
						print(data["1"])
						rects = data["1"]
						saveName = fileName
						saveTypes = [self.labelName] * len(rects)
						img = cv2.imread(self.path + "/" + fileName)
						cv2.imshow("saving", img)
						cv2.waitKey(1)
						labelCount += len(rects)
						imgCount += 1
						self.saveXML(self.path, saveName, rects, saveTypes, img.shape[0:2])
				else:
					print("unknown file", fileName, "type", fileName)
			


		if showImgs:
			cv2.destroyWindow('saving')


			
		print("saved", labelCount, "labels from", imgCount, "frames")


		print("Saved all")


	def saveXML(self, folder, fileName, rectangles, saveTypes, imshape):

		text = """<annotation>
		<folder>""" + folder + """</folder>
		<filename>""" + fileName[0:-4] + ".jpg" + """</filename>
		<path>""" + folder + "/"  + fileName[0:-4] + ".jpg" + """</path>
		<source>
		<database>Unknown</database>
		</source>
		<size>
		<width>""" + str(imshape[0]) + """</width>
		<height>""" + str(imshape[1]) + """</height>
		<depth>3</depth>
		</size>
		<segmented>0</segmented>"""

		i=0
		while i < len(rectangles):
			l = rectangles[i]
			text += """<object>
			<name>""" + saveTypes[i] + """</name>
			<pose>Unspecified</pose>
			<truncated>0</truncated>
			<difficult>0</difficult>
			<bndbox>
			<xmin>""" + str(l[0]) + """</xmin>
			<ymin>""" + str(l[1]) + """</ymin>
			<xmax>""" + str(l[2]) + """</xmax>
			<ymax>""" + str(l[3]) + """</ymax>
			</bndbox>
			</object>"""
			i+=1

		text += """\n</annotation>"""


		fileName = fileName[0:-4] + ".xml"

		if True:
			with open( folder + "/" + fileName, 'w') as fileHandle:
				fileHandle.write(text)
				fileHandle.close()


	def fixVideo(self, fileName):
		print("fixing the video. This might take a while, but you only have to do it once")
		oldFileName = fileName[0:-4] + "_corrupted.avi_bad"
		os.rename(fileName, oldFileName)
		cap = cv2.VideoCapture(oldFileName)

		frame_width = int(cap.get(3))
		frame_height = int(cap.get(4))
		   
		size = (frame_width, frame_height)

		frameCount = 0
		saveMethod = 'MJPG'
		colorWriter = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc(*saveMethod), 30, size)
		ret = cap.isOpened()
		while ret:
			
			ret, imgOG = cap.read()
			if ret:
				img = imgOG.copy()
				frameCount += 1
				if frameCount%10 == 0:
					cv2.imshow("img", img)
					colorWriter.write(img)
					k = cv2.waitKey(1)
					if k == 27:
						break

		cap.release()
		colorWriter.release()

		print("done")






annot = Annotator(videosPath, saveName, modelName, largerRectName)
annot.goThroughDir()
