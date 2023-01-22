
import os
import cv2

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
 
from detection_result import *

import uuid


class ImageLogger:
	def __init__(self, log_dir_name="image_log"):
		self.log_dir_name = log_dir_name

	def log(self, image_file, detection_result_file):
		#check logging dir exists
		if not os.path.exists(self.log_dir_name):
			print("log_dir_name %s is not exists. now making it" % (self.log_dir_name))
			os.mkdir(self.log_dir_name)

		#generate uuid
		dir_name = self.log_dir_name + "/" + str(uuid.uuid4()) 
		#check logging dir exists
		if not os.path.exists(dir_name):
			os.mkdir(dir_name)
			
		#load image file and detection_result
		image = cv2.imread(image_file, cv2.IMREAD_COLOR) 
		dres = DetectionResultContainer()
		dres.load(detection_result_file)
		
		#iterate detection_result and extract image from input
		i = 0
		for d in dres.res:
			r = d.rect
			temp = image[r.y:r.y+r.height, r.x:r.x+r.width, :]
			score = str(int(d.score * 100))
			cv2.imwrite(dir_name + "/%s_%s_%d.jpg" % (d.label, score, i),temp)


if __name__ == "__main__":
	print("INFO: mini test")
	img_file =   "temp/img_logger_test/ru_Screenshot_2022-12-04-23-53-51-24_56bd83b73c18fa95b476c6c0f96c6836.jpg"
	dres_file = "temp/img_logger_test/result_data.pickle"

	img_logger = ImageLogger("test_img_logger")
	img_logger.log(img_file, dres_file)
