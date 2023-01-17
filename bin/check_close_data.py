#!/usr/bin/env python3
import xml.etree.ElementTree as ET

import argparse
import subprocess
import glob
import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from detection_result import *

from collections import defaultdict


class ScoreCalc:
	def __init__(self, anno, resdata, exactness=0.1):
		self.exactness = exactness
		self.anno = anno
		self.resdata = resdata
		self.box_score_dict = defaultdict(lambda:0)
		self.label_score_dict = defaultdict(lambda:0)

	def box_score(self):
		score = defaultdict(lambda:0)
		for a in self.anno.objects:
			a.print()
			for d in self.resdata.res:
				w = a.width * self.exactness
				if a.x - w <= d.rect.x and d.rect.x <= a.x + w:
					print("  DEBUG: %s, good x" % (a.name))
					score[a] += 1
				h = a.height * self.exactness
				if a.y - h <= d.rect.y and d.rect.y <= a.y + h:
					print("  DEBUG: %s, good y" % (a.name))
					score[a] += 1
				if abs(a.width - d.rect.width) / float(a.width) <= self.exactness:
					print("  DEBUG: %s, good w" % (a.name))
					score[a] += 1
				if abs(a.height - d.rect.height) / float(a.height) <= self.exactness:
					print("  DEBUG: %s, good h" % (a.name))
					score[a] += 1

		self.box_score_dict = score

	def print_box_score(self):
		acc = 0.0
		for a, score in self.box_score_dict.items():
			print("anno_label=%s, score=%d" % (a.name, score))
			acc += score

		print("INFO: result of box score = %f" % (acc/(len(self.anno.objects)*4)))

		

	def label_score(self):
		pass

	def print_label_score(self):
		pass


class GAAAnnotationObject:
    def __init__(self, xml_element):
        self.name = xml_element.find("name").text
        self.xmin = int(xml_element.find("bndbox").find("xmin").text)
        self.ymin = int(xml_element.find("bndbox").find("ymin").text)
        self.xmax = int(xml_element.find("bndbox").find("xmax").text)
        self.ymax = int(xml_element.find("bndbox").find("ymax").text)
        self.width = abs(self.xmax-self.xmin)
        self.height = abs(self.ymax - self.ymin)
        self.x = self.xmin
        self.y = self.ymin

    def print(self):
        print("name:%s, xmin=%d, ymin=%d, xmax=%d, ymax=%d" % (self.name, self.xmin, self.ymin, self.xmax, self.ymax))


class GAAAnnotation:
	def __init__(self, file_name):
		self.root = ET.parse(xmlfile_name).getroot()
		self.objects = []

		for obj in self.root.findall("object"):
			self.objects.append(GAAAnnotationObject(obj))

	def print(self):
		for o in self.objects:
			o.print()

parser = argparse.ArgumentParser()
parser.add_argument("imgfile", type=str)
parser.add_argument("resdatafile", type=str)
args = parser.parse_args()

one_file = args.imgfile

dir_name = os.path.dirname(one_file)
basename = os.path.basename(one_file)

file_name = os.path.splitext(basename)[0]
ext_name  = os.path.splitext(basename)[1]

xmlfile_name = dir_name + "/" + file_name + ".xml"

anno = GAAAnnotation(xmlfile_name)
anno.print()

dres = DetectionResultContainer()
dres.load("./temp/result_data.pickle")
test_res1 = DetectionResult("close", 0.1, ((33,28), 36, 36))
dres.res.append(test_res1)
dres.print()

print("INFO: ScoreCalc")
calc = ScoreCalc(anno, dres)
calc.box_score()
print("INFO: print box score")
calc.print_box_score()



