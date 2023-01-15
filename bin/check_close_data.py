#!/usr/bin/env python3
import xml.etree.ElementTree as ET

import argparse
import subprocess
import glob
import os

class GAAAnnotationObject:
    def __init__(self, xml_element):
        self.name = xml_element.find("name").text
        self.xmin = int(xml_element.find("bndbox").find("xmin").text)
        self.ymin = int(xml_element.find("bndbox").find("ymin").text)
        self.xmax = int(xml_element.find("bndbox").find("xmax").text)
        self.ymax = int(xml_element.find("bndbox").find("ymax").text)

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
args = parser.parse_args()

one_file = args.imgfile

dir_name = os.path.dirname(one_file)
basename = os.path.basename(one_file)

file_name = os.path.splitext(basename)[0]
ext_name  = os.path.splitext(basename)[1]

xmlfile_name = dir_name + "/" + file_name + ".xml"

GAAAnnotation(xmlfile_name).print()

