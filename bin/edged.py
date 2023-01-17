#!/usr/bin/env python3
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("image_file", type=str)
args = parser.parse_args()


#thanks!
#https://qiita.com/shoku-pan/items/328edcde833307b164f4
#https://qiita.com/shoku-pan/items/07ec25f1d50629fed698

path = args.image_file

img_gray = cv2.imread(path,cv2.IMREAD_GRAYSCALE) # pathは画像を置いている場所を指定
img_canny = cv2.Canny(img_gray, 100, 200)
#cv2.imshow("image", img_canny)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite("./edged.jpg", img_canny)
