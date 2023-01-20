import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
 
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from ssd import build_ssd
from matplotlib import pyplot as plt
from data import VOC_CLASSES as voc_labels
from detection_result import *

# GPUの設定
torch.cuda.is_available() 
torch.set_default_tensor_type('torch.FloatTensor')  
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# SSDネットワークを定義し、学習済みパラメータを読み込む
net = build_ssd('test', 300, 21)   
#net.load_weights('./weights/ssd300_mAP_77.43_v2.pth')
#net.load_weights('./weights/BCCD.pth')
#net.load_weights('./weights/close_weight.pth')
#net.load_weights('./weights/close_weight_1.0647010359653208.pth')
#net.load_weights('./weights/good_backup/20221212/close_weight.pth')
#net = net.to(device)


# 物体検出関数 
def detect(image, labels):

    res = DetectionResultContainer()

    # 画像を(1,3,300,300)のテンソルに変換
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(image, (300, 300)).astype(np.float32)  
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)  
    xx = Variable(x.unsqueeze(0))    
     
    # 順伝播を実行し、推論結果を出力
    if torch.cuda.is_available():
      xx = xx.cuda()
    y = net(xx)

    # 表示設定 
    plt.figure(figsize=(8,8))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)
    currentAxis = plt.gca()

    # 推論結果をdetectionsに格納
    detections = y.data
    # 各検出のスケールのバックアップ
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    
    # バウンディングボックスとクラス名を表示
    for i in range(detections.size(1)):
        j = 0
        # 確信度confが0.6以上のボックスを表示
        # jは確信度上位200件のボックスのインデックス
        # detections[0,i,j]は[conf,xmin,ymin,xmax,ymax]の形状
        while detections[0,i,j,0] >= 0.1:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            res.add(label_name, score, coords)
            j+=1
    #plt.show()
    plt.savefig("./result.jpg")
    plt.close()
    return res

net.load_weights(sys.argv[1])
net = net.to(device)

# 物体検出実行
#file = './data/person.jpg'
#file = './data/bccd.jpg'
#file = './data/BloodImage_00219.jpg'
#file = './VOCdevkit/BCCD/JPEGImages/BloodImage_00000.jpg'
file = sys.argv[2]
image = cv2.imread(file, cv2.IMREAD_COLOR) 
res = detect(image, voc_labels)

res.sort_by_score()
res.print()
res.save("result_data.pickle")
