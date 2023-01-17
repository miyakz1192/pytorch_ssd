from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import warnings  
warnings.filterwarnings('ignore')  

# 初期設定
args = {'dataset':'close_weight',   
        'basenet':'vgg16_reducedfc.pth',
        'batch_size':32,
        'resume':'ssd300_mAP_77.43_v2.pth',
        #'resume':'good_backup/close_weight_0.4379034004363406_20221227.pth',
        'max_iter':500,
        'num_workers':4,  
        'cuda':True,
        'lr':0.001,
        'lr_steps':(8000, 10000, 12000),  
        'momentum':0.9,
        'weight_decay':5e-4,
        'gamma':0.1,
        'save_folder':'weights/'
       }

# Tensor作成時のデフォルトにGPU Tensorを設定
if torch.cuda.is_available():
    if args['cuda']:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args['cuda']:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# 訓練データの設定
cfg = voc
dataset = VOCDetection(root=VOC_ROOT,
                       transform=SSDAugmentation(cfg['min_dim'],
                                                 MEANS))

print("INFO: num_classes = %d" % (cfg["num_classes"]))

# ネットワークの定義
ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ssd_net.to(device)

# 学習済みパラメータのロード
if args['resume']:
    print('Resuming training, loading {}...'.format(args['resume']))
    ssd_net.load_weights(args['save_folder'] + args['resume'])  
else:
    vgg_weights = torch.load(args['save_folder'] + args['basenet'])
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

# GPU設定
if args['cuda']:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

# learning_rate の段階調整関数
def adjust_learning_rate(optimizer, gamma, step):
    lr = args['lr'] * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# xavierの初期化関数
def xavier(param):
    init.xavier_uniform_(param)

# パラメータ初期化関数
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

# 新規学習時のパラメータ初期化
if not args['resume']:
    print('Initializing weights...')
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

# 損失関数の設定
criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                         False, args['cuda'])

# 最適化手法の設定
optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=args['momentum'],
                      weight_decay=args['weight_decay'])

# 訓練モード
net.train()

# データローダの設定
data_loader = data.DataLoader(dataset, args['batch_size'],
                              num_workers=args['num_workers'],
                              shuffle=True, collate_fn=detection_collate,
                              pin_memory=True)
# 学習ループ
step_index = 0
batch_iterator = None
epoch_size = len(dataset) // args['batch_size']

best_loss = 100000000 # 適当
best_file_name = None


for iteration in range(args['max_iter']):   
    if (not batch_iterator) or (iteration % epoch_size ==0):
        batch_iterator = iter(data_loader)
        loc_loss = 0
        conf_loss = 0

    # lrの調整
    if iteration in args['lr_steps']:
        step_index += 1
        adjust_learning_rate(optimizer, args['gamma'], step_index)
        
    # バッチサイズ分のデータをGPUへ
    images, targets = next(batch_iterator)
    images = images.to(device) 
    targets = [ann.to(device) for ann in targets]

    # 順伝播
    t0 = time.time()
    out = net(images)

    # 逆伝播
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, targets)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    t1 = time.time()
    loc_loss += loss_l.item()
    conf_loss += loss_c.item()
    
    #ログの出力
#    if iteration % 10 == 0:
#        print('timer: %.4f sec.' % (t1 - t0))
#        print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

    print('timer: %.4f sec.' % (t1 - t0))
    print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

    if best_loss > loss.item():
        best_loss = loss.item()
        print("best_loss updated")
        if best_file_name != None:
            os.remove(best_file_name)
        torch.save(ssd_net.state_dict(),
            args['save_folder'] + '' + args['dataset'] + "_" + str(best_loss)+'.pth')
        best_file_name = args['save_folder'] + '' + args['dataset'] + "_" + str(best_loss)+'.pth'


    sys.stdout.flush()


# 学習済みモデルの保存
torch.save(ssd_net.state_dict(),
           args['save_folder'] + '' + args['dataset'] + '.pth')
