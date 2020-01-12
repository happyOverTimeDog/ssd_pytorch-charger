from voc0712 import *
import Config
from detection import *
from ssd_net_vgg import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import SIXray_ROOT, SIXrayAnnotationTransform, SIXrayDetection, BaseTransform
from data import SIXray_CLASSES as labelmap

import os
import argparse
import numpy as np
import cv2
import utils


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default="weights/ssd_voc_100EPOCH.pth", type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder',
                    default="./eval/", type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.2, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=Config.use_cuda, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--SIXray_root', default=SIXray_ROOT,type=str,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_known_args()
args=args[0]
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        # print("WARNING: It looks like you have a CUDA device, but aren't using \
        #         CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

YEAR = '2007'

devkit_path = args.save_folder
dataset_mean = (104, 117, 123)
set_type = 'test'

def  test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    det_result_core_file= 'predicted_file/det_test_带电芯充电宝3.txt'
    det_result_core_less_file = 'predicted_file/det_test_不带电芯充电宝3.txt'
    num_images = len(dataset)
    # all_boxes = [[[] for _ in range(num_images)]
    #              for _ in range(len(labelmap) + 1)]

    # output_dir = get_output_dir(args.SIXray_root, set_type)
    # det_file = os.path.join(output_dir, 'detections.pkl')
    for root, dirs, files in os.walk(dataset):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for file in files:
            file_with_path = os.path.join(root, file)

            #print('Testing image {:d}/{:d}....'.format(id+1, num_images))
            #a = dataset.pull_item(id)
            #im, gt, h, w, im_og, img_id= dataset.pull_item(id)
            # 这里im的颜色偏暗，因为BaseTransform减去了一个mean
            # im_saver = cv2.resize(im[(a2,a1,0),:,:].permute((a1,a2,0)).numpy(), (w,h))
            im_og = cv2.imread(file_with_path, cv2.IMREAD_COLOR)
            x = cv2.resize(im_og, (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            # plt.imshow(x)
            x = torch.from_numpy(x).permute(2, 0, 1)
            xx = Variable(x.unsqueeze(0))
            if args.cuda:
                xx = x.cuda()
            y = net(xx)
            softmax = nn.Softmax(dim=-1)
            detect = Detect(config.class_num, 0, 200, 0.01, 0.45)
            priors = utils.default_prior_box()

            loc, conf = y
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

            detections = detect(
                loc.view(loc.size(0), -1, 4),
                softmax(conf.view(conf.size(0), -1, config.class_num)),
                torch.cat([o.view(-1, 4) for o in priors], 0)
            ).data
            labels = ['core', 'coreless']

            # plt.imshow(rgb_image)  # plot the image for matplotlib

            # scale each detection back up to the image
            scale = torch.Tensor(im_og.shape[1::-1]).repeat(2)
            for i in range(detections.size(1)):
                for j in range(detections.size(2)):
                    if detections[0, i, j, 0] >= 0.01:
                        score = detections[0, i, j, 0]
                        label_name = labels[i - 1]
                        display_txt = '%s: %.2f' % (label_name, score)
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                        color = (0,0,255)
                        cv2.rectangle(im_og, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
                        cv2.putText(im_og, display_txt, (int(pt[0]), int(pt[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 0, 0), 1, 8)
                        text = '%s %.2f %.2f %.2f %.2f %.2f' %(file[:-4], score, pt[0], pt[1], pt[2], pt[3])
                        if label_name == 'core':
                            with open(det_result_core_file, 'a+') as f:
                                f.write(text + '\n')
                        if label_name == 'coreless':
                            with open(det_result_core_less_file, 'a+') as f:
                                f.write(text + '\n')

            # if label_name == 'core':
            # cv2.imshow('test',im_og)
            # cv2.waitKey(0)

def test_one_picture(picture_path):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),(158, 218, 229),(158, 218, 229)]

    net = SSD()    # initialize SSD
    net = torch.nn.DataParallel(net)
    net.train(mode=False)
    net.load_state_dict(torch.load('./weights/ssd_voc_100EPOCH.pth',map_location=lambda storage, loc: storage))
    image = cv2.imread(picture_path, cv2.IMREAD_COLOR)
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    # plt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    softmax = nn.Softmax(dim=-1)
    detect = Detect(config.class_num, 0, 200, 0.01, 0.45)
    priors = utils.default_prior_box()

    loc,conf = y
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    detections = detect(
        loc.view(loc.size(0), -1, 4),
        softmax(conf.view(conf.size(0), -1,config.class_num)),
        torch.cat([o.view(-1, 4) for o in priors], 0)
    ).data

    labels = VOC_CLASSES

    # scale each detection back up to the image
    scale = torch.Tensor(image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        for j in range(detections.size(2)):
            if detections[0,i,j,0] >= 0.1:
                score = detections[0,i,j,0]
                label_name = labels[i-1]
                display_txt = '%s: %.2f'%(label_name, score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                color = colors_tableau[i]
                cv2.rectangle(image,(pt[0],pt[1]), (pt[2],pt[3]), color, 2)
                cv2.putText(image, display_txt, (int(pt[0]), int(pt[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
    cv2.imshow('test',image)
    cv2.waitKey(100000)


def test(img_path, anno_path):
    # load net
    num_classes = len(labelmap) + 1  # +a1 for background
    net = SSD()  # initialize SSD
    net = torch.nn.DataParallel(net)
    net.train(mode=False)
    net.load_state_dict(torch.load('./weights/ssd_voc_100EPOCH.pth', map_location=lambda storage, loc: storage))
    # print('Finished loading model!')
    # load data

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_data_path = img_path
    test_net(devkit_path, net, args.cuda, test_data_path,
             BaseTransform(Config.image_size, Config.MEANS),
             args.top_k, 300, thresh=args.confidence_threshold)



if __name__ == '__main__':
    img_path = 'map/Image_level3'
    anno_path = ''
    test(img_path, anno_path)
    pass