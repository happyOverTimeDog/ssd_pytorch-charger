# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:44:26 2019

@author: DELL
"""
import os
import numpy as np
import pickle
import cv2
import argparse

labelmap = (
    '带电芯充电宝',
    '不带电芯充电宝',
)
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument('--predicted_file', default='predicted_file', type=str, help='Checkpoint state_dict file for transfer learning')
args = parser.parse_args()

annopath = os.path.join('map/Anno_level3', '%s.xml')#测试集标注文件路径
test_imagesetfile = 'map/core_coreless_level3.txt'#测试集里面的图片名
'''
def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path
'''


def do_python_eval(output_dir='output', use_07=False):
    cachedir = 'annotations_cache'
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    # if not os.path.isdir(output_dir):
    #    os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        # filename = get_voc_results_file_template('test', cls)
        # filename = 'det_test_橙色液体.txt'
        filename = os.path.join(args.predicted_file, 'det_' + 'test' + '_%s3.txt' % cls)
        rec, prec, ap = voc_eval(
            filename, annopath, test_imagesetfile, cls, cachedir,
            ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        #    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))

    # f1.writelines(e + 'epoch: ' + '{:.3f}'.format(np.mean(aps)) + '\n')

    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    '''
    imagenames = []
    listdir = os.listdir(osp.join('%s' % args.SIXray_root, 'Annotation'))
    for name in listdir:
        imagenames.append(osp.splitext(name)[0])
    '''

    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath % (imagename))
        if i % 100 == 0:
            print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))
        # save
    # print('Saving cached annotations to {:s}'.format(cachefile))
    # with open(cachefile, 'wb') as f:
    #    pickle.dump(recs, f)
    '''
    else:
        #print('no,no,no')
        # load
    '''
    # with open(cachefile, 'rb') as f:
    #    recs = pickle.load(f)

    # print (recs)
    # print (classname)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]

        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # print (class_recs)

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    # tree = ET.parse(filename)
    # filename = filename[:-3] + 'txt'

    filename = filename.replace('.xml', '.txt')
    imagename0 = filename.replace('Anno', 'Image')

    imagename1 = imagename0.replace('.txt', '.TIFF')
    imagename2 = imagename0.replace('.txt', '.tiff')
    imagename3 = imagename0.replace('.txt', '.jpg')
    objects = []
    # 还需要同时打开图像，读入图像大小
    img = cv2.imread(imagename1)
    if img is None:
        img = cv2.imread(imagename2)
    if img is None:
        img = cv2.imread(imagename3)
    # print(str(img) +  ' ||||||||||| '+imagename3)
    try:
        height, width, channels = img.shape
    except:
        print('error')
    with open(filename, "r", encoding='utf-8') as f1:
        dataread = f1.readlines()
        for annotation in dataread:
            obj_struct = {}
            temp = annotation.split()
            name = temp[1]
            if name not in labelmap:
                continue
            xmin = int(temp[2])
            # 只读取V视角的
            # if int(xmin) > width/2:
            try:
                if int(xmin) > width:
                    continue
            except:
                print('errore')
            if xmin < 0:
                xmin = 1
            ymin = int(temp[3])
            if ymin < 0:
                ymin = 1
            xmax = int(temp[4])
            if xmax > width:  # 是这么个意思吧？
                xmax = width - 1
            ymax = int(temp[5])
            if ymax > height:
                ymax = height - 1
            ##name

            obj_struct['name'] = name
            obj_struct['pose'] = 'Unspecified'
            obj_struct['truncated'] = 0
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [float(xmin) - 1,
                                  float(ymin) - 1,
                                  float(xmax) - 1,
                                  float(ymax) - 1]
            objects.append(obj_struct)

    '''
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text.lower().strip()
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [float(bbox.find('xmin').text) - 1,
                              float(bbox.find('ymin').text) - 1,
                              float(bbox.find('xmax').text) - 1,
                              float(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)
    '''
    # print(objects)
    return objects


do_python_eval()






