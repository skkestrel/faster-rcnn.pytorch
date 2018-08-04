# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="/srv/share/jyang375/models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]']

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  pascal_classes = np.asarray(['__background__', 'Piano', 'Beer', 'Chopsticks', 'Cat', 'Snowmobile', 'Drill', 'Tiara', 'Motorcycle', 'Tennis racket', 'Football', 'Mobile phone', 'Flute', 'Tennis ball', 'Chair', 'Woman', 'Boy', 'Coffee table', 'Handbag', 'Fork', 'Girl', 'Coffee cup', 'Violin', 'Backpack', 'Knife', 'Snowboard', 'Rugby ball', 'Snake', 'Oven', 'Microwave oven', 'Man', 'Bench', 'Desk', 'Suitcase', 'Sofa bed', 'Horse', 'Hat', 'Microphone', 'Table tennis racket', 'Monkey', 'Surfboard', 'Table', 'Bottle', 'Mug', 'Wine glass', 'Drum', 'Dog', 'Guitar', 'Ski', 'Bed', 'Spoon', 'Briefcase', 'Hamster', 'Envelope', 'Car', 'Bicycle', 'Camera', 'Dolphin', 'Taxi', 'Van', 'Elephant', 'Racket', 'Pretzel'])
  class_dict = {'Tiara': '/m/01krhy', 'Coffee cup': '/m/02p5f1q', 'Drill': '/m/01d380', 'Hamster': '/m/03qrc', 'Guitar': '/m/0342h', 'Flute': '/m/0l14j_', 'Camera': '/m/0dv5r', 'Football': '/m/01226z', 'Table': '/m/04bcr3', 'Boy': '/m/01bl7v', 'Briefcase': '/m/0584n8', 'Ski': '/m/071p9', 'Taxi': '/m/0pg52', 'Rugby ball': '/m/0wdt60w', 'Car': '/m/0k4j', 'Bottle': '/m/04dr76w', 'Oven': '/m/029bxz', 'Tennis racket': '/m/0h8my_4', 'Coffee table': '/m/078n6m', 'Wine glass': '/m/09tvcd', 'Girl': '/m/05r655', 'Elephant': '/m/0bwd_0j', 'Table tennis racket': '/m/05_5p_0', 'Dolphin': '/m/02hj4', 'Snowmobile': '/m/01x3jk', 'Monkey': '/m/08pbxl', 'Beer': '/m/01599', 'Cat': '/m/01yrx', 'Knife': '/m/04ctx', 'Surfboard': '/m/019w40', 'Envelope': '/m/0frqm', 'Snowboard': '/m/06__v', 'Tennis ball': '/m/05ctyq', 'Violin': '/m/07y_7', 'Piano': '/m/05r5c', 'Mobile phone': '/m/050k8', 'Snake': '/m/078jl', 'Bed': '/m/03ssj5', 'Horse': '/m/03k3r', 'Fork': '/m/0dt3t', 'Drum': '/m/026t6', 'Pretzel': '/m/01f91_', 'Chair': '/m/01mzpv', 'Racket': '/m/0dv9c', 'Van': '/m/0h2r6', 'Microphone': '/m/0hg7b', 'Spoon': '/m/0cmx8', 'Chopsticks': '/m/01_5g', 'Desk': '/m/01y9k5', 'Hat': '/m/02dl1y', 'Man': '/m/04yx4', 'Mug': '/m/02jvh9', 'Bicycle': '/m/0199g', 'Backpack': '/m/01940j', 'Suitcase': '/m/01s55n', 'Bench': '/m/0cvnqh', 'Dog': '/m/0bt9lr', 'Motorcycle': '/m/04_sv', 'Woman': '/m/03bt1vf', 'Microwave oven': '/m/0fx9l', 'Sofa bed': '/m/03m3pdh', 'Handbag': '/m/080hkjn'}





  # initilize the network here.
  if args.net == 'vgg16':
    print(pascal_classes)
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))

  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.05

  imglist = os.listdir(args.image_dir)
  num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))


 
  with open("detections.csv", "w") as out:
    out.write("ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,Label\n")
    while (num_images >= 0):
      total_tic = time.time()
      num_images -= 1

      im_file = os.path.join(args.image_dir, imglist[num_images])
      # im = cv2.imread(im_file)
      im_in = np.array(imread(im_file))
      if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
      # rgb -> bgr
      im = im_in[:,:,::-1]

      blobs, im_scales = _get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
      im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
      gt_boxes.data.resize_(1, 1, 5).zero_()
      num_boxes.data.resize_(1).zero_()

      # pdb.set_trace()
      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      im2show = np.copy(im)
      for j in xrange(1, len(pascal_classes)):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            for k in range(cls_dets.shape[0]):
              out.write("{0},none,{1},{2},{3},{4},{5},{6},1,0,0,0,0,{1}\n".format( \
                   imglist[num_images][:-4], class_dict[pascal_classes[j]], cls_dets[k, -1], cls_dets[k, 0] / im.shape[1], cls_dets[k, 2] / im.shape[1], cls_dets[k, 1] / im.shape[0], cls_dets[k, 3] / im.shape[0]))
            # im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                           .format(num_images + 1, len(imglist), detect_time, nms_time))
      sys.stdout.flush()

      # result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
      # cv2.imwrite(result_path, im2show)