# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid

class openimages(imdb):
  def __init__(self, image_set):
    imdb.__init__(self, 'openimages_2018_' + image_set)
    # name, paths
    self._image_set = image_set
    self._data_path = osp.join(cfg.DATA_DIR, 'openimages_2018')

    # load COCO API, classes, class <-> id mappings
    self._read_ann_file()
    self._load_image_set_index()
    cats = self._load_classes()

    self._classes = tuple(['__background__'] + [c[1] for c in cats])
    self._class_to_ind = dict(list(zip(self._classes, list(range(self.num_classes)))))
    self._class_to_coco_cat_id = dict(list(zip([c[1] for c in cats], [c[0] for c in cats])))

    # Default to roidb handler
    self.set_proposal_method('gt')
    self.competition_mode(False)

    # Some image sets are "views" (i.e. subsets) into others.
    # This mapping tells us where the view's images and proposals come from.
    self._view_map = {
      'val2018': 'train2018', 
      'trainval2018': 'train2018', 
    }
    coco_name = image_set + "2018"  # e.g., "val2014"
    self._data_name = (self._view_map[coco_name]
                       if coco_name in self._view_map
                       else coco_name)

    # Dataset splits that have ground-truth annotations (test splits
    # do not have gt annotations)
    self._gt_splits = ('train', 'val', 'trainval')

  def _load_classes(self):
    cats = []
    with open(osp.join(self._data_path, 'challenge-2018-classes-vrd.csv'), "r") as f:
      line = f.readline()
      while line != "":
        line = line.strip().split(",")
        cats.append((line[0], line[1]))
        line = f.readline()
    return cats

  def _get_ann_files(self):
    return ("challenge-2018-train-vrd-bbox.csv", "image-sizes")

  def _load_image_set_index(self):
    """
    Load image ids.
    """
    return [i[0] for i in self._imsize]

  def _get_widths(self):
    return [i[1] for i in self._imsize]

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_id_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self._image_index[i]

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # Example image path for index=119993:
    #   images/train2014/COCO_train2014_000000119993.jpg
    file_name = index + '.jpg'
    image_path = osp.join(self._data_path, 'images',
                          self._data_name, file_name)
    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_openimages_annotation(i)
                for i in range(len(self._image_index))]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _read_ann_file(self):
    self._annotations = []
    self._imsize = []
    with open(osp.join(self._data_path, self._get_ann_file()[0])) as f:
      if f.readline() != "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,Label"
        raise Exception()
      line = f.readline()
      while line != "":
        line = line.[:-1].split(',')
# imageID, labelName, xmin, xmax, ymin, ymax
        tup = (line[0], line[2], float(line[4]), float(line[5]), float(line[6]), float(line[7]))

        # LabelName = Label
        if line[2] != line[-1]:
          raise Exception("?")
        # Confidence = 1
        if line[3] != "1":
          raise Exception("?")

        self._annotations.append(tup)
        line = f.readline()

    with open(osp.join(self._data_path, self._get_ann_file()[1])) as f:
      line = f.readline()
      while line != "":
        line = line.[:-1].split(',')

        # imageid, width, height
        tup = (line[0], int(line[1]), int(line[2]))

        self._imsize.append(tup)
        line = f.readline()

  def _load_openimages_annotation(self, i):
    """
    Loads openimages bounding-box instance annotations. Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This
    overlap value means that crowd "instances" are excluded from training.
    """
    width = self._imsize[i][1]
    height = self._imsize[i][2]

    index = self._image_index[i]
    objs = [entry for entry in self._annotations if entry[0] == index]

    # Sanitize bboxes -- some are invalid
    valid_objs = []
    for obj in objs:
      x1 = np.max((0, int(width * obj[2])))
      x2 = np.min((width - 1, int(width * obj[3])))
      y1 = np.max((0, int(height * obj[4])))
      y2 = np.max((height - 1, int(height * obj[5])))

      # TODO are the BB cooridnates diferent?

      if x2 >= x1 and y2 >= y1:
        valid_objs.append((obj[0], obj[1], x1, y1, x2, y2))

    objs = valid_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Lookup table to map from COCO category ids to our internal class
    # indices
    coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                      self._class_to_ind[cls])
                                     for cls in self._classes[1:]])

    for ix, obj in enumerate(objs):
      cls = coco_cat_id_to_class_ind[obj[1]]
      boxes[ix, :] = [obj[2], obj[3], obj[4], obj[5]]
      gt_classes[ix] = cls
      seg_areas[ix] = (obj[4] - obj[2]) * (obj[5] - obj[3])

      # TODO what does overlaps do?
      overlaps[ix, cls] = 1.0

    ds_utils.validate_boxes(boxes, width=width, height=height)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_widths(self):
    return [r['width'] for r in self.roidb]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}

      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def _get_box_file(self, index):
    # first 14 chars / first 22 chars / all chars + .mat
    # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
    file_name = 'OI_' + index + '.mat'
    return osp.join(file_name[:4], file_name[:8], file_name)

  def evaluate_detections(self, all_boxes, output_dir):
    pass

  def competition_mode(self, on):
    pass
