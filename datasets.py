import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
import glob

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']

class COCODataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, image_folder_path, json_file_path, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split
        self.split = split.upper()

        self.image_folder_path = image_folder_path
        self.json_file_path = json_file_path
        self.keep_difficult = keep_difficult

        #self.image_folder = os.path.join(self.data_folder, 'images')
        self.imgs = list(sorted(os.listdir(self.image_folder_path)))
        self.anno = json.load(open(self.json_file_path))
        self.annotation = self.anno["annotations"]

    def __getitem__(self, i):
        # Read image
        img_path = os.path.join(self.image_folder_path, self.imgs[i])
        image = Image.open(img_path).convert("RGB")
        height = image.height
        width = image.width
        
        target = {}
        boxes = []
        labels = []
        difficulties = []
        area = []
        image_id = []
        iscrowd = []

        for j in range (0,len(self.annotation)):
          if self.annotation[j]["image_id"] == i :

            self.annotation[j]["bbox"][2] = self.annotation[j]["bbox"][0] + self.annotation[j]["bbox"][2]
            self.annotation[j]["bbox"][3] = self.annotation[j]["bbox"][1] + self.annotation[j]["bbox"][3]

            boxes.append(self.annotation[j]["bbox"])
            labels.append(self.annotation[j]["category_id"]+1)
            difficulties.append(self.annotation[j]["difficulties"])
            area.append(self.annotation[j]["area"])
            iscrowd.append(self.annotation[j]["iscrowd"])

        # Read objects in this image (bounding boxes, labels, difficulties)
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)
        difficulties = torch.ByteTensor(difficulties)  # (n_objects)
        area = torch.FloatTensor(area)
        iscrowd = torch.ByteTensor(iscrowd)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
        area = area * (300*300)/(height*width)

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.as_tensor(i, dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = iscrowd 
        target["difficulties"] = difficulties

        return image, target

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        # images = list()
        # boxes = list()
        # labels = list()
        # difficulties = list()

        # for b in batch:
        #     images.append(b[0])
        #     boxes.append(b[1])
        #     labels.append(b[2])
        #     difficulties.append(b[3])

        # images = torch.stack(images, dim=0)

        # return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
        return tuple(zip(*batch))
