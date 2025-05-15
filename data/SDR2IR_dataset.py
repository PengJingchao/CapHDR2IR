import numpy as np
import cv2
import lmdb
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import data.util as util
import pickle


class SDR2IR_dataset(data.Dataset):

    def __init__(self, opt):
        """
            init function of the RGBTHDRDataset.

            Args:
                opt, which includes:
                    hdr_dir, ldr_dir, and t_dir (str): paths of hdr, ldr, and infrared images
                    index_path (str): path of the index of the training or test set
                    data_len (int): length of the dataset, -1 means use all data from index
                    do_transform (bool): whether to do the data augmentation
                    lmdb_path (str): path of the lmdb (for hdr images), None means not using lmdb

            """
        super(SDR2IR_dataset, self).__init__()
        self.opt = opt
        do_transform = True if opt['do_transform'] else False
        self.ldr_images = []  # 存储LDR图像的列表
        self.t_images = []  # 存储红外图像的列表


        if do_transform:
            self.color_transform = util.RandomColorDisturbance(probability=0.3)
        else:
            self.color_transform = util.RandomColorDisturbance(probability=0)

        if do_transform:
            self.transform = [
                util.RandomErasing(probability=0.3, mean=[0., 0., 0.]),
                util.RandomHorizontalFlip(),  # Random horizontal flip
                util.RandomRotation(10),  # Random rotation
                util.RandomResizedCrop(opt['GT_size'], scale=(0.4, 1.0), ratio=(1.0, 1.0)),  # Random resized crop
                util.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),  # Random affine transformation
            ]

        else:
            self.transform = [
                util.Resize([opt['GT_size'], opt['GT_size']]),
            ]

        self.LDR_Normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 读txt文件获取id
        with open(self.opt['index_path'], 'r') as file:
            for line in file:
                data_id = line.strip()
                ldr_image_path = os.path.join(opt['ldr_dir'], 'RGB' + data_id + '.JPG')
                self.ldr_images.append(ldr_image_path)                                      # well-exposured image
                t_image_path = os.path.join(opt['t_dir'], 'T' + data_id + '.tiff')
                self.t_images.append(t_image_path)


    def __getitem__(self, index):

        ldr_image_path = self.ldr_images[index]
        ldr_image = cv2.imread(ldr_image_path)
        ldr_image = cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB)

        ldr_image = ldr_image.astype(np.float32) / 255.0

        t_image_path = self.t_images[index]
        t_image = cv2.imread(t_image_path, cv2.IMREAD_GRAYSCALE)
        t_image = t_image.astype(np.float32) / 255.0

        seed = np.random.randint(4354356)

        # 对HDR和LDR图像进行数据增广/预处理
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        for t in self.transform:
            ldr_image = t(ldr_image)
        ldr_image = torch.from_numpy(ldr_image.transpose((2, 0, 1)))
        ldr_image = self.LDR_Normalize(ldr_image)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        for t in self.transform:
            t_image = t(t_image)
        t_image = torch.from_numpy(t_image).unsqueeze(0)

        return {
            'Input': ldr_image,
            'GT': t_image.expand_as(ldr_image),
            'Input_path': ldr_image_path,
            'GT_path': t_image_path
        }

    def __len__(self):
        return len(self.ldr_images)

