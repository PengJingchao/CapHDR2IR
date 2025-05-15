import numpy as np
import cv2
import lmdb
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import data.util as util
import pickle


class HDR2IR_dataset(data.Dataset):

    def __init__(self, opt):
        """
            init function of the RGBTHDRDataset.

            Args:
                opt, which includes:
                    hdr_dir and t_dir (str): paths of hdr and infrared images
                    index_path (str): path of the index of the training or test set
                    data_len (int): length of the dataset, -1 means use all data from index
                    do_transform (bool): whether to do the data augmentation
                    lmdb_path (str): path of the lmdb (for hdr images), None means not using lmdb

            """
        super(HDR2IR_dataset, self).__init__()
        self.opt = opt
        do_transform = True if opt['do_transform'] else False
        self.hdr_images = []  # 存储HDR图像的列表
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

        # 读txt文件获取id
        with open(self.opt['index_path'], 'r') as file:
            for line in file:
                data_id = line.strip()

                hdr_image_path = os.path.join(opt['hdr_dir'], 'RGB' + data_id + '.hdr')
                self.hdr_images.append(hdr_image_path)

                t_image_path = os.path.join(opt['t_dir'], 'T' + data_id + '.tiff')
                self.t_images.append(t_image_path)

        self.lmdb_path = opt['lmdb_path'] if opt.get('lmdb_path', False) else None

    def open_lmdb(self):
        if self.lmdb_path is None:
            self.env = None
        else:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False,
                                 create=False)
            self.txn = self.env.begin(buffers=True, write=False)

    def __getitem__(self, index):
        hdr_image_path = self.hdr_images[index]
        if not hasattr(self, 'txn'):
            self.open_lmdb()

        if self.env is None:
            hdr_image = cv2.imread(hdr_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        else:
            imgbuf = self.txn.get(hdr_image_path.encode())
            mapped_image, shape = pickle.loads(imgbuf)
            hdr_image = mapped_image.reshape(shape)

        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)

        t_image_path = self.t_images[index]
        t_image = cv2.imread(t_image_path, cv2.IMREAD_GRAYSCALE)
        t_image = t_image.astype(np.float32) / 255.0

        seed = np.random.randint(4354356)

        # 对HDR和LDR图像进行数据增广/预处理
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        hdr_image = util.map_range(hdr_image)
        for t in self.transform:
            hdr_image = t(hdr_image)
        hdr_image = torch.from_numpy(hdr_image.transpose((2, 0, 1)))

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        for t in self.transform:
            t_image = t(t_image)
        t_image = torch.from_numpy(t_image).unsqueeze(0)

        return {
            'Input': hdr_image,
            'GT': t_image.expand_as(hdr_image),
            'Input_path': hdr_image_path,
            'GT_path': t_image_path
        }

    def __len__(self):
        return len(self.hdr_images)
def batch_process_hdr_images(hdr_images):
    """
    Mantiuk tone mapping method for batching images

    parameter:
    hdr_images (torch.tensor): HDR images with the size of B,C,H,W，

    return:
    sdr_images (torch.tensor): SDR images with the same size as hdr_images。
    """

    device = hdr_images.device
    tonemap = cv2.createTonemapMantiuk(gamma=2.6, scale=1.0, saturation=1.2)
    hdr_images = hdr_images.permute(0, 2, 3, 1).cpu().numpy()
    sdr_images = np.empty_like(hdr_images)

    # 遍历每张 HDR 图像进行色调映射
    for i in range(hdr_images.shape[0]):
        hdr_image = hdr_images[i]
        sdr_image = tonemap.process(hdr_image)
        sdr_image = np.nan_to_num(sdr_image, nan=0.0, posinf=1.0, neginf=0.0)  # nan ->  0, inf -> 1
        # sdr_image = hdr_image
        sdr_images[i] = np.clip(sdr_image, 0, 1)

    sdr_images = torch.from_numpy(sdr_images).permute(0, 3, 1, 2).to(device).float()

    sdr_images = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(sdr_images)

    return sdr_images

