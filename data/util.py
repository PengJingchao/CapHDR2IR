import os
import math
import pickle
import random
import numpy as np
import torch
import cv2
import scipy.ndimage
import re

####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.npy']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    '''get image path list from lmdb meta info'''
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def get_image_paths(data_type, dataroot):
    '''get image path list
    support lmdb or image files'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return sizes, paths


###################### read images ######################
def _read_img_lmdb(env, key, size):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        if os.path.splitext(path)[1] == '.npy':
            img = np.load(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_img_lmdb(env, path, size)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def read_npy(path):
    return np.load(path)


def read_imgdata(path, ratio=255.0):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED) / ratio


def expo_correct(img, exposures, idx):
    floating_exposures = exposures - exposures[1]
    gamma = 2.24
    img_corrected = (((img ** gamma) * 2.0 ** (-1 * floating_exposures[idx])) ** (1 / gamma))
    return img_corrected


####################
# image processing
# process on numpy image
####################


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def augment_flow(img_list, flow_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            flow = flow[:, ::-1, :]
            flow[:, :, 0] *= -1
        if vflip:
            flow = flow[::-1, :, :]
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    rlt_img_list = [_augment(img) for img in img_list]
    rlt_flow_list = [_augment_flow(flow) for flow in flow_list]

    return rlt_img_list, rlt_flow_list


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def calculate_gradient(img, ksize=-1):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return sobelxy.astype(np.float32) / 255.


####################
# Functions
####################


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (
            absx <= 2)).type_as(
        absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()


def filtering(img_gray, r, eps):
    img = np.copy(img_gray)
    H = 1 / np.square(r) * np.ones([r, r])
    meanI = scipy.ndimage.correlate(img, H, mode='nearest')

    var = scipy.ndimage.correlate(img * img, H, mode='nearest') - meanI * meanI
    a = var / (var + eps)
    b = meanI - a * meanI

    meana = scipy.ndimage.correlate(a, H, mode='nearest')
    meanb = scipy.ndimage.correlate(b, H, mode='nearest')
    output = meana * img + meanb
    return output


def guided_filter(img_LR, r=5, eps=0.01):
    img = np.copy(img_LR)
    for i in range(3):
        img[:, :, i] = filtering(img[:, :, i], r, eps)
    return img


def scale_to_01(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    scaled_tensor = torch.div(tensor - min_val, max_val - min_val)
    return scaled_tensor


def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


def add_number_to_filename(filename, number):
    # 通过正则表达式提取文件名中的数字
    match = re.search(r'\d+\.', filename)
    if match:
        original_number = int(match.group().split('.')[0])
        new_number = original_number + number
        new_number_str = str(new_number).zfill(len(match.group().split('.')[0]))
        new_filename = filename.replace(str(match.group().split('.')[0]), new_number_str)
        return new_filename
    else:
        return "No number found in filename."


class RandomHorizontalFlip:
    """Horizontally flip the given image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be flipped.

        Returns:
            numpy.ndarray: Randomly flipped image.
        """
        if np.random.rand() < self.p:
            img = np.fliplr(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotation:
    """Rotate the image by angle randomly within a given range.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)
        self.degrees = degrees

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be rotated.

        Returns:
            numpy.ndarray: Rotated image.
        """
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)

        return rotated_img


class RandomColorDisturbance:
    """Randomly apply color disturbance to an image with a given probability.

    Args:
        probability (float): The probability that the operation will be performed.
        brightness_range (tuple): Range for brightness factor.
        contrast_range (tuple): Range for contrast factor.
        saturation_range (tuple): Range for saturation factor.
        hue_range (tuple): Range for hue shift.
    """

    def __init__(self, probability=0.3, brightness_range=(-25., 25.), contrast_range=(0.8, 1.2),
                 saturation_range=(0.8, 1.1), hue_range=(-10, 10)):
        self.probability = probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def __call__(self, ldr_img, hdr_img):
        if np.random.uniform(0, 1) > self.probability:
            return ldr_img, hdr_img

        # Randomly select a disturbance type
        disturbance_type = np.random.choice(['brightness_and_contrast', 'saturation', 'hue'])

        if disturbance_type == 'brightness_and_contrast':
            beta = np.random.uniform(*self.brightness_range)
            ldr_img = self.adjust_brightness_ldr(ldr_img, beta)
            hdr_img = self.adjust_brightness_hdr(hdr_img, beta)
            alpha = np.random.uniform(*self.contrast_range)
            ldr_img = self.adjust_contrast_ldr(ldr_img, alpha)
            hdr_img = self.adjust_contrast_hdr(hdr_img, alpha)
        elif disturbance_type == 'saturation':
            factor = np.random.uniform(*self.saturation_range)
            ldr_img = self.adjust_saturation(ldr_img, factor)
            hdr_img = self.adjust_saturation(hdr_img, factor)
        elif disturbance_type == 'hue':
            shift = np.random.uniform(*self.hue_range)
            ldr_img = self.adjust_hue(ldr_img, shift)
            hdr_img = self.adjust_hue(hdr_img, shift)

        return ldr_img, hdr_img

    def adjust_brightness_ldr(self, img, beta):
        return cv2.convertScaleAbs(img, beta=beta)
    def adjust_brightness_hdr(self, img, beta):
        return img + beta

    def adjust_contrast_ldr(self, img, alpha):
        return cv2.convertScaleAbs(img, alpha=alpha)
    def adjust_contrast_hdr(self, img, alpha):
        return img * alpha

    def adjust_saturation(self, img, factor):
        hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], factor)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    def adjust_hue(self, img, shift):
        hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + shift)  # % 180
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

class RandomErasing:
    """Randomly erase a rectangle region in an image.

    Args:
        probability (float): The probability that the operation will be performed.
        sl (float): Minimum proportion of erased area against input image.
        sh (float): Maximum proportion of erased area against input image.
        r1 (float): Minimum aspect ratio of erased area.
        mean (sequence): Mean value for each channel to fill the erased area.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=None):
        if mean is None:
            mean = [0.4914, 0.4822, 0.4465]
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean
        assert self.mean == [0., 0., 0.], "other values rather than 0 are not verified."

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be erased.

        Returns:
            numpy.ndarray: Image with randomly erased rectangle.
        """
        if np.random.uniform(0, 1) > self.probability:
            return img

        h, w = img.shape[:2]
        area = h * w

        for _ in range(100):
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)

            erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(math.sqrt(target_area / aspect_ratio)))

            if erase_w < w and erase_h < h:
                x1 = np.random.randint(0, h - erase_h)
                y1 = np.random.randint(0, w - erase_w)

                if img.ndim == 3 and img.shape[2] == 3:
                    img[x1:x1 + erase_h, y1:y1 + erase_w, 0] = self.mean[0] * 255
                    img[x1:x1 + erase_h, y1:y1 + erase_w, 1] = self.mean[1] * 255
                    img[x1:x1 + erase_h, y1:y1 + erase_w, 2] = self.mean[2] * 255
                else:
                    img[x1:x1 + erase_h, y1:y1 + erase_w] = self.mean[0] * 255
                return img

        return img

class RandomResizedCrop:
    """Crop a random portion of image and resize it to a given size.

    Args:
        size (int or sequence): expected output size of the crop, for each edge.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before resizing.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img_shape, scale, ratio):
        height, width = img_shape
        area = height * width

        for _ in range(10):
            target_area = area * np.random.uniform(scale[0], scale[1])
            aspect_ratio = np.exp(np.random.uniform(math.log(ratio[0]), math.log(ratio[1])))

            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = np.random.randint(0, height - crop_height + 1)
                left = np.random.randint(0, width - crop_width + 1)
                return top, left, crop_height, crop_width

        # Fallback to central crop
        in_ratio = width / height
        if in_ratio < min(ratio):
            crop_width = width
            crop_height = round(crop_width / min(ratio))
        elif in_ratio > max(ratio):
            crop_height = height
            crop_width = round(crop_height * max(ratio))
        else:  # whole image
            crop_width = width
            crop_height = height

        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        return top, left, crop_height, crop_width

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped and resized.

        Returns:
            numpy.ndarray: Randomly cropped and resized image.
        """
        height, width = img.shape[:2]
        top, left, crop_height, crop_width = self.get_params((height, width), self.scale, self.ratio)
        img = img[top:top + crop_height, left:left + crop_width]

        if isinstance(self.size, int):
            size = (self.size, self.size)
        else:
            size = self.size

        img = cv2.resize(img, size)
        return img

class AddNoise:
    """Add gamma correction noise to an image."""

    def __init__(self, prob=0.01, gamma_range=(0.5, 2.0)):
        """
        Args:
            prob (float): Probability of each pixel being affected by the noise.
            gamma_range (tuple): Range of gamma values to apply.
        """
        self.prob = prob
        self.gamma_range = gamma_range

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to which noise will be added.

        Returns:
            numpy.ndarray: Image with gamma correction noise added.
        """
        if np.random.uniform(0, 1) > 0.5:
            return img

        noisy_img = np.copy(img)
        noise_mask = np.random.rand(*img.shape[:2]) < self.prob

        # Generate random gamma value for each pixel
        gamma_values = np.random.uniform(self.gamma_range[0], self.gamma_range[1], size=(noisy_img.shape[0], noisy_img.shape[1], 1))

        # Apply gamma correction
        noisy_img[noise_mask] = np.clip(255 * ((noisy_img[noise_mask] / 255) ** gamma_values[noise_mask]), 0, 255)

        return noisy_img


from typing import Tuple, Optional


class RandomAffine:
    """Random affine transformation of the image keeping center invariant.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or number, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x-axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x-axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            an x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
    """

    def __init__(
            self,
            degrees,
            translate=None,
            scale=None,
            shear=None,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    @staticmethod
    def get_params(
            degrees: Tuple[float, float],
            translate: Optional[Tuple[float, float]],
            scale_ranges: Optional[Tuple[float, float]],
            shears: Optional[Tuple[float, float]],
            img_size: Tuple[int, int],
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(np.random.uniform(degrees[0], degrees[1]))
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(np.random.uniform(-max_dx, max_dx)))
            ty = int(round(np.random.uniform(-max_dy, max_dy)))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(np.random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(np.random.uniform(shears[0], shears[1]))
            if len(shears) == 4:
                shear_y = float(np.random.uniform(shears[2], shears[3]))

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def __call__(self, img):
        """
            img (numpy.ndarray): Image to be transformed.

        Returns:
            numpy.ndarray: Affine transformed image.
        """
        height, width, = img.shape[:2]
        img_size = (width, height)  # flip for keeping BC on get_params call

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        M = cv2.getRotationMatrix2D((width / 2, height / 2), ret[0], ret[2])
        M[:, 2] += np.array(ret[1])

        if ret[3][0] != 0.0 or ret[3][1] != 0.0:
            shear_matrix = np.array([[1, ret[3][0], 0],
                                     [ret[3][1], 1, 0],
                                     [0, 0, 1]])
            M = np.dot(shear_matrix, M)

        img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        return img


class Resize:
    def __init__(self, size, interpolation=cv2.INTER_LINEAR, max_size=None, antialias=True):
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
        else:  # Assuming PIL Image
            w, h = img.size

        if isinstance(self.size, int):
            if self.max_size is not None and max(h, w) > self.max_size:
                if h > w:
                    new_w = int(self.max_size * w / h)
                    img = cv2.resize(img, (new_w, self.max_size), interpolation=self.interpolation)
                else:
                    new_h = int(self.max_size * h / w)
                    img = cv2.resize(img, (self.max_size, new_h), interpolation=self.interpolation)
            else:
                img = cv2.resize(img, (self.size, self.size), interpolation=self.interpolation)
        elif isinstance(self.size, (tuple, list)) and len(self.size) == 2:
            img = cv2.resize(img, self.size, interpolation=self.interpolation)
        else:
            raise ValueError("Size should be an integer or a tuple/list of two integers.")

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation={self.interpolation}, max_size={self.max_size}, antialias={self.antialias})"
