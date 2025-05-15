import torch
import torch.nn as nn
from torch.nn import functional as F

# Matrix definitions
CovRGB2LMS = torch.tensor([[0.3811, 0.5783, 0.0402],
                           [0.1967, 0.7244, 0.0782],
                           [0.0241, 0.1288, 0.8444]])

CovLMS2lab = torch.tensor([[0.577350, 0.577350, 0.577350],
                           [0.408248, 0.408248, -0.816497],
                           [0.707107, -0.707107, 0.000000]])


def rgb2lab(rgb):
    assert rgb.size(1) == 3, "In the rgb2lab function, the image size must be B, 3, H, W!"
    lms = torch.einsum('bchw,cd->bdhw', rgb, CovRGB2LMS.t().to(rgb.device))
    loglms = torch.log10(lms)
    lab = torch.einsum('bchw,cd->bdhw', loglms, CovLMS2lab.t().to(loglms.device))
    lab[torch.isnan(lab)] = 0
    lab[lab == float('-inf')] = 0
    return lab


class lab_L1Loss(nn.Module):
    def __init__(self):
        super(lab_L1Loss, self).__init__()

    def forward(self, x, y):
        return F.smooth_l1_loss(rgb2lab(x), rgb2lab(y))

class gradient_smooth_l1loss(nn.Module):
    def __init__(self):
        super(gradient_smooth_l1loss, self).__init__()

    def forward(self, target_image, original_image):
        # 计算目标图像在x和y方向上的梯度
        target_dx = torch.abs(target_image[:, :, :, :-1] - target_image[:, :, :, 1:])
        target_dy = torch.abs(target_image[:, :, :-1, :] - target_image[:, :, 1:, :])

        # 计算原始图像在x和y方向上的梯度
        original_dx = torch.abs(original_image[:, :, :, :-1] - original_image[:, :, :, 1:])
        original_dy = torch.abs(original_image[:, :, :-1, :] - original_image[:, :, 1:, :])


        # 计算加权平滑损失
        smoothness = (F.smooth_l1_loss(target_dx, original_dx) +
                      F.smooth_l1_loss(target_dy, original_dy)) / 2.0


        # smoothness = torch.mean(target_dx) + torch.mean(target_dy)# - torch.mean(original_dx) - torch.mean(original_dy)

        return F.smooth_l1_loss(target_image, original_image) + 0.5 * smoothness

class tanh_L1Loss(nn.Module):
    def __init__(self):
        super(tanh_L1Loss, self).__init__()

    def forward(self, x, y):
        return F.smooth_l1_loss(torch.tanh(x), torch.tanh(y))


class tanh_L2Loss(nn.Module):
    def __init__(self):
        super(tanh_L2Loss, self).__init__()

    def forward(self, x, y):
        return F.mse_loss(torch.tanh(x), torch.tanh(y))


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class VGGLoss(nn.Module):
    def __init__(self, ):
        super(VGGLoss, self).__init__()
        try:
            self.vgg = vgg19().cuda()
        except:
            self.vgg = vgg19()
            print('CUDA is not available when loading vgg19, please check!')
        self.criterion = nn.SmoothL1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class vgg19(torch.nn.Module):
    def __init__(self, ):
        super(vgg19, self).__init__()
        from torchvision import models
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for i in range(2):
            self.slice1.add_module(str(i), vgg_pretrained_features[i])
        for i in range(2, 7):
            self.slice2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(7, 12):
            self.slice3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(12, 21):
            self.slice4.add_module(str(i), vgg_pretrained_features[i])
        for i in range(21, 30):
            self.slice5.add_module(str(i), vgg_pretrained_features[i])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def EOTF_PQ_cuda(ERGB):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    ERGB = torch.clamp(ERGB, min=1e-10, max=1)

    X1 = ERGB ** (1 / m2)
    X2 = X1 - c1
    X2[X2 < 0] = 0

    X3 = c2 - c3 * X1

    X4 = (X2 / X3) ** (1 / m1)
    return X4 * 10000

def EOTF_PQ_cuda_inverse(LRGB):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    RGB_l = LRGB / 10000
    RGB_l = torch.clamp(RGB_l, min=1e-10, max=1)

    X1 = c1 + c2 * RGB_l ** m1
    X2 = 1 + c3 * RGB_l ** m1
    X3 = (X1 / X2) ** m2
    return X3


def HDR_to_ICTCP(ERGB, dim=-1):
    LRGB = EOTF_PQ_cuda(ERGB)  # hw3
    LR, LG, LB = torch.split(LRGB, 1, dim=dim)  # hw1

    L = (1688 * LR + 2146 * LG + 262 * LB) / 4096
    M = (683 * LR + 2951 * LG + 462 * LB) / 4096
    S = (99 * LR + 309 * LG + 3688 * LB) / 4096
    LMS = torch.cat([L, M, S], dim=dim)  # hw3

    ELMS = EOTF_PQ_cuda_inverse(LMS)  # hw3

    EL, EM, ES = torch.split(ELMS, 1, dim=dim)  # hw1
    I = (2048 * EL + 2048 * EM + 0 * ES) / 4096
    T = (6610 * EL - 13613 * EM + 7003 * ES) / 4096
    P = (17933 * EL - 17390 * EM - 543 * ES) / 4096

    ITP = torch.cat([I, T, P], dim=dim)  # hw3
    return ITP

class ITP_L1Loss(nn.Module):  # for ICATNet
    def __init__(self):
        super(ITP_L1Loss, self).__init__()

    def forward(self, x, y):
        return F.l1_loss(HDR_to_ICTCP(x, dim=1), HDR_to_ICTCP(y, dim=1))
