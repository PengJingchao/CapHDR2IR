import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision
from torchvision.models import ResNet101_Weights
from torchvision.transforms import transforms


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

def conv3x3(in_planes, out_planes, stride=1):
    '''3x3 conv with padding'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class TwoStreamBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=None):
        super(TwoStreamBottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(in_planes * 2, planes * 2)
        self.bn1 = norm_layer(planes * 2)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes * 2, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x[:, :x.size(1) // 2, :, :]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity

        return out


class TwoStreamResnet101(nn.Module):
    def __init__(self, block=Bottleneck, layers=(3, 4, 23, 3), num_class=1, norm_layer=None):
        super(TwoStreamResnet101, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        '''Fusion'''
        self.att_module_1 = SpatialAttentionModule(64)
        self.att_module_2 = SpatialAttentionModule(256)
        self.att_module_3 = SpatialAttentionModule(512)
        self.att_module_4 = SpatialAttentionModule(1024)

    def _make_layer(self, block, planes, blocks, stride=1):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(TwoStreamBottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, caption_out_1, caption_out_2, caption_out_3, caption_out_4):
        input = x
        x = self.conv1(x)  # 1/2  128
        x = self.bn1(x)
        x = self.relu(x)
        f1 = self.maxpool(x)  # 1/4 64
        caption_out_1_att_m = self.att_module_1(caption_out_1, f1)
        caption_out_1 = caption_out_1 * caption_out_1_att_m

        f2 = self.layer1(torch.cat([f1, caption_out_1], dim=1))  # 1/4 64
        caption_out_2_att_m = self.att_module_2(caption_out_2, f2)
        caption_out_2 = caption_out_2 * caption_out_2_att_m

        f3 = self.layer2(torch.cat([f2, caption_out_2], dim=1))  # 1/8 32
        caption_out_3_att_m = self.att_module_3(caption_out_3, f3)
        caption_out_3 = caption_out_3 * caption_out_3_att_m

        f4 = self.layer3(torch.cat([f3, caption_out_3], dim=1))  # 1/16 16
        caption_out_4_att_m = self.att_module_4(caption_out_4, f4)
        caption_out_4 = caption_out_4 * caption_out_4_att_m
        
        f5 = self.layer4(torch.cat([f4, caption_out_4], dim=1))  # 1/32 8

        return [x, f2, f3, f4, f5, input]


class UNetConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, normal_layer=None):
        super(UNetConvBlock, self).__init__()
        if normal_layer is None:
            normal_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn1 = normal_layer(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = normal_layer(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans,
                 up_conv_in_channels=None, up_conv_out_channels=None, up_mode='upconv'):
        super(UNetUpBlock, self).__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_chans
        if up_conv_out_channels == None:
            up_conv_out_channels = out_chans

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_chans, out_chans)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

class SpatialAttentionModule(nn.Module):
    def __init__(self, dim):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map

class ResNetUNet(nn.Module):
    def __init__(
            self,
            n_classes=1,
            norm_layer=None,
            up_mode='upconv',
    ):
        super(ResNetUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        '''Encoder'''
        self.encoder = TwoStreamResnet101()
        # Load pretrained weight from ResNet-101
        # self.encoder.load_state_dict(
        #      torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).state_dict(), strict=False)
        state_dict = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT).state_dict()

        # ignore the layer which the size is different
        current_state_dict = self.encoder.state_dict()
        for name, param in state_dict.items():
            if name in current_state_dict and param.shape == current_state_dict[name].shape:
                current_state_dict[name].copy_(param)
            else:
                print(f"Ignoring {name} due to shape mismatch")

        self.encoder.load_state_dict(current_state_dict, strict=False)

        '''Decoder'''
        self.decoder1 = UNetUpBlock(2048, 1024)
        self.decoder2 = UNetUpBlock(1024, 512)
        self.decoder3 = UNetUpBlock(512, 256)
        self.decoder4 = UNetUpBlock(in_chans=128 + 64, out_chans=128,
                                    up_conv_in_channels=256, up_conv_out_channels=128)
        self.decoder5 = UNetUpBlock(in_chans=64 + 3, out_chans=64,
                                    up_conv_in_channels=128, up_conv_out_channels=64)

        self.last = nn.Sequential(nn.Conv2d(64, n_classes, kernel_size=1), nn.Sigmoid())

        '''Caption Branch'''
        # self.caption_branch = torchvision.models.resnet101()
        self.caption_branch = nn.ModuleList(
            list(torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT).children())[:-3])
        # self.caption_branch.load_state_dict(torch.load('model-1000000.pth', map_location=torch.device('cpu'))['state_dict'])  # map_location=torch.device('cpu')
        try:
            original_params = {}
            for name, param in self.caption_branch.named_parameters():
                original_params[name] = param.clone()

            target_weights = torch.load(r'models\modules\HDR2IR\model-1000000.pth')['state_dict']
            state_dict_modified = {}
            for key, value in target_weights.items():
                if key.startswith('features.'):
                    new_key = key[len('features.'):]
                    state_dict_modified[new_key] = value
                else:
                    state_dict_modified[key] = value

            # 加载修改后的状态字典
            self.caption_branch.load_state_dict(state_dict_modified, strict=False)

            # 比较加载后的模型参数与加载前是否有变化
            parameters_changed = False
            for name, param in self.caption_branch.named_parameters():
                if not torch.equal(param, original_params[name]):
                    parameters_changed = True
                    break

            if not parameters_changed:
                raise RuntimeError("Attention please, cannot load caption_branch branch weights, Please Check!")

            print("Well-trained caption_branch branch weights load successful, will continue....")

        except:
            print("Attention please, cannot load caption_branch branch weights, Please Check!")

        self.caption_branch.eval()
        for param in self.caption_branch.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            caption_out = self.caption_branch[0](batch_process_hdr_images(x))  # 1/2  128
            caption_out = self.caption_branch[1](caption_out)
            caption_out = self.caption_branch[2](caption_out)
            caption_out_1 = self.caption_branch[3](caption_out)  # 1/4 64

            caption_out_2 = self.caption_branch[4](caption_out_1)  # 1/4 64
            caption_out_3 = self.caption_branch[5](caption_out_2)  # 1/8 32
            caption_out_4 = self.caption_branch[6](caption_out_3)  # 1/16 16

        encoder_output = self.encoder(x, caption_out_1, caption_out_2, caption_out_3, caption_out_4)
        encode0 = encoder_output[0]  # [1, 64, 128, 128]
        encode1 = encoder_output[1]  # [1, 256, 64, 64]
        encode2 = encoder_output[2]  # [1, 512, 32, 32]
        encode3 = encoder_output[3]  # [1, 1024, 16, 16]
        encode4 = encoder_output[4]  # [1, 2048, 8, 8]
        input_img = encoder_output[5]  # [1, 3, 256, 256]

        decode1 = self.decoder1(encode4, encode3)  # [1, 1024, 16, 16]
        decode2 = self.decoder2(decode1, encode2)  # [1, 512, 32, 32]
        decode3 = self.decoder3(decode2, encode1)  # [1, 256, 64, 64]
        decode4 = self.decoder4(decode3, encode0)  # [1, 128, 128, 128]

        decode5 = self.decoder5(decode4, input_img)  # [1, 64, 256, 256]

        out = self.last(decode5)

        return out.expand_as(x)

