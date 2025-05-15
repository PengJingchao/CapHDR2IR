import torch
import logging

import models.modules.HDR2IR.Resnet101_Unet_feature_with_spatial_attention as HDR2IR_UNet_ours

import models.modules.discriminator as Discriminators

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'HDR2IR_Feature_Ours':
        netG = HDR2IR_UNet_ours.ResNetUNet(n_classes=opt_net['out_nc'])


    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


def define_D():
    netD = Discriminators.VGGStyleDiscriminator(input_size=256)
    return netD
