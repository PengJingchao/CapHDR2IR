import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.modules as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.customize_loss import lab_L1Loss, tanh_L1Loss, tanh_L2Loss, GANLoss, VGGLoss, ITP_L1Loss, gradient_smooth_l1loss

logger = logging.getLogger('base')


def create_model(opt):
    m = GenerationModel(opt)
    logger.info('Model [{:s}] is created.'.format(GenerationModel(opt).__class__.__name__))
    return m


class GenerationModel(BaseModel):
    def __init__(self, opt):
        super(GenerationModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        self.netD = networks.define_D().to(self.device)

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.SmoothL1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'lab_L1':
                self.cri_pix = lab_L1Loss().to(self.device)
            elif loss_type == 'gradient_smooth_l1':
                self.cri_pix = gradient_smooth_l1loss().to(self.device)
            elif loss_type == 'tanh_l1':
                self.cri_pix = tanh_L1Loss().to(self.device)
            elif loss_type == 'tanh_l2':
                self.cri_pix = tanh_L2Loss().to(self.device)
            elif loss_type == 'ITP_l1':
                self.cri_pix = ITP_L1Loss().to(self.device)

            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
            self.l_sim_w = train_opt['similarity_weight']

            self.cri_gan = GANLoss().to(self.device)
            self.l_gan_w = train_opt['ganloss_weight']

            self.perceptual_loss = VGGLoss()
            self.l_per_w = train_opt['perceptual_loss_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            optim_params = []
            for k, v in self.netD.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_D = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        if torch.isnan(data['Input']).any().item():
            logger.info('Nan data found in {}'.format(data['Input_path']))
        if torch.isnan(data['GT']).any().item():
            logger.info('Nan data found in {}'.format(data['GT_path']))
        self.var_L = data['Input'].to(self.device)  # Input

        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)  # self.netG((self.var_L, self.var_cond))

        # optimize net_d
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        # gan loss (relativistic gan)

        # real
        fake_d_pred = self.netD(self.fake_H).detach()
        real_d_pred = self.netD(self.real_H)
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True)

        # fake
        fake_d_pred = self.netD(self.fake_H.detach())
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True)
        l_d_total = (l_d_real + l_d_fake) / 2
        l_d_total.backward(retain_graph=True)

        self.optimizer_D.step()

        # set log
        self.log_dict['l_d_real'] = l_d_real
        self.log_dict['l_d_fake'] = l_d_fake
        self.log_dict['l_d_total'] = l_d_total

        # optimize net_g
        for p in self.netD.parameters():
            p.requires_grad = False

        # self.optimizer_G.zero_grad()
        # self.fake_H = self.netG(self.var_L)  # self.netG((self.var_L, self.var_cond))

        l_g_total = torch.tensor(0., device=self.device)

        # pixel loss
        l_pix = self.cri_pix(self.fake_H, self.real_H)
        l_g_total += l_pix * self.l_pix_w
        # set log
        self.log_dict['l_pix'] = l_pix.item()

        l_similarity = (1 - self.similarity(self.fake_H, self.real_H)).mean()
        l_g_total += l_similarity * self.l_sim_w
        # set log
        self.log_dict['similarity'] = l_similarity.item()

        l_pre = self.perceptual_loss(self.fake_H, self.real_H)
        l_g_total += l_pre * self.l_per_w
        # set log
        self.log_dict['perceptual_loss'] = l_pre.item()

        # gan loss (relativistic gan)
        real_d_pred = self.netD(self.real_H).detach()
        fake_g_pred = self.netD(self.fake_H)
        l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
        l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
        l_g_gan = (l_g_real + l_g_fake) / 2

        l_g_total += l_g_gan * self.l_gan_w
        self.log_dict['l_g_gan'] = l_g_gan.item()

        l_g_total.backward()
        self.optimizer_G.step()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)  # self.netG((self.var_L, self.var_cond))
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['Input'] = self.var_L.detach()[0].float().cpu()
        out_dict['Output'] = self.fake_H.detach()[0].float().cpu()

        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
            if self.opt['is_train']:
                try:  # if there are .pths for discriminator
                    logger.info('Loading model for D [{:s}] ...'.format(load_path_G.replace('_G.pth', '_D.pth')))
                    self.load_network(load_path_G.replace('_G.pth', '_D.pth'), self.netD)
                except:
                    logger.info('Cannot load the discriminator!')

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        if hasattr(self, 'netD'):
            self.save_network(self.netD, 'D', iter_label)
