import os
import math
import argparse
import random
import logging
import time

import torch
from tensorboardX import SummaryWriter

import options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

import numpy as np


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.',
                        default=r'options/train/HDR2IR_UNet_ours.yml')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if resume_state is None:
        util.mkdir_and_rename(
            opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        tb_logger = SummaryWriter(log_dir=os.path.join(opt['path']['root'], 'experiments', opt['name'], 'tb_logger'))

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_epochs = int(opt['train']['nepoch'])

            train_loader = create_dataloader(train_set, dataset_opt, opt)

            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_epochs * train_size))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    first_time = True
    for epoch in range(start_epoch, total_epochs + 1):
        # train
        for i, train_data in enumerate(train_loader):
            if first_time:
                start_time = time.time()
                first_time = False
            current_step += 1

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                end_time = time.time()
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, time:{:.3f}> '.format(
                    epoch, current_step, model.get_current_learning_rate(), end_time - start_time)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
                start_time = time.time()

        # val
        avg_psnr = 0.0
        avg_normalized_psnr = 0.0
        avg_tonemapped_psnr = 0.0
        idx = 0
        for val_data in val_loader:
            idx += 1

            model.feed_data(val_data)
            model.test()

            visuals = model.get_current_visuals()

            sr_img = util.tensor2numpy(visuals['Output'])  # float32
            gt_img = util.tensor2numpy(visuals['GT'])  # float32

            # calculate PSNR
            avg_psnr += util.calculate_psnr(sr_img, gt_img)
            avg_normalized_psnr += util.calculate_normalized_psnr(sr_img, gt_img, np.max(gt_img))
            avg_tonemapped_psnr += util.calculate_tonemapped_psnr(sr_img, gt_img, percentile=99, gamma=2.24)

        avg_psnr = avg_psnr / idx
        avg_normalized_psnr = avg_normalized_psnr / idx
        avg_tonemapped_psnr = avg_tonemapped_psnr / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}, norm_PSNR: {:.4e}, mu_PSNR: {:.4e}'.format(avg_psnr,
                                                                                             avg_normalized_psnr,
                                                                                             avg_tonemapped_psnr))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} norm_PSNR: {:.4e} mu_PSNR: {:.4e}'.format(
            epoch, current_step, avg_psnr, avg_normalized_psnr, avg_tonemapped_psnr))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            tb_logger.add_scalar('psnr', avg_psnr, current_step)
            tb_logger.add_scalar('norm_PSNR', avg_normalized_psnr, current_step)
            tb_logger.add_scalar('mu_PSNR', avg_tonemapped_psnr, current_step)

        #### save models and training states
        logger.info('Saving models and training states.')
        model.save(current_step)
        model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
