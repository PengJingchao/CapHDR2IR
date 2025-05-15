import os.path as osp
import logging
import time
import argparse
import cv2
import options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

import numpy as np
import matlab.engine
import gc
import skimage.metrics as metric
import lpips
import torch

if __name__ == '__main__':
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YMAL file.',
                        default=r'options/test/RGB2IR_others/HDR2IR_UNet_ours.yml')
    parser.add_argument('-saveimg', type=bool, help='Whether to save output images.',
                        default=False)
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    lpips_model = lpips.LPIPS(net='vgg').cuda()

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        psnr = 0.0
        ssim = 0.0
        mse = 0.0
        lpips_dis = 0

        for data in test_loader:
            model.feed_data(data)
            img_path = data['Input_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            model.test()
            visuals = model.get_current_visuals()
            with torch.no_grad():
                lpips_dis += lpips_model(visuals['Output'].cuda(), visuals['GT'].cuda()).item()

            output_hdrimg = util.tensor2numpy(visuals['Output'])
            output_hdrimg = np.clip(output_hdrimg * 255, 0, 255).astype('uint8')
            if parser.parse_args().saveimg:
                image_path = util.generate_paths(dataset_dir, img_name)
                cv2.imwrite(image_path, output_hdrimg)

            logger.info('{:20s}'.format(img_name))

            gt_hdrimg = util.tensor2numpy(visuals['GT'])
            gt_hdrimg = np.clip(gt_hdrimg * 255, 0, 255).astype('uint8')

            # Metric results
            psnr += metric.peak_signal_noise_ratio(output_hdrimg, gt_hdrimg)
            ssim += metric.structural_similarity(output_hdrimg, gt_hdrimg, channel_axis=-1)
            mse += metric.normalized_root_mse(output_hdrimg, gt_hdrimg)

        # log
        logger.info('########### Metric results: ############')
        logger.info('PSNR: {:.4e}, SSIM: {:.4e}, NRMSE: {:.4e}'.format(psnr / test_loader.__len__(),
                                                                     ssim / test_loader.__len__(),
                                                                     mse / test_loader.__len__()))

        logger.info('Lpips: {:.4e}'.format(lpips_dis / test_loader.__len__()))

        logger.info('######### End Metric Results. ##########')
