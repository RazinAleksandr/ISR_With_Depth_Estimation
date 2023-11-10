import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

import warnings
warnings.filterwarnings("ignore")

from src.utils import utils_logger
from src.utils import utils_image as util
from src.utils import utils_option as option
from src.utils.utils_dist import get_dist_info, init_dist

from src.data.select_dataset import define_Dataset
from src.models.select_model import define_Model

import wandb


def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    
    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)
    
    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            #raise NotImplementedError("Phase [%s] is not recognized." % phase)
            continue

    # ----------------------------------------
    # 1) create_dataset for Depth Estimation
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['Depth_datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt, depth=True)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler_depth = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                depth_train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler_depth)
            else:
                depth_train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt, depth=True)
            depth_test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            #raise NotImplementedError("Phase [%s] is not recognized." % phase)
            continue
    

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    model = define_Model(opt)
    model.init_train()

    #########################FREEZE#########################
    print('#'*100)
    total_params  = sum(p.numel() for p in model.netG.parameters() if p.requires_grad)  
    print(f'Total parameters number: {total_params}')
    print('Trainable parameters:')
    for i, (name, param) in enumerate(model.netG.named_parameters()):
        if i <= 487:
            param.requires_grad = False
        else:
            print(name)
            param.requires_grad = True
    total_params  = sum(p.numel() for p in model.netG.parameters() if p.requires_grad)  
    print(f'Trainable parameters number: {total_params}')
    print('#'*100)
    ########################################################
    

    if opt['rank'] == 0:
        pass
        #logger.info(model.info_network())
        #logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    # -------------------------------
    # define wanb
    # -------------------------------
    wandb.init(
      # Set the project where this run will be logged
      project="SwinIR with depth estimation", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name="experiment_8", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": 1e-4,
      "architecture": "SwinIR",
      "dataset": "HRWSI",
      "epochs": 100,
      "batch_size": 48,
      })
    im_logs, depth_logs = [], []
    
    for epoch in range(200):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)
            train_sampler_depth.set_epoch(epoch)
        im_logs = []
        for i, (train_data, depth_data) in enumerate(zip(train_loader, depth_train_loader)):
            #print(train_data['L'].shape, train_data['H'].shape)
            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data, depth_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)
            

            # -------------------------------
            # 4) training information
            # -------------------------------
            wandb.log({
                  "Train G_loss": model.current_log()['G_loss'], 
                  },
                  step=current_step
                  )
                  
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data, depth_data in zip(test_loader, depth_test_loader):
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    #image_name_ext_depth = os.path.basename(depth_data['L_path'][0])
                    #print(image_name_ext, image_name_ext_depth)
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data, depth_data)

                    model.test()
                    
                    # add the same for the train for depth images!!!
                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr

                    # ---------------------------
                    # wanb log
                    # ---------------------------
                    wandb.log({"Test avg_psnr": avg_psnr}) 
                    im_logs.append(wandb.Image(
                      E_img,
                      caption='Predicted {:s}_{:d}'.format(img_name, current_step)
                    ))
                    im_logs.append(wandb.Image(
                      H_img,
                      caption='Ground truth {:s}'.format(img_name)
                    ))
                    

                avg_psnr = avg_psnr / idx
                # ---------------------------
                # wanb log
                # ---------------------------
                wandb.log({
                  "Test avg_psnr": avg_psnr, 
                  'Image prediction': im_logs,
                  },
                  step=current_step
                  )

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))
    
    # Mark the run as finished
    wandb.finish()

if __name__ == '__main__':
    main()