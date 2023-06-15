import os
from collections import OrderedDict
import json
import re
import glob


def parse(opt_path):

    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['opt_path'] = opt_path

    # ----------------------------------------
    # set default
    # ----------------------------------------
    if 'merge_bn' not in opt:
        opt['merge_bn'] = False
        opt['merge_bn_startpoint'] = -1

    if 'scale' not in opt:
        opt['scale'] = 1

    # ----------------------------------------
    # datasets
    # ----------------------------------------
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = opt['scale']  # broadcast
        dataset['n_channels'] = opt['n_channels']  # broadcast
        if 'dataroot_H' in dataset and dataset['dataroot_H'] is not None:
            dataset['dataroot_H'] = os.path.expanduser(dataset['dataroot_H'])
        if 'dataroot_L' in dataset and dataset['dataroot_L'] is not None:
            dataset['dataroot_L'] = os.path.expanduser(dataset['dataroot_L'])
    # ----------------------------------------
    # depth datasets
    # ----------------------------------------
    for phase, dataset in opt['Depth_datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = opt['scale']  # broadcast
        dataset['n_channels'] = opt['n_channels']  # broadcast
        if 'dataroot_H' in dataset and dataset['dataroot_H'] is not None:
            dataset['dataroot_H'] = os.path.expanduser(dataset['dataroot_H'])
        if 'dataroot_L' in dataset and dataset['dataroot_L'] is not None:
            dataset['dataroot_L'] = os.path.expanduser(dataset['dataroot_L'])

    # ----------------------------------------
    # path
    # ----------------------------------------
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)

    path_task = os.path.join(opt['path']['root'], opt['task'])
    opt['path']['task'] = path_task
    opt['path']['log'] = path_task

    
    
    opt['path']['images'] = os.path.join(path_task, 'inferenced_images')

    # ----------------------------------------
    # network
    # ----------------------------------------
    opt['netG']['scale'] = opt['scale'] if 'scale' in opt else 1

    # ----------------------------------------
    # GPU devices
    # ----------------------------------------
    if opt['gpu_ids']:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # ----------------------------------------
    # default setting for distributeddataparallel
    # ----------------------------------------
    if 'find_unused_parameters' not in opt:
        opt['find_unused_parameters'] = True
    if 'use_static_graph' not in opt:
        opt['use_static_graph'] = False
    if 'dist' not in opt:
        opt['dist'] = False
    if opt['gpu_ids']:
        opt['num_gpu'] = len(opt['gpu_ids'])
        print('number of GPUs is: ' + str(opt['num_gpu']))

    # ----------------------------------------
    # default setting for discriminator
    # ----------------------------------------
    if 'netD' in opt:
        if 'net_type' not in opt['netD']:
            opt['netD']['net_type'] = 'discriminator_patchgan'  # discriminator_unet
        if 'in_nc' not in opt['netD']:
            opt['netD']['in_nc'] = 3
        if 'base_nc' not in opt['netD']:
            opt['netD']['base_nc'] = 64
        if 'n_layers' not in opt['netD']:
            opt['netD']['n_layers'] = 3
        if 'norm_type' not in opt['netD']:
            opt['netD']['norm_type'] = 'spectral'

    return opt

def find_last_checkpoint(save_dir, net_type='G', pretrained_path=None):
    """
    Args: 
        save_dir: model folder
        net_type: 'G' or 'D' or 'optimizerG' or 'optimizerD'
        pretrained_path: pretrained model path. If save_dir does not have any model, load from pretrained_path

    Return:
        init_iter: iteration number
        init_path: model path
    """
    file_list = glob.glob(os.path.join(save_dir, '*_{}.pth'.format(net_type)))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(net_type), file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(save_dir, '{}_{}.pth'.format(init_iter, net_type))
    else:
        init_iter = 0
        init_path = pretrained_path
    return init_iter, init_path