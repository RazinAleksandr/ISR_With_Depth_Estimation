import os.path
import argparse
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

from swinir_pipeline.src.utils import utils_image as util
from swinir_pipeline.src.utils import utils_option_inference as option_inference
from swinir_pipeline.src.utils import utils_option as option
from swinir_pipeline.src.utils.utils_dist import get_dist_info, init_dist

from swinir_pipeline.src.data.select_dataset import define_Dataset
from swinir_pipeline.src.models.select_model import define_Model


def main(json_path='configurations/swinir_sr_classical.json'):

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

    opt = option_inference.parse(parser.parse_args().opt)
    opt['dist'] = parser.parse_args().dist
    
    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    

    border = opt['scale']
    opt = option.dict_to_nonedict(opt)
    

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''
    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'inference':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            continue

    # ----------------------------------------
    # 1) create_dataset for Depth Estimation
    # 2) creat_dataloader for test
    # ----------------------------------------
    for phase, dataset_opt in opt['Depth_datasets'].items():
        if phase == 'inference':
            test_set = define_Dataset(dataset_opt, depth=True)
            depth_test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            continue
    

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    model = define_Model(opt)
    model.init_inference()
    

    '''
    # ----------------------------------------
    # Step--4 (run inference)
    # ----------------------------------------
    '''
    print('Start inference ...')
    idx, avg_psnr = 0, 0
    for test_data, depth_data in zip(test_loader, depth_test_loader):
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'], img_name)
        util.mkdir(img_dir)

        model.feed_data(test_data, depth_data)

        model.test()
                    
        visuals = model.current_visuals()
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])

        # -----------------------
        # save estimated image E
        # -----------------------
        save_img_path = os.path.join(img_dir, '{:s}_.png'.format(img_name))
        util.imsave(E_img, save_img_path)

        # -----------------------
        # calculate PSNR
        # -----------------------
        current_psnr = util.calculate_psnr(E_img, H_img, border=border)

        print('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

        avg_psnr += current_psnr

    avg_psnr = avg_psnr / idx

    # testing log
    print('Average PSNR : {:<.2f}dB\n'.format(avg_psnr))
    print('Inference finished!')
    

if __name__ == '__main__':
    main()
