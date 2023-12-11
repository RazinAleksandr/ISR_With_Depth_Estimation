import os.path
import argparse
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

from src.utils import utils_image as util
from src.utils import utils_option_inference as option_inference
from src.utils import utils_option as option
from src.utils.utils_dist import get_dist_info, init_dist

from src.data.select_dataset import define_Dataset
from src.models.select_model import define_Model
from src.data.dataset_sr import CombinedDataset


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
            image_set = define_Dataset(dataset_opt)
            depth_set = define_Dataset(dataset_opt, depth=True)
            test_set = CombinedDataset(image_set, depth_set)
            test_loader = DataLoader(test_set, batch_size=1,
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
    for i, (test_data, depth_data) in enumerate(test_loader):
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'], img_name)
        util.mkdir(img_dir)

        model.feed_data(test_data, depth_data)
        
        model.test()
                    
        visuals = model.current_visuals()
        L_img = util.tensor2uint(test_data["L"])
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])

        # -----------------------
        # save estimated image E
        # -----------------------
        save_img_path_L = os.path.join(img_dir, '{:s}_degradated.png'.format(img_name))
        save_img_path_E = os.path.join(img_dir, '{:s}_generated.png'.format(img_name))
        save_img_path_H = os.path.join(img_dir, '{:s}_true.png'.format(img_name))
        util.imsave(L_img, save_img_path_L)
        util.imsave(E_img, save_img_path_E)
        util.imsave(H_img, save_img_path_H)

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
