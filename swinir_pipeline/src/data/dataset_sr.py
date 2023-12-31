import random
import torch.utils.data as data
from torchvision.transforms import Resize

import src.utils.utils_image as util


class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt, depth=False):
        super(DatasetSR, self).__init__()
        self.opt = opt

        if depth == False:
            self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        else:    
            self.n_channels = opt['n_channels_depth'] if opt['n_channels_depth'] else 1
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf
        
        
        self.resize_transform_H = Resize((self.patch_size, self.patch_size))
        self.resize_transform_L = Resize((self.L_size, self.L_size))
        
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        if depth == False:
            self.paths_H = util.get_image_paths(opt['dataroot_H'])
            self.paths_L = util.get_image_paths(opt['dataroot_L'])
        else:
            self.paths_H = util.get_image_paths(opt['depth_dataroot_H'])
            self.paths_L = util.get_image_paths(opt['depth_dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

        self.paths_H = sorted(self.paths_H)
        # self.paths_H = self.paths_H[:10000]
        
    def __getitem__(self, index):
        random.seed(index)

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        if self.opt['phase'] == 'test':
            img_H = self.resize_transform_H(img_H)
            img_L = self.resize_transform_L(img_L)
        
        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)


class CombinedDataset(data.Dataset):
    """
    Dataset class to combine Image and Condition datasets.
    
    :param image_dataset: An instance of the ImageDataset.
    :param cond_dataset: An instance of the ConditionDataset.
    """
    def __init__(self, image_dataset, cond_dataset):
        self.image_dataset = image_dataset
        self.cond_dataset = cond_dataset
        assert len(self.image_dataset) == len(self.cond_dataset), "Datasets must be of the same size"

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, idx: int):
        image_data = self.image_dataset[idx]
        depth_data = self.cond_dataset[idx]

        return image_data, depth_data