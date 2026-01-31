import os
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import yaml
from datetime import datetime
import random

from scipy.ndimage import rotate

def set_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

class Config:
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def __repr__(self):

        return str(self.__dict__)

def load_config(path='config.yml'):

    print(f"Loading configuration from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(config_dict)
    return config
    
def get_mgrid(sidelen, dim=2):

    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_mgrid_hermitian(sidelen, dim=2):

    if dim != 2:
        raise ValueError("This function is designed for dim=2")
    
    half_sidelen = sidelen // 2 + 1
    tensors = (torch.linspace(0, 1, steps=half_sidelen),
               torch.linspace(-1, 1, steps=sidelen))
    
    hermitian_grid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    hermitian_grid = hermitian_grid.reshape(-1, dim)
    
    y_indices, x_indices = torch.meshgrid(torch.arange(sidelen), torch.arange(sidelen), indexing="ij")

    mask_2d = y_indices <= sidelen // 2
    
    mask = mask_2d.reshape(-1)

    return hermitian_grid, mask

def get_circular_mgrid(sidelen, radius):

    full_grid = get_mgrid(sidelen, dim=2)

    dist_sq = torch.sum(full_grid**2, dim=1)

    mask = dist_sq <= radius**2

    circular_grid = full_grid[mask]

    return circular_grid, mask

def get_circular_mgrid_hermitian(sidelen, radius, dim=2):
    """
    Generates a grid of coordinates within a circular region of the 
    non-redundant half of a Fourier plane (exploiting Hermitian symmetry).
    This is the corrected version.

    Args:
        sidelen (int): The side length of the full square grid.
        radius (float): The radius of the circular sampling region in normalized 
                        coordinates (from 0 to sqrt(2)).
        dim (int): The dimension of the grid (should be 2).

    Returns:
        torch.Tensor: A tensor of shape [num_coords, 2] containing the unique coordinates.
        torch.Tensor: A 1D boolean tensor of shape [sidelen*sidelen] that can be used 
                      to index into a flattened full grid.
    """
    full_grid = get_mgrid(sidelen, dim=2)

    dist_sq = torch.sum(full_grid**2, dim=1)
    circular_mask = dist_sq <= radius**2

    y_indices, x_indices = torch.meshgrid(torch.arange(sidelen), torch.arange(sidelen), indexing="ij")
    hermitian_mask_2d = y_indices <= sidelen // 2
    hermitian_mask = hermitian_mask_2d.reshape(-1)

    final_mask = circular_mask & hermitian_mask

    final_grid = full_grid[final_mask]

    return final_grid, final_mask


def read_speckles_from_folder(folder_path, data_config):
    image_array_list = []
    image_torch_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(data_config.format):
            file_path = os.path.join(folder_path, filename)

            with Image.open(file_path) as img:
                if img.mode != 'L' and data_config.format == 'bmp':
                    img = img.convert('L')
                image_array = np.array(img)
                image_array = image_array.astype(np.float32)

                # If loading from preprocessed TIFF (uint16 format), convert back to [0,1]
                if data_config.format == 'tif' and image_array.max() > 1:
                    # Assuming uint16 range [0, 65535]
                    image_array = image_array / 65535.0
                #image_array = image_array-np.mean(image_array)
                image_torch = torch.from_numpy(image_array)
                image_torch = image_torch.unsqueeze(0).unsqueeze(0)

                image_array_list.append(image_array)
                image_torch_list.append(image_torch)

    return image_array_list, image_torch_list



def crop_center(image, crop_size_h,crop_size_w=None):

    h = image.shape[0]
    w = image.shape[1]

    if crop_size_w is None:
        crop_size_w = crop_size_h
    start_h = (h-crop_size_h)//2
    start_w = (w-crop_size_w)//2

    return image[start_h:start_h+crop_size_h,start_w:start_w+crop_size_w]

from scipy.fft import fft2, ifft2, fftshift, ifftshift




def total_variation_loss(img):

    h_tv = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    w_tv = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    
    return h_tv + w_tv