import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import zoom
from PIL import Image
import os
# real speckle process
from datetime import datetime
from utils import read_speckles_from_folder
import argparse
from utils import Config, load_config, crop_center

# try to import torch for optional GPU acceleration
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

# implementation of HIO+ER algorithm with optional PyTorch support
def HIO_ER(speckle, num_hio=800, num_er=200, sup_size=(60, 40), use_gpu=False, beta=0.7):
    """Hybrid Input-Output (HIO) followed by Error Reduction (ER) algorithm.
    If use_gpu=True and torch.cuda is available, computations will run on the GPU."""

    # choose backend/device
    if use_gpu and _TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.device('cuda')
    elif _TORCH_AVAILABLE:
        device = torch.device('cpu')
    else:
        device = None

    if device is None:
        # fallback to numpy implementation
        F = np.fft.fft2(speckle)
        F_conjugate = np.conj(F)
        power_spectrum = F * F_conjugate
        ac_speckle = np.real(np.fft.ifftshift(np.fft.ifft2(power_spectrum)))
        
        size = ac_speckle.shape[0]
        supp_size_w = sup_size[0]
        supp_size_h = sup_size[1]

        mat2 = np.zeros_like(ac_speckle)
        start_index_w = (size - supp_size_w) // 2
        end_index_w = start_index_w + supp_size_w
        start_index_h = (size - supp_size_h) // 2
        end_index_h = start_index_h + supp_size_h
        mat2[start_index_h:end_index_h, start_index_w:end_index_w] = 1

        ac_speckle = (ac_speckle - np.min(ac_speckle)) / (np.max(ac_speckle) - np.min(ac_speckle) + 1e-12)
        ac_speckle = ac_speckle - np.mean(ac_speckle)
        speckle_ft = np.sqrt(np.abs(np.fft.fft2(ac_speckle)))

        # Initial guess
        obj = np.random.rand(size, size)
        phase = np.angle(np.fft.fft2(obj))

        # HIO iterations
        for i in range(num_hio):
            obj_prime = np.real(np.fft.ifft2(speckle_ft * np.exp(1j * phase)))
            mask = (mat2 > 0) & (obj_prime >= 0)
            obj = np.where(mask, obj_prime, obj - beta * obj_prime)
            phase = np.angle(np.fft.fft2(obj))
        
        # ER iterations
        for i in range(num_er):
            obj_prime = np.real(np.fft.ifft2(speckle_ft * np.exp(1j * phase)))
            obj = obj_prime * mat2
            obj[obj < 0] = 0
            phase = np.angle(np.fft.fft2(obj))
            
        return obj

    # PyTorch implementation
    speckle_t = torch.from_numpy(speckle).to(device=device, dtype=torch.float32)

    F = torch.fft.fft2(speckle_t)
    F_conjugate = torch.conj(F)
    power_spectrum = F * F_conjugate

    ac_speckle = torch.real(torch.fft.ifftshift(torch.fft.ifft2(power_spectrum)))

    size = ac_speckle.shape[0]
    supp_size_w = sup_size[0]
    supp_size_h = sup_size[1]

    mat2 = torch.zeros_like(ac_speckle)
    start_index_w = (size - supp_size_w) // 2
    end_index_w = start_index_w + supp_size_w
    start_index_h = (size - supp_size_h) // 2
    end_index_h = start_index_h + supp_size_h
    mat2[start_index_h:end_index_h, start_index_w:end_index_w] = 1.0

    ac_speckle = (ac_speckle - torch.min(ac_speckle)) / (torch.max(ac_speckle) - torch.min(ac_speckle) + 1e-12)
    ac_speckle = ac_speckle - torch.mean(ac_speckle)
    speckle_ft = torch.sqrt(torch.abs(torch.fft.fft2(ac_speckle)))

    # Initial guess
    obj = torch.rand((size, size), device=device)
    phase = torch.angle(torch.fft.fft2(obj))

    # HIO iterations
    for i in range(num_hio):
        obj_prime = torch.real(torch.fft.ifft2(speckle_ft * torch.exp(1j * phase)))
        mask = (mat2 > 0) & (obj_prime >= 0)
        obj = torch.where(mask, obj_prime, obj - beta * obj_prime)
        phase = torch.angle(torch.fft.fft2(obj))

    # ER iterations
    for i in range(num_er):
        obj_prime = torch.real(torch.fft.ifft2(speckle_ft * torch.exp(1j * phase)))
        obj = obj_prime * mat2
        obj = torch.where(obj < 0, torch.zeros_like(obj), obj)
        phase = torch.angle(torch.fft.fft2(obj))

    # move to CPU numpy
    obj_cpu = obj.detach().to('cpu').numpy()
    return obj_cpu



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a model using a YAML config file.")
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--use_gpu', action='store_true', help='Use torch CUDA if available')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    speckles_array_list,_ = read_speckles_from_folder(config.data.path,config.data)

    num_hio = 2000
    num_er = 500

    import matplotlib as mpl
    mpl.rcParams['font.size'] = 30
    import time

    # decide on GPU usage
    use_gpu_flag = args.use_gpu and _TORCH_AVAILABLE and torch.cuda.is_available()
    if args.use_gpu and not (_TORCH_AVAILABLE and torch.cuda.is_available()):
        print("Warning: --use_gpu specified but CUDA/Torch not available. Falling back to CPU.")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_path = os.path.join('./outputs', f"HIOER")
    os.makedirs(experiment_path, exist_ok=True)

    # timing
    per_sample_times = []
    total_start = time.time()

    for i in range(len(speckles_array_list)):
        sample_start = time.time()
        recons = HIO_ER(speckles_array_list[i], num_hio=num_hio, num_er=num_er, sup_size=(config.data.supp_size_w, config.data.supp_size_h), use_gpu=use_gpu_flag)
        if use_gpu_flag and _TORCH_AVAILABLE:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        sample_elapsed = time.time() - sample_start
        per_sample_times.append(sample_elapsed)
        print(f"Sample {i} time (s): {sample_elapsed:.6f}")

        recons = crop_center(recons,100)
        recons = (recons - np.min(recons)) / (np.max(recons) - np.min(recons)) * 255.0
        recons = Image.fromarray(recons.astype(np.uint8))
        recons.save(os.path.join(experiment_path, '%d.png' % i))

    total_elapsed = time.time() - total_start
    print(f"Total HIO run time (use_gpu={use_gpu_flag}): {total_elapsed:.6f} s")

    # save timing info
    with open(os.path.join(experiment_path, 'HIO_time.txt'), 'w') as f:
        f.write(f"use_gpu: {use_gpu_flag}\n")
        f.write(f"total_time_s: {total_elapsed:.6f}\n")
        for idx, t in enumerate(per_sample_times):
            f.write(f"sample_{idx}_time_s: {t:.6f}\n")