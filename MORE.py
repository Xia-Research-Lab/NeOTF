import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import zoom
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
from utils import read_speckles_from_folder, crop_center, Config, load_config
import argparse

# try to import torch
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

# implementation of MORE algorithm using PyTorch (optional CUDA)
def MORE(speckle_list, num_epoch=50, sup_size=(50, 50), use_gpu=False):
    """MORE using PyTorch. If use_gpu=True and CUDA is available, computations run on GPU."""
    # decide device
    device = torch.device('cuda') if (use_gpu and _TORCH_AVAILABLE and torch.cuda.is_available()) else torch.device('cpu') if _TORCH_AVAILABLE else None

    if device is None:
        # fallback to original numpy implementation when torch not available
        speckles_FFTM = list()
        speckles_FFTP = list()

        for speckle in speckle_list:
            FFT = np.fft.fft2(speckle)
            FFT_shift = np.fft.fftshift(FFT)
            mag = np.abs(FFT_shift)
            mean_mag = np.mean(mag)
            normalized_fft = FFT_shift / mean_mag
            speckles_FFTM.append(np.abs(normalized_fft))
            speckles_FFTP.append(normalized_fft)

        height, width = speckle.shape[:2]
        s = np.random.rand(height, width)
        OTF = np.fft.fft2(s)

        sup_mat = np.zeros((height, width))
        rx, ry = sup_size
        sup_mat[(height // 2) - (rx // 2):(height // 2) + (rx // 2), (width // 2) - (ry // 2):(width // 2) + (ry // 2)] = 1

        for i in range(num_epoch):
            for j in range(len(speckle_list)):
                k_space = speckles_FFTM[j] * np.exp(1j * (np.angle(speckles_FFTP[j]) - np.angle(OTF)))
                r_space = np.real(np.fft.ifft2(np.fft.ifftshift(k_space)))
                r_sp = r_space * sup_mat
                r_sp[r_sp < 0] = 0
                k_space = np.fft.fftshift(np.fft.fft2(r_sp))
                OTF = speckles_FFTP[j] / (k_space + 1e-5)

        objs_list = list()
        for i in range(len(speckle_list)):
            obj = np.real(np.fft.ifft2(
                np.fft.ifftshift(speckles_FFTM[i] * np.exp(1j * (np.angle(speckles_FFTP[i]) - np.angle(OTF))))))
            obj[obj < 0] = 0
            obj = crop_center(obj,100)
            objs_list.append(obj)

        return objs_list, OTF

    # use torch backend
    xp = torch

    speckles_FFTM = []
    speckles_FFTP = []

    for speckle in speckle_list:
        s_arr = torch.from_numpy(speckle).to(dtype=torch.float32, device=device)
        FFT = torch.fft.fft2(s_arr)

        FFT_shift = torch.fft.fftshift(FFT)
        mag = torch.abs(FFT_shift)

        mean_mag = torch.mean(mag)
        normalized_fft = FFT_shift / (mean_mag)

        speckles_FFTM.append(torch.abs(normalized_fft))
        speckles_FFTP.append(normalized_fft)

    height, width = speckle.shape[:2]

    s = torch.rand((height, width), device=device)
    OTF = torch.fft.fft2(s)

    sup_mat = torch.zeros((height, width), device=device)
    rx, ry = sup_size
    sup_mat[(height // 2) - (rx // 2):(height // 2) + (rx // 2), (width // 2) - (ry // 2):(width // 2) + (ry // 2)] = 1

    for i in range(num_epoch):
        for j in range(len(speckle_list)):
            phase_diff = torch.angle(speckles_FFTP[j]) - torch.angle(OTF)
            k_space = speckles_FFTM[j] * torch.exp(1j * phase_diff)
            r_space = torch.real(torch.fft.ifft2(torch.fft.ifftshift(k_space)))

            r_sp = r_space * sup_mat
            r_sp[r_sp < 0] = 0

            k_space = torch.fft.fftshift(torch.fft.fft2(r_sp))
            OTF = speckles_FFTP[j] / (k_space + 1e-5)

    objs_list = []
    for i in range(len(speckle_list)):
        phase_diff = torch.angle(speckles_FFTP[i]) - torch.angle(OTF)
        obj = torch.real(torch.fft.ifft2(torch.fft.ifftshift(speckles_FFTM[i] * torch.exp(1j * phase_diff))))
        obj[obj < 0] = 0

        # move to CPU numpy for crop_center and saving
        obj_cpu = obj.detach().to('cpu').numpy()
        obj_cpu = crop_center(obj_cpu, 200)
        objs_list.append(obj_cpu)

    OTF_cpu = OTF.detach().to('cpu').numpy()

    return objs_list, OTF_cpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a model using a YAML config file.")
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--use_gpu', action='store_true', help='Use torch CUDA if available')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    speckles_array_list,_ = read_speckles_from_folder(config.data.path,config.data)

    num_frames=5

    # decide on GPU usage
    use_gpu_flag = args.use_gpu and _TORCH_AVAILABLE and torch.cuda.is_available()
    if args.use_gpu and not (_TORCH_AVAILABLE and torch.cuda.is_available()):
        print("Warning: --use_gpu specified but CUDA/Torch not available. Falling back to CPU.")

    import time
    start_time = time.time()
    objs, _ = MORE(speckles_array_list, num_epoch=5, sup_size=(config.data.supp_size_h, config.data.supp_size_w), use_gpu=use_gpu_flag)
    # synchronize if using GPU
    if use_gpu_flag:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    elapsed = time.time() - start_time
    print(f"MORE run time (use_gpu={use_gpu_flag}): {elapsed:.6f} s")

    import matplotlib as mpl
    mpl.rcParams['font.size'] = 30

    experiment_path = os.path.join('./outputs', f"MORE")
    os.makedirs(experiment_path, exist_ok=True)
    with open(os.path.join(experiment_path, 'MORE_time.txt'), 'w') as f:
        f.write(f"MORE run time (s) (use_gpu={use_gpu_flag}): {elapsed:.6f}\n")

    # Save OTF phase map similar to NeOTF
    otf_pha_np = np.angle(_)
    plt.figure(figsize=(10, 8))
    plt.imshow(otf_pha_np, cmap='twilight')
    plt.colorbar()
    plt.savefig(os.path.join(experiment_path, 'otf_pha.png'), dpi=100, bbox_inches='tight')
    plt.close()

    for i in range(len(objs)):
        recons = objs[i]
        recons = crop_center(recons,100)
        recons = (recons - np.min(recons)) / (np.max(recons) - np.min(recons)) * 255.0
        recons = Image.fromarray(recons.astype(np.uint8))
        recons.save(os.path.join(experiment_path, '%d.png' % i))
