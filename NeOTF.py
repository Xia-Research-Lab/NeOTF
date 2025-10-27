import os
import matplotlib.pyplot as plt

import numpy as np
import torch

from SIREN import Siren
from utils import read_speckles_from_folder,crop_center,get_mgrid,pad_to_size,total_variation_loss, set_seed,get_circular_mgrid
from utils import Config, load_config
import argparse
from PIL import Image
import shutil
import random
from tqdm import tqdm
def test(config,config_path):
    if config.training.random_seed:
        seed = random.randint(0, 10000)
        print(f"Using random seed: {seed}")
        set_seed(seed)
    else:
        print(f"Using determined seed: {config.training.seed}")
        set_seed(config.training.seed)

    experiment_path = os.path.join(config.output_dir, f"{config.experiment_name}_{config.timestamp}")
    os.makedirs(experiment_path, exist_ok=True)

    try:
        shutil.copy(config_path, os.path.join(experiment_path, 'config.yml'))
        print(f"Successfully copied config file to: {experiment_path}")
    except Exception as e:
        print(f"Error copying config file: {e}")

    size = config.data.desired_size

    tensor = torch.zeros(size, size)

    supp_size_w =config.data.supp_size_w
    supp_size_h =config.data.supp_size_h

    start_index_w = (size - supp_size_w) // 2
    end_index_w = start_index_w + supp_size_w

    start_index_h = (size - supp_size_h) // 2
    end_index_h = start_index_h + supp_size_h

    tensor[start_index_h:end_index_h, start_index_w:end_index_w] = 1
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    _, speckle_torch_list, = read_speckles_from_folder(config.data.path, config.data)

    device = config.training.device

    speckle_torch_list = [tensor.to(device) for tensor in speckle_torch_list]

    speckle_ft = [torch.abs(torch.fft.fftshift(torch.fft.fft2(tensor))) for tensor in speckle_torch_list]
    for i in range(len(speckle_ft)):
        speckle_ft[i][0,0,0,0]= speckle_ft[i][0,0,1,1]

    if config.training.center_sample:
        coords, mask = get_circular_mgrid(size,config.training.center_sample_radius)
    else:
        coords = get_mgrid(size, 2).to(device)


    coords = coords.to(device)

    speckle_pha = [torch.angle(torch.fft.fftshift(torch.fft.fft2(tensor))) for tensor in speckle_torch_list]
    tensor = tensor.to(device)
    num_epochs = config.training.epochs
    num_frames = config.training.num_frames

    if config.model.type == 'SIREN':
        model = Siren(in_features=config.model.in_features,
                      out_features=config.model.out_features,
                      hidden_features=config.model.hidden_features,
                      hidden_layers=config.model.hidden_layers,
                      outermost_linear=config.model.outermost_linear,
                      first_omega_0=config.model.first_omega_0,
                      hidden_omega_0=config.model.hidden_omega_0,
                      num_frequencies=config.model.num_frequencies).to(device)
    else:
        raise ValueError(f"Unsupported model type: {config.model.type}")
    

    if config.training.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    elif config.training.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training.lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.training.optimizer}")

    if config.training.loss == 'L1':
        criterion = torch.nn.L1Loss()
    elif config.training.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    else :
        raise ValueError(f"Unsupported loss type: {config.training.loss}")
    criterion = torch.nn.L1Loss()
    losses = []

    

    epoch_pbar = tqdm(range(num_epochs), desc=f"Training Epochs", unit="epoch")
    # The main loop now iterates over the tqdm object
    for epoch in epoch_pbar:
        optimizer.zero_grad()  # Clear gradients
        
        out, coords_out = model(coords)

        if config.training.center_sample:
            phase_map = torch.zeros(size*size).to(device)
            phase_map[mask]=out.view(-1)
            outputs = phase_map.view(size, size)
        else:
            outputs = out.view(size, size)

        loss = 0.0
        for i in range(num_frames):
            obj_pha = speckle_pha[i] - outputs
            obj = torch.real(torch.fft.ifft2(torch.fft.ifftshift((speckle_ft[i] * torch.exp(1j * obj_pha)))))
            obj = obj * tensor
            obj = torch.relu(obj)
            
            # Calculate reconstruction loss
            outputs_ft = torch.fft.fftshift(torch.abs(torch.fft.fft2(obj)))
            reconstruction_loss_i = criterion(outputs_ft.squeeze(), speckle_ft[i].squeeze())
            
            # Add reconstruction loss to the total
            loss += reconstruction_loss_i
            
            # Add TV regularization loss if enabled
            if config.training.tv_regularization:
                # Reshape obj for the TV loss function: (H, W) -> (1, 1, H, W)
                obj_4d = obj.unsqueeze(0).unsqueeze(0)
                tv_loss_i = total_variation_loss(obj_4d)
                loss += config.training.tv_weight * tv_loss_i
                
        loss = loss / num_frames
        
        loss.backward()  # Backpropagation
        optimizer.step()   # Update parameters

        losses.append(loss.item())

        epoch_pbar.set_postfix(loss=f"{loss.item():.6f}")

    # The loop is finished, the progress bar will automatically close.
    print("\nTraining finished!")

    import matplotlib as mpl
    mpl.rcParams['font.size'] = 30
    from pathlib import Path


    np.save(os.path.join(experiment_path, 'loss.npy'), losses)
    model.eval()
    out, coords_out = model(coords)
    if config.training.center_sample:
        phase_map = torch.zeros(size*size).to(device)
        phase_map[mask]=out.view(-1)
        outputs = phase_map.view(size, size)
    else:
        outputs = out.view(size, size)

    for i in range(len(speckle_torch_list)):
        obj_pha = speckle_pha[i] - outputs
        obj = torch.real(torch.fft.ifft2(torch.fft.ifftshift((speckle_ft[i] * torch.exp(1j * obj_pha)))))
        obj[obj<0]=0
        outputs_show = obj.squeeze()
        recons = outputs_show.cpu().detach().numpy()
        recons = crop_center(recons,100)
        recons = (recons - np.min(recons)) / (np.max(recons) - np.min(recons)) * 255.0
        recons = Image.fromarray(recons.astype(np.uint8))
        recons.save(os.path.join(experiment_path, '%d.png' % i))

    outputs = outputs.cpu().detach().numpy()

    plt.imshow(outputs,cmap='twilight')
    plt.colorbar()
    plt.savefig(os.path.join(experiment_path, 'otf_pha.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using a YAML config file.")
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    test(config, args.config)
