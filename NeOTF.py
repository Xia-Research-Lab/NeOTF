import os
import shutil
import random
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

from SIREN import Siren
from utils import (
    read_speckles_from_folder,
    crop_center,
    get_mgrid_hermitian,
    get_circular_mgrid_hermitian,
    total_variation_loss,
    set_seed
)

class NeOTF_Trainer:
    def __init__(self, config, config_path):
        self.config = config
        self.config_path = config_path
        self.device = torch.device(config.training.device)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.losses = []
        
        self.setup_experiment()

    def setup_experiment(self):
        if self.config.training.random_seed:
            seed = random.randint(0, 10000)
            print(f"Using random seed: {seed}")
            set_seed(seed)
        else:
            print(f"Using determined seed: {self.config.training.seed}")
            set_seed(self.config.training.seed)

        self.experiment_path = os.path.join(self.config.output_dir, f"{self.config.experiment_name}")
        os.makedirs(self.experiment_path, exist_ok=True)
        
        try:
            shutil.copy(self.config_path, os.path.join(self.experiment_path, 'config.yml'))
            print(f"Successfully copied config file to: {self.experiment_path}")
        except Exception as e:
            print(f"Error copying config file: {e}")

    def _prepare_data(self):

        print("Preparing data...")
        _, speckle_torch_list = read_speckles_from_folder(self.config.data.path, self.config.data)
        self.size = speckle_torch_list[0].shape[-1]

        num_frames = self.config.training.num_frames
        speckle_torch_list = speckle_torch_list[:num_frames]
        
        speckle_batch = torch.cat(speckle_torch_list, dim=0).to(self.device)
        
        # FFT transform
        speckle_fft = torch.fft.fft2(speckle_batch)
        
        self.speckle_ft = torch.abs(torch.fft.fftshift(speckle_fft, dim=(-2, -1)))
        self.speckle_pha = torch.angle(torch.fft.fftshift(speckle_fft, dim=(-2, -1)))

        mean_ft = torch.mean(self.speckle_ft, dim=(-2, -1), keepdim=True)
        self.speckle_ft = self.speckle_ft / mean_ft

        if self.config.training.center_sample:
            self.coords, self.mask = get_circular_mgrid_hermitian(self.size, self.config.training.center_sample_radius)
        else:
            self.coords, self.mask = get_mgrid_hermitian(self.size, 2)

        self.coords = self.coords.to(self.device)

        self.support_tensor = self._create_support_tensor()
        self.support_tensor = self.support_tensor.to(self.device)
        print("Data preparation complete.")

    def _create_support_tensor(self):
        tensor = torch.zeros(self.size, self.size)
        supp_size_w = self.config.data.supp_size_w
        supp_size_h = self.config.data.supp_size_h
        start_index_w = (self.size - supp_size_w) // 2
        end_index_w = start_index_w + supp_size_w
        start_index_h = (self.size - supp_size_h) // 2
        end_index_h = start_index_h + supp_size_h
        tensor[start_index_h:end_index_h, start_index_w:end_index_w] = 1
        return tensor.unsqueeze(0).unsqueeze(0)


    def _build_model(self):

        print("Building model...")
        if self.config.model.type == 'SIREN':

            if self.config.model.outermost_linear:

                assert self.config.model.out_features == 1, \
                    "For 'outermost_linear=True' (using atan2), config 'out_features' must be 1."
                
                print("Building SIREN model with linear outer layer for atan2 phase output.")

            self.model = Siren(
                in_features=self.config.model.in_features,
                out_features=self.config.model.out_features,
                hidden_features=self.config.model.hidden_features,
                hidden_layers=self.config.model.hidden_layers,
                outermost_linear=self.config.model.outermost_linear,
                first_omega_0=self.config.model.first_omega_0,
                hidden_omega_0=self.config.model.hidden_omega_0,
                num_frequencies=self.config.model.num_frequencies
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.type}")

        if self.config.training.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.training.lr)
        elif self.config.training.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.training.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.training.optimizer}")

        if self.config.training.scheduler == 'CosineAnnealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.training.epochs, eta_min=1e-6)
            print("Using CosineAnnealingLR scheduler.")
        elif self.config.training.scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
            print("Using StepLR scheduler.")
        else:
            self.scheduler = None

        if self.config.training.loss == 'L1':
            self.criterion = torch.nn.L1Loss()
        elif self.config.training.loss == 'MSE':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.config.training.loss}")
        
        
            
        print("Model build complete.")
        
    def _reconstruct_full_phase_map(self, out):

        phase_map = torch.zeros(self.size * self.size, device=self.device)
        phase_map[self.mask] = out.view(-1)
        phase_map = phase_map.view(self.size, self.size)
        bottom_half_rows = -torch.flip(phase_map[1:self.size//2, :], dims=[0, 1])
        phase_map[self.size // 2 + 1 :, :] = bottom_half_rows
        return phase_map

    def train(self):

        self._prepare_data()
        self._build_model()
        
        num_epochs = self.config.training.epochs

        print(f"Shape of coords tensor fed to the model: {self.coords.shape}")
        
        epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs", unit="epoch")

        for epoch in epoch_pbar:
            self.model.train()
            self.optimizer.zero_grad()
            
            out, _ = self.model(self.coords)
            
            outputs_phase = self._reconstruct_full_phase_map(out)


            obj_pha = self.speckle_pha - outputs_phase.unsqueeze(0).unsqueeze(0)
            obj_ft_complex = self.speckle_ft * torch.exp(1j * obj_pha)
            
            obj = torch.real(torch.fft.ifft2(torch.fft.ifftshift(obj_ft_complex, dim=(-2, -1))))

            obj = obj * self.support_tensor
            obj[obj<0]=0

            outputs_ft_abs = torch.fft.fftshift(torch.abs(torch.fft.fft2(obj)), dim=(-2, -1))
            
            base_loss = self.criterion(outputs_ft_abs, self.speckle_ft)
            loss = base_loss
            
            weighted_tv_loss = torch.tensor(0.0, device=self.device)
            if self.config.training.tv_regularization:
                tv_val = total_variation_loss(obj)
                weighted_tv_loss = (self.config.training.tv_weight * tv_val) / obj.shape[0]
                loss = loss + weighted_tv_loss
            

            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.losses.append(loss.item())

            epoch_pbar.set_postfix({
                'Total': f"{loss.item():.6f}",
                'L1': f"{base_loss.item():.6f}",
                'TV': f"{weighted_tv_loss.item():.6f}"
            })

        print("\nTraining finished!")
        self.save_results()

    def save_results(self):

        print("Saving results...")
        np.save(os.path.join(self.experiment_path, 'loss.npy'), self.losses)

        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(self.coords)
            outputs_phase = self._reconstruct_full_phase_map(out)

        otf_pha_np = outputs_phase.cpu().numpy()
        plt.imshow(otf_pha_np, cmap='twilight')
        plt.colorbar()
        plt.savefig(os.path.join(self.experiment_path, 'otf_pha.png'))
        plt.close()

        with torch.no_grad():
            obj_pha = self.speckle_pha - outputs_phase.unsqueeze(0).unsqueeze(0)
            obj_ft_complex = self.speckle_ft * torch.exp(1j * obj_pha)
            obj = torch.real(torch.fft.ifft2(torch.fft.ifftshift(obj_ft_complex, dim=(-2, -1))))
            obj = torch.relu(obj)
            objs_np = obj.squeeze(1).cpu().numpy() # (N, H, W)

        for i in range(objs_np.shape[0]):
            recons_np = objs_np[i]
            recons_np = crop_center(recons_np, 100)
            
            if np.max(recons_np) - np.min(recons_np) > 1e-6:
                recons_np = (recons_np - np.min(recons_np)) / (np.max(recons_np) - np.min(recons_np)) * 255.0
            
            recons_img = Image.fromarray(recons_np.astype(np.uint8))
            recons_img.save(os.path.join(self.experiment_path, f'{i}.png'))
        print(f"Results saved to: {self.experiment_path}")


if __name__ == "__main__":
    import argparse
    from utils import Config, load_config
    
    parser = argparse.ArgumentParser(description="Train a NeOTF model using a YAML config file.")
    parser.add_argument('--config', type=str, default='config.yml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    
    trainer = NeOTF_Trainer(config, args.config)
    trainer.train()