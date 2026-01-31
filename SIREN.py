import numpy as np
import torch
import torch.nn as nn
from thop import profile

class SIREN(nn.Module):
    def __init__(self, omega_0=30.0):
        super(SIREN, self).__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


def fourier_encode(coords, in_features, num_frequencies):
    # coords shape: (num_points, 2)
    # frequencies shape: (1, num_frequencies)
    frequencies = 2 ** torch.arange(num_frequencies, device=coords.device, dtype=coords.dtype).unsqueeze(0)
    
    # x_args, y_args shape: (num_points, num_frequencies)
    x_args = coords[..., 0:1] * frequencies
    y_args = coords[..., 1:2] * frequencies
    
    encoded = torch.cat([
        torch.sin(x_args),
        torch.cos(x_args),
        torch.sin(y_args),
        torch.cos(y_args)
    ], dim=-1) # shape: (num_points, 4 * num_frequencies)

    return encoded


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):

        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class Siren(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30, num_frequencies=10):
        super().__init__()
        
        self.num_frequencies = num_frequencies
        self.in_features_encoded = num_frequencies * 4 
        self.out_features = out_features
        
        self.net = []
        self.net.append(SineLayer(self.in_features_encoded, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            # For outermost_linear, output 2 features for atan2
            final_linear = nn.Linear(hidden_features, 2)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                final_linear.bias.fill_(0)
                
            self.net.append(final_linear)

        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)
        self.outermost_linear = outermost_linear

    def forward(self, coords):
        coords_input = coords.clone().detach().requires_grad_(True)
        
        encoded_coords = fourier_encode(coords_input, self.in_features_encoded, self.num_frequencies)

        output_vector = self.net(encoded_coords)
        
        if self.outermost_linear:
            # Use atan2 for phase output when outermost_linear=True
            output_phase = torch.atan2(output_vector[..., 0], output_vector[..., 1])
        else:
            output_phase = output_vector.squeeze(-1)

        return output_phase, coords_input

if __name__ == "__main__":

    model = Siren(in_features=2, out_features=1, hidden_features=128, hidden_layers=2, outermost_linear=True,
                  first_omega_0=30, hidden_omega_0=30, num_frequencies=8)
    
    num_points = int(512 * 512)
    coords = torch.rand((num_points, 2))
    
    output, coords_out = model(coords)

    print(f"Input coordinates shape: {coords.shape}")
    print(f"Final phase output shape: {output.shape}")

    assert output.dim() == 1 and output.shape[0] == num_points

    try:
        encoded_coords = fourier_encode(coords, model.in_features_encoded, model.num_frequencies)
        macs, params = profile(model.net, inputs=(encoded_coords,), verbose=False)
        gflops = macs * 2
        print(f"Network (self.net) MACs: {macs/1e6:.2f} M, Params: {params/1e6:.2f} M, GFLOPs: {gflops / 1e9:.4f}")
    except Exception as e:
        print(f"Could not profile the model: {e}")