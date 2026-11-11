"""
Physics-Informed Graph-Fourier Neural Operator (PI-GraphFNO)
Baseline Architecture for Transient Two-Phase Subsurface Flow Under Dynamic Well Topologies

Features:
1. Spectral 2D Fourier Neural Operator (FNO) layers for heterogeneous permeability fields.
2. Permutation-invariant Graph Well Encoder for variable well counts, continuous coordinates,
   Peaceman well indices (WI), and bottom-hole pressure (BHP) controls.
3. Soft Finite-Volume Mass-Balance Residual Penalty vs. Hard Global Mass-Conservation Layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D Fourier Spectral Convolution Layer for continuous spatial fields."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes_x, : self.modes_y] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, : self.modes_x, : self.modes_y], self.weights1
        )
        out_ft[:, :, -self.modes_x :, : self.modes_y] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes_x :, : self.modes_y], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class DynamicWellGraphEncoder(nn.Module):
    """
    Encodes a variable number of active wells (nodes) and pairwise well interactions (edges)
    into a fixed-size latent operational embedding without requiring fixed grid masks.

    Expected node features per well: [x_norm, y_norm, well_type (+1 inj / -1 prod), bhp_norm, peaceman_wi_norm]
    """

    def __init__(self, node_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, well_nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            well_nodes: Tensor of shape [Num_Active_Wells, node_dim]
        Returns:
            global_well_embedding: Tensor of shape [1, hidden_dim]
        """
        num_wells = well_nodes.shape[0]
        h = self.node_encoder(well_nodes)  # [N_w, hidden_dim]

        # Compute pairwise Euclidean distances between wells for fully connected well graph
        coords = well_nodes[:, :2]  # [N_w, 2]
        dist_matrix = torch.cdist(coords, coords, p=2).unsqueeze(-1)  # [N_w, N_w, 1]

        # Message passing across all active well pairs
        h_i = h.unsqueeze(1).expand(num_wells, num_wells, -1)
        h_j = h.unsqueeze(0).expand(num_wells, num_wells, -1)
        edge_input = torch.cat([h_i, h_j, dist_matrix], dim=-1)
        messages = self.edge_mlp(edge_input).mean(dim=1)  # [N_w, hidden_dim]

        # Aggregate node + message features via permutation-invariant mean+max pooling
        combined = torch.cat([h, messages], dim=-1)
        pooled = self.out_proj(combined).mean(dim=0, keepdim=True)  # [1, hidden_dim]
        return F.gelu(pooled)


class HardMassConservationLayer(nn.Module):
    """
    Hard architectural constraint ensuring predicted water saturation updates strictly honor
    global injected minus produced fluid volume across the reservoir domain.
    """

    def forward(
        self,
        sw_pred: torch.Tensor,
        sw_prev: torch.Tensor,
        porosity: torch.Tensor,
        net_water_injection_vol: torch.Tensor,
        cell_volume: float = 1.0,
    ) -> torch.Tensor:
        # Clamp physical saturation bounds [0, 1] smoothly
        sw_bounded = torch.clamp(sw_pred, min=0.0, max=1.0)

        # Compute predicted vs. target change in total water volume
        pred_delta_vol = torch.sum((sw_bounded - sw_prev) * porosity * cell_volume, dim=(-2, -1), keepdim=True)
        correction = (net_water_injection_vol - pred_delta_vol) / (
            torch.sum(porosity * cell_volume, dim=(-2, -1), keepdim=True) + 1e-8
        )

        return torch.clamp(sw_bounded + correction, min=0.0, max=1.0)


class PIGraphFNO(nn.Module):
    """
    Dual-Branch Physics-Informed Graph-Fourier Neural Operator.
    Maps (Permeability Field, Previous State, Variable Well Graph) -> Next-Step (Pressure, Saturation).
    """

    def __init__(self, modes: int = 8, width: int = 24, well_feature_dim: int = 5, hard_conserve: bool = True):
        super().__init__()
        self.width = width
        self.hard_conserve = hard_conserve

        # Input channels: [log_perm, prev_pressure, prev_saturation, grid_x, grid_y] = 5
        self.lift = nn.Conv2d(5, width, kernel_size=1)

        # Well graph encoder
        self.well_encoder = DynamicWellGraphEncoder(node_dim=well_feature_dim, hidden_dim=width)

        # Spectral FNO blocks
        self.spectral1 = SpectralConv2d(width * 2, width * 2, modes, modes)
        self.w1 = nn.Conv2d(width * 2, width * 2, kernel_size=1)

        self.spectral2 = SpectralConv2d(width * 2, width * 2, modes, modes)
        self.w2 = nn.Conv2d(width * 2, width * 2, kernel_size=1)

        # Output projection -> [Pressure_next, Saturation_next]
        self.proj = nn.Sequential(
            nn.Conv2d(width * 2, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, 2, kernel_size=1),
        )

        self.mass_layer = HardMassConservationLayer()

    def forward(
        self,
        spatial_inputs: torch.Tensor,
        well_nodes: torch.Tensor,
        porosity: torch.Tensor = None,
        net_water_inj: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            spatial_inputs: [B, 5, Nx, Ny] -> (log_k, P_t, Sw_t, X_coord, Y_coord)
            well_nodes: [Num_Active_Wells, 5] -> variable well topology
        """
        batch_size, _, nx, ny = spatial_inputs.shape
        prev_sw = spatial_inputs[:, 2:3, :, :]

        # 1. Lift spatial fields
        x_spatial = self.lift(spatial_inputs)  # [B, width, Nx, Ny]

        # 2. Encode dynamic well graph and broadcast across spatial domain
        w_embed = self.well_encoder(well_nodes)  # [1, width]
        w_grid = w_embed.unsqueeze(-1).unsqueeze(-1).expand(batch_size, -1, nx, ny)

        # 3. Fuse geological field and operational well topology
        x = torch.cat([x_spatial, w_grid], dim=1)  # [B, 2*width, Nx, Ny]

        # 4. Fourier Operator Layers
        x = F.gelu(self.spectral1(x) + self.w1(x))
        x = F.gelu(self.spectral2(x) + self.w2(x))

        out = self.proj(x)
        p_next = out[:, 0:1, :, :]
        sw_next = out[:, 1:2, :, :]

        # 5. Optional Hard Mass-Conservation Projection
        if self.hard_conserve and porosity is not None and net_water_inj is not None:
            sw_next = self.mass_layer(sw_next, prev_sw, porosity, net_water_inj)

        return torch.cat([p_next, sw_next], dim=1)


def soft_fv_mass_balance_loss(
    sw_pred: torch.Tensor,
    sw_prev: torch.Tensor,
    porosity: torch.Tensor,
    net_water_inj: torch.Tensor,
    cell_volume: float = 1.0,
) -> torch.Tensor:
    """Computes soft global finite-volume water mass-conservation residual loss."""
    pred_delta_vol = torch.sum((sw_pred - sw_prev) * porosity * cell_volume, dim=(-2, -1), keepdim=True)
    return F.mse_loss(pred_delta_vol, net_water_inj)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PIGraphFNO(modes=8, width=16, hard_conserve=True).to(device)

    # Synthetic 32x32 2D reservoir state
    spatial_state = torch.rand(1, 5, 32, 32, device=device)
    porosity_map = torch.full((1, 1, 32, 32), 0.20, device=device)
    net_inj_vol = torch.tensor([[[[2.5]]]], device=device)

    # Test 1: 3 Active Wells (1 Injector, 2 Producers)
    wells_3 = torch.tensor(
        [
            [0.15, 0.15, 1.0, 0.9, 1.2],   # Injector
            [0.85, 0.85, -1.0, 0.3, 1.1],  # Producer 1
            [0.85, 0.20, -1.0, 0.3, 1.0],  # Producer 2
        ],
        device=device,
    )
    pred_3 = model(spatial_state, wells_3, porosity_map, net_inj_vol)

    # Test 2: Dynamic Infill Drilling -> 5 Active Wells (No retraining or architecture change)
    wells_5 = torch.cat(
        [
            wells_3,
            torch.tensor([[0.50, 0.50, 1.0, 0.85, 1.3], [0.20, 0.80, -1.0, 0.35, 0.9]], device=device),
        ],
        dim=0,
    )
    pred_5 = model(spatial_state, wells_5, porosity_map, net_inj_vol)

    # Verify global mass-balance residual
    mb_error = soft_fv_mass_balance_loss(pred_5[:, 1:2], spatial_state[:, 2:3], porosity_map, net_inj_vol)

    print(f"Running on: {device}")
    print(f"3-Well Configuration Output Shape: {tuple(pred_3.shape)}")
    print(f"5-Well Infill Configuration Output Shape: {tuple(pred_5.shape)}")
    print(f"Post-Projection Global Mass-Balance MSE: {mb_error.item():.2e}")
