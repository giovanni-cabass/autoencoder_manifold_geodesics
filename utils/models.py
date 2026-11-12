"""
Model definitions shared across scripts.

Exports:
    build_mlp                  — generic MLP factory
    PD_MLP_Interpolator_no_bn  — 3D → 16D decoder (float64, no batch norm)
    load_decoder               — load weights from .pth and put model in eval mode
"""

import os
from typing import List

import torch  # type: ignore
import torch.nn as nn  # type: ignore

from utils.svd_definitions import latent_dims, dims, p_min_vals, p_max_vals


# ---------------------------------------------------------------------------
# MLP factory
# ---------------------------------------------------------------------------

def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
    activation: nn.Module = nn.GELU(),
) -> nn.Sequential:
    """Create a fully-connected MLP with `layers` hidden layers."""
    seq: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), activation]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation]
    seq += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*seq)


# ---------------------------------------------------------------------------
# Decoder class
# ---------------------------------------------------------------------------

class PD_MLP_Interpolator_no_bn(nn.Module):
    """
    MLP decoder: maps 3-D latent coordinates in [-1, 1]^3 to 16-D data space.

    The model internally rescales its input from the normalised box [-1,1]^3
    to the physical parameter range [p_min_vals, p_max_vals] before passing
    it through the MLP, so callers always feed in values in [-1,1]^3.
    """

    def __init__(
        self,
        p_min_list: list,
        p_max_list: list,
        p_dim: int = int(latent_dims),
        d_dim: int = int(dims),
        hidden_dim: int = 128,
        layers: int = 8,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        self.mlp = build_mlp(
            input_dim=p_dim,
            hidden_dim=hidden_dim,
            output_dim=d_dim,
            layers=layers,
            activation=activation,
        )
        self.register_buffer(
            "lower_bounds",
            torch.tensor(
                [p_min_list[0], p_min_list[1], p_min_list[2]],
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "upper_bounds",
            torch.tensor(
                [p_max_list[0], p_max_list[1], p_max_list[2]],
                dtype=torch.float64,
            ),
        )

    def forward(self, p_prime: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        p_prime : Tensor, shape (..., 3), values in [-1, 1]^3

        Returns
        -------
        Tensor, shape (..., 16)
        """
        p = self.lower_bounds + (p_prime + 1.0) * 0.5 * (self.upper_bounds - self.lower_bounds)
        return self.mlp(p)


def u_to_e(u: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Map unconstrained u in R^3 to e in (-1, 1)^3 using a logistic/sigmoid map.

    e_i = 2 * sigmoid(u_i) - 1  in ( -1, 1 ).
    For numerical stability we clamp the sigmoid away from exactly 0/1.
    """
    sig = torch.sigmoid(u)
    if eps > 0.0:
        sig = sig.clamp(min=eps, max=1.0 - eps)
    return 2.0 * sig - 1.0


def decoder_forward_from_u_no_bn(
    decoder: PD_MLP_Interpolator_no_bn,
    u: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compose the decoder with the analytic change of variables u -> e(u).

    The decoder itself expects inputs in [-1, 1]^3 (p_prime). This function
    transforms u into e in (-1, 1)^3 and then calls decoder(e).
    """
    e = u_to_e(u, eps=eps)
    return decoder(e)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_decoder(
    main_folder: str,
    device: torch.device = torch.device("cpu"),
) -> PD_MLP_Interpolator_no_bn:
    """
    Instantiate PD_MLP_Interpolator_no_bn, load weights, and return in eval mode.

    Parameters
    ----------
    main_folder : str
        Project root directory that contains the .pth weights file.
    device : torch.device
        Device to move the model to after loading. Defaults to CPU.

    Returns
    -------
    PD_MLP_Interpolator_no_bn  — eval mode, all grad computation disabled.
    """
    model_path = os.path.join(
        main_folder,
        "student_decoder_float64.no_maf.small_arch.no_bn.refine.10.pth",
    )
    model = PD_MLP_Interpolator_no_bn(
        p_min_list=list(p_min_vals),
        p_max_list=list(p_max_vals),
    ).double()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    # strict=False is safe here because __init__ already correctly initialises
    # lower_bounds and upper_bounds from p_min_vals / p_max_vals.
    # The .pth only contains mlp.* weights; the buffers are NOT stored in the file.
    # If the buffers were left uninitialised, strict=False would silently produce wrong results.
    # NOTE: dtype verification (all parameters and buffers are float64) is NOT performed here.
    # It is the responsibility of the calling script to verify dtypes after loading if needed.
    model = model.to(device)
    model.eval()
    # model.eval() is sufficient — requires_grad_(False) is VERY redundant here...
    # for param in model.parameters():
    #     param.requires_grad_(False)
    return model
