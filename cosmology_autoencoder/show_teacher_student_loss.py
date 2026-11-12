import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

INPUT_DIM: int = 3
OUTPUT_DIM: int = 16
HIDDEN_DIM: int = 128
N_LAYERS: int = 8
BATCH: int = 512

TEACHER_RELATIVE_PATHS: List[str] = [
    os.path.join(
        "teachers",
        "teacher_1",
        "decoder_float64.no_maf.small_arch.no_bn.refine.8.pth",
    ),
    os.path.join(
        "teachers",
        "teacher_2",
        "decoder_float64.no_maf.small_arch.no_bn.refine.5.pth",
    ),
    os.path.join(
        "teachers",
        "teacher_3",
        "decoder_float64.no_maf.small_arch.no_bn.refine.3.pth",
    ),
    os.path.join(
        "teachers",
        "teacher_4",
        "decoder_float64.no_maf.small_arch.no_bn.refine.3.pth",
    ),
    os.path.join(
        "teachers",
        "teacher_5",
        "decoder_float64.no_maf.small_arch.no_bn.refine.2.pth",
    ),
]
STUDENT_RELATIVE_PATH: str = (
    "student_decoder_float64.no_maf.small_arch.no_bn.refine.10.pth"
)
D_RELATIVE_PATH: str = os.path.join("data", "d_valid.npy")
THETA_RELATIVE_PATH: str = os.path.join(
    "data",
    "random_test_points_in_Plin_CAMB_LCDM_Wide.dat.txt",
)


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
) -> nn.Sequential:
    """Linear(3, 128), GELU, then `layers` times (Linear(128, 128), GELU), then Linear(128, 16)."""
    modules: List[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
    ]
    for _ in range(layers):
        modules.append(nn.Linear(hidden_dim, hidden_dim))
        modules.append(nn.GELU())
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


class PD_MLP_Interpolator_no_bn(nn.Module):
    """Forward decoder: cosmological parameters in, 16 normalized coefficients out."""

    def __init__(
        self,
        p_dim: int = INPUT_DIM,
        d_dim: int = OUTPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        layers: int = N_LAYERS,
    ) -> None:
        super().__init__()
        self.mlp: nn.Sequential = build_mlp(
            input_dim=p_dim,
            hidden_dim=hidden_dim,
            output_dim=d_dim,
            layers=layers,
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.mlp(p)


def assert_float64(model: nn.Module, name: str) -> None:
    for param_name, param in model.named_parameters():
        if param.dtype != torch.float64:
            raise RuntimeError(
                f"{name} parameter '{param_name}' has dtype {param.dtype}, expected float64."
            )
    for buffer_name, buffer in model.named_buffers():
        if buffer.dtype != torch.float64:
            raise RuntimeError(
                f"{name} buffer '{buffer_name}' has dtype {buffer.dtype}, expected float64."
            )


def load_decoder(checkpoint_path: str) -> PD_MLP_Interpolator_no_bn:
    model: PD_MLP_Interpolator_no_bn = PD_MLP_Interpolator_no_bn(
        p_dim=INPUT_DIM,
        d_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        layers=N_LAYERS,
    ).double()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    assert_float64(model, checkpoint_path)
    return model


def load_validation_arrays(script_dir: str) -> tuple[np.ndarray, np.ndarray]:
    d_path: str = os.path.join(script_dir, D_RELATIVE_PATH)
    theta_path: str = os.path.join(script_dir, THETA_RELATIVE_PATH)
    d_valid: np.ndarray = np.ascontiguousarray(
        np.load(d_path),
        dtype=np.float64,
    )
    theta: np.ndarray = np.ascontiguousarray(
        np.loadtxt(theta_path, dtype=np.float64),
        dtype=np.float64,
    )
    if d_valid.ndim != 2 or int(d_valid.shape[1]) != OUTPUT_DIM:
        raise ValueError(
            f"d_valid.npy must have shape (N, {OUTPUT_DIM}), got {d_valid.shape}."
        )
    if theta.ndim != 2 or int(theta.shape[1]) != INPUT_DIM:
        raise ValueError(
            f"theta file must have shape (N, {INPUT_DIM}), got {theta.shape}."
        )
    n_d: int = int(d_valid.shape[0])
    n_theta: int = int(theta.shape[0])
    if n_d != n_theta:
        raise ValueError(
            f"Row counts differ: d_valid has {n_d} rows, theta has {n_theta} rows."
        )
    if n_d < BATCH:
        raise ValueError(
            f"Need at least {BATCH} rows, found {n_d}."
        )
    return d_valid, theta


def main() -> None:
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    d_valid, theta = load_validation_arrays(script_dir)
    n_rows: int = int(d_valid.shape[0])

    teachers: List[PD_MLP_Interpolator_no_bn] = []
    for relative_path in TEACHER_RELATIVE_PATHS:
        teacher: PD_MLP_Interpolator_no_bn = load_decoder(
            os.path.join(script_dir, relative_path)
        )
        for param in teacher.parameters():
            param.requires_grad_(False)
        teachers.append(teacher)

    student: PD_MLP_Interpolator_no_bn = load_decoder(
        os.path.join(script_dir, STUDENT_RELATIVE_PATH)
    )
    student.eval()

    theta_batch: torch.Tensor = torch.tensor(
        theta[:BATCH],
        dtype=torch.float64,
    )
    d_batch: torch.Tensor = torch.tensor(
        d_valid[:BATCH],
        dtype=torch.float64,
    )

    with torch.no_grad():
        student_out: torch.Tensor = student(theta_batch)
        teacher_outputs: List[torch.Tensor] = [
            teacher(theta_batch) for teacher in teachers
        ]
        teacher_mean: torch.Tensor = torch.mean(
            torch.stack(teacher_outputs, dim=0),
            dim=0,
        )
        reconstruction: torch.Tensor = nn.functional.mse_loss(student_out, d_batch)
        distillation: torch.Tensor = nn.functional.mse_loss(student_out, teacher_mean)
        total: torch.Tensor = reconstruction + distillation
        reconstruction_value: float = reconstruction.item()
        distillation_value: float = distillation.item()
        total_value: float = total.item()

    print(f"N = {n_rows}")
    print(f"batch size = {BATCH}")
    print(f"reconstruction = {reconstruction_value:.16e}")
    print(f"distillation = {distillation_value:.16e}")
    print(f"sum = {total_value:.16e}")


if __name__ == "__main__":
    main()
