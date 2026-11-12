"""Task 43 flat-chart time-1 flow, ReduceLROnPlateau sibling.

Same shear, network, RK4, probes, and closure check as
run_task43_flat_chart_flow.py. The learning rate starts at LR and is cut by
ReduceLROnPlateau when max rho stops improving. The step cap is
MAX_OPT_STEPS.
"""

import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch

from utils.models import build_mlp
from utils.logging_scripts import Tee
from utils.workspace_folders import out_err_folder

# Frozen §1 knobs, plus the §5 step cap. Later waves read this header.
# They do not add a second constants block and do not retune these values.
# Phase 1 reads A_PSI, the box, and N_PER_AXIS. The other names are declared
# for later waves. This wave does not apply them.
A_PSI: np.float64 = np.float64(0.5)
THETA_MIN: np.float64 = np.float64(-0.5)
THETA_MAX: np.float64 = np.float64(0.5)
N_PER_AXIS: np.int64 = np.int64(5)
SEED: np.int64 = np.int64(42)
DT_INIT: np.float64 = np.float64(1.0 / 16.0)
DT_MIN: np.float64 = np.float64(1.0 / 512.0)
HALVE_EVERY_STEPS: np.int64 = np.int64(10)
R_MAX: np.float64 = np.float64(1.0e-6)
RHO_PASS: np.float64 = np.float64(1.0e-4)
RHO_MARK: np.float64 = np.float64(1.0e-2)
MLP_WIDTH: np.int64 = np.int64(64)
LR: np.float64 = np.float64(1.0e-3)
WEIGHT_DECAY: np.float64 = np.float64(0.0)
# Plateau on max rho. Factor 0.1 is one decade per stall. The finished
# patience-50 run cut at step 366 and was on the 1e-6 floor by step 472.
# Patience 500 does not treat a 51-step pause as a stall. Cooldown 500
# keeps the new rate for 500 steps before another cut can be counted.
# Threshold 1e-3 is the relative improvement the sqrt(g) trainers call
# "improved". The floor stops the cuts at 1e-6.
FACTOR_SCHEDULER: np.float64 = np.float64(0.1)
PATIENCE_SCHEDULER: np.int64 = np.int64(500)
COOLDOWN_SCHEDULER: np.int64 = np.int64(500)
THRESHOLD_SCHEDULER: np.float64 = np.float64(1.0e-3)
LR_MIN: np.float64 = np.float64(1.0e-6)
G_ABS_ATOL: np.float64 = np.float64(1.0e-10)
FD_STEP: np.float64 = np.float64(1.0e-6)
FD_REL_ATOL: np.float64 = np.float64(1.0e-4)
R_ZERO_MAX: np.float64 = np.float64(1.0e-12)
# The patience-50 run was still descending at step 10000. This cap
# continues past that point.
MAX_OPT_STEPS: np.int64 = np.int64(20000)


def main() -> None:
    # --- SETUP LOGGING ---
    start_time = datetime.now()
    tz_name = time.tzname[0] if time.tzname else "UTC"

    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    out_name = (
        f'{script_name}_output_{start_time.strftime("%Y%m%d_%H%M%S")}_{tz_name}.txt'
    )
    err_name = (
        f'{script_name}_errors_{start_time.strftime("%Y%m%d_%H%M%S")}_{tz_name}.txt'
    )

    log_file_path = os.path.join(out_err_folder, out_name)
    error_file_path = os.path.join(out_err_folder, err_name)

    log_file_handle = open(log_file_path, "w", encoding="utf-8")
    error_file_handle = open(error_file_path, "w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        sys.stdout = Tee(original_stdout, log_file_handle)
        sys.stderr = Tee(original_stderr, error_file_handle)

        print(f"🚀 Script: {script_name}.py")
        print(f"📅 Started: {start_time}")

        # float64 before any tensor or module. One GPU, cuda:0.
        # Phase 2 applies SEED before build_mlp. This wave does not.
        torch.set_default_dtype(torch.float64)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "this script requires one CUDA device; none is available"
            )
        torch.cuda.set_device(0)
        device: torch.device = torch.device("cuda:0")
        warmup_anchor: torch.Tensor = torch.zeros(
            1, dtype=torch.float64, device=device
        )
        torch.cuda.synchronize(device)
        del warmup_anchor
        warmup_x: torch.Tensor = torch.zeros(
            2, 2, dtype=torch.float64, device=device, requires_grad=True
        )
        warmup_y: torch.Tensor = torch.mm(warmup_x, warmup_x.t())
        warmup_y.sum().backward()
        torch.cuda.synchronize(device)
        del warmup_x, warmup_y

        # === TASK43 PHASE1 FLAT CONTROL ===
        # build_mlp is imported for Phase 2. This wave does not call it.
        print(
            "Frozen knobs (phase 1 applies A_PSI, the box, and N_PER_AXIS only): "
            f"A_PSI={float(A_PSI)} "
            f"THETA_MIN={float(THETA_MIN)} THETA_MAX={float(THETA_MAX)} "
            f"N_PER_AXIS={int(N_PER_AXIS)} SEED={int(SEED)} "
            f"DT_INIT={float(DT_INIT)} DT_MIN={float(DT_MIN)} "
            f"HALVE_EVERY_STEPS={int(HALVE_EVERY_STEPS)} "
            f"R_MAX={float(R_MAX)} RHO_PASS={float(RHO_PASS)} "
            f"RHO_MARK={float(RHO_MARK)} MLP_WIDTH={int(MLP_WIDTH)} "
            f"LR={float(LR)} WEIGHT_DECAY={float(WEIGHT_DECAY)} "
            f"FACTOR_SCHEDULER={float(FACTOR_SCHEDULER)} "
            f"PATIENCE_SCHEDULER={int(PATIENCE_SCHEDULER)} "
            f"COOLDOWN_SCHEDULER={int(COOLDOWN_SCHEDULER)} "
            f"THRESHOLD_SCHEDULER={float(THRESHOLD_SCHEDULER)} "
            f"LR_MIN={float(LR_MIN)} "
            f"G_ABS_ATOL={float(G_ABS_ATOL)} FD_STEP={float(FD_STEP)} "
            f"FD_REL_ATOL={float(FD_REL_ATOL)} R_ZERO_MAX={float(R_ZERO_MAX)} "
            f"MAX_OPT_STEPS={int(MAX_OPT_STEPS)} "
            f"build_mlp={build_mlp.__name__}"
        )

        def psi(theta: torch.Tensor) -> torch.Tensor:
            # Embedding of Euclidean 3-space.
            # psi(theta) = (theta_0 + a theta_1^2, theta_1, theta_2), a = A_PSI.
            # Inverse, documentation only: theta_0 = y_0 - a y_1^2,
            # theta_1 = y_1, theta_2 = y_2. The learned chart is not asked
            # to equal this inverse. The later pass bar is g^(u) -> I_3.
            if int(theta.shape[-1]) != 3:
                raise ValueError(
                    f"psi expected last axis length 3, got {tuple(theta.shape)}"
                )
            theta_0: torch.Tensor = theta[..., 0]
            theta_1: torch.Tensor = theta[..., 1]
            theta_2: torch.Tensor = theta[..., 2]
            y_0: torch.Tensor = theta_0 + (theta_1 * theta_1) * A_PSI
            y: torch.Tensor = torch.stack((y_0, theta_1, theta_2), dim=-1)
            return y.to(dtype=torch.float64)

        def d_psi(theta: torch.Tensor) -> torch.Tensor:
            # Analytic Jacobian. Rows are components of psi.
            # Columns are partial / partial theta_mu.
            # D psi = [[1, 2 a theta_1, 0], [0, 1, 0], [0, 0, 1]].
            # det D psi = 1: psi preserves orientation and volume.
            # That identity is not the flatness test.
            if int(theta.shape[-1]) != 3:
                raise ValueError(
                    f"d_psi expected last axis length 3, got {tuple(theta.shape)}"
                )
            theta_1: torch.Tensor = theta[..., 1]
            zeros: torch.Tensor = torch.zeros_like(theta_1)
            ones: torch.Tensor = torch.ones_like(theta_1)
            two_a: np.float64 = np.float64(2.0) * A_PSI
            row_0: torch.Tensor = torch.stack(
                (ones, theta_1 * two_a, zeros), dim=-1
            )
            row_1: torch.Tensor = torch.stack((zeros, ones, zeros), dim=-1)
            row_2: torch.Tensor = torch.stack((zeros, zeros, ones), dim=-1)
            jac: torch.Tensor = torch.stack((row_0, row_1, row_2), dim=-2)
            return jac.to(dtype=torch.float64).contiguous()

        n_per_axis: int = int(N_PER_AXIS)
        axis: torch.Tensor = torch.linspace(
            float(THETA_MIN),
            float(THETA_MAX),
            n_per_axis,
            dtype=torch.float64,
            device=device,
        )
        theta_0_axis: torch.Tensor
        theta_1_axis: torch.Tensor
        theta_2_axis: torch.Tensor
        theta_0_axis, theta_1_axis, theta_2_axis = torch.meshgrid(
            axis, axis, axis, indexing="ij"
        )
        theta_grid: torch.Tensor = torch.stack(
            (
                torch.reshape(theta_0_axis, (-1,)),
                torch.reshape(theta_1_axis, (-1,)),
                torch.reshape(theta_2_axis, (-1,)),
            ),
            dim=1,
        ).to(dtype=torch.float64, device=device).contiguous()

        n_points: int = n_per_axis ** 3
        if theta_grid.shape != (n_points, 3):
            raise RuntimeError(
                f"theta_grid shape {tuple(theta_grid.shape)} != {(n_points, 3)}"
            )
        if theta_grid.dtype != torch.float64:
            raise RuntimeError(f"theta_grid dtype {theta_grid.dtype} != float64")
        if not bool(theta_grid.is_contiguous()):
            raise RuntimeError("theta_grid is not contiguous")
        if theta_grid.device != device:
            raise RuntimeError(
                f"theta_grid device {theta_grid.device} != {device}"
            )

        def g_theta(theta: torch.Tensor) -> torch.Tensor:
            # Pullback of the Euclidean dot product. Left as a matrix product
            # on the last two axes: g = (D psi)^T (D psi). Components of g
            # are not written out.
            jac: torch.Tensor = d_psi(theta)
            g: torch.Tensor = torch.matmul(jac.transpose(-1, -2), jac)
            return g.to(dtype=torch.float64)

        def rho_from_g(g: torch.Tensor) -> torch.Tensor:
            # rho = ||g - I_3||_F / sqrt(3). sqrt(3) = ||I_3||_F.
            # Divide by sqrt(3), not by 3.
            eye: torch.Tensor = torch.eye(
                3, dtype=torch.float64, device=g.device
            )
            deviation: torch.Tensor = g - eye
            frobenius: torch.Tensor = torch.linalg.matrix_norm(
                deviation, ord="fro"
            )
            sqrt_three: torch.Tensor = torch.sqrt(
                torch.tensor(
                    np.float64(3.0), dtype=torch.float64, device=g.device
                )
            )
            rho: torch.Tensor = frobenius / sqrt_three
            return rho.to(dtype=torch.float64)

        y_on_grid: torch.Tensor = psi(theta_grid)
        g_on_grid: torch.Tensor = g_theta(theta_grid)
        rho_on_grid: torch.Tensor = rho_from_g(g_on_grid)
        print(
            "Phase 1 flat control: "
            f"psi {tuple(y_on_grid.shape)} "
            f"d_psi {tuple(d_psi(theta_grid).shape)} "
            f"theta_grid {tuple(theta_grid.shape)} "
            f"dtype={theta_grid.dtype} "
            f"contiguous={bool(theta_grid.is_contiguous())} "
            f"device={theta_grid.device} "
            f"g_theta {tuple(g_on_grid.shape)} "
            f"rho {tuple(rho_on_grid.shape)}"
        )

        # === TASK43 PHASE2 FLOW ===
        # v_eta and the time-1 chart. build_mlp always starts with one
        # Linear+activation pair, then appends `layers` further
        # Linear+activation pairs, then a bare output Linear. Hidden
        # Linear stages = 1 + layers. Three hidden stages of width
        # MLP_WIDTH, so the argument is 2. That integer is not the
        # contract. The contract is the Sequential 3 -> 64 -> 64 -> 64 -> 3,
        # checked against the module list below before any training.
        # The factory inserts one GELU instance in every hidden slot.
        chart_dim: int = int(theta_grid.shape[1])
        if chart_dim != 3:
            raise RuntimeError(
                f"chart dimension {chart_dim} != 3"
            )
        n_hidden_stages: int = 3
        layers_argument: int = n_hidden_stages - 1
        gelu: torch.nn.GELU = torch.nn.GELU()
        torch.manual_seed(int(SEED))
        v_eta: torch.nn.Sequential = build_mlp(
            input_dim=chart_dim,
            hidden_dim=int(MLP_WIDTH),
            output_dim=chart_dim,
            layers=layers_argument,
            activation=gelu,
        ).double().to(device)

        child_modules: list[torch.nn.Module] = list(v_eta)
        expected_kinds: tuple[type, ...] = (
            torch.nn.Linear,
            torch.nn.GELU,
            torch.nn.Linear,
            torch.nn.GELU,
            torch.nn.Linear,
            torch.nn.GELU,
            torch.nn.Linear,
        )
        if len(child_modules) != len(expected_kinds):
            raise RuntimeError(
                f"v_eta has {len(child_modules)} children, "
                f"expected {len(expected_kinds)} for 3->64->64->64->3"
            )
        for child_index, (child, expected_kind) in enumerate(
            zip(child_modules, expected_kinds)
        ):
            if not isinstance(child, expected_kind):
                raise RuntimeError(
                    f"v_eta[{child_index}] is {type(child).__name__}, "
                    f"expected {expected_kind.__name__}"
                )
        linear_children: list[torch.nn.Linear] = [
            child
            for child in child_modules
            if isinstance(child, torch.nn.Linear)
        ]
        if len(linear_children) != 4:
            raise RuntimeError(
                f"v_eta has {len(linear_children)} Linear modules, expected 4"
            )
        weight_shapes: list[tuple[int, int]] = [
            (int(linear.weight.shape[0]), int(linear.weight.shape[1]))
            for linear in linear_children
        ]
        width: int = int(MLP_WIDTH)
        expected_shapes: list[tuple[int, int]] = [
            (width, chart_dim),
            (width, width),
            (width, width),
            (chart_dim, width),
        ]
        if weight_shapes != expected_shapes:
            raise RuntimeError(
                f"weight shapes {weight_shapes} != {expected_shapes} "
                "(Sequential 3->64->64->64->3)"
            )
        output_linear: torch.nn.Linear = linear_children[-1]
        if output_linear is not child_modules[-1]:
            raise RuntimeError(
                "output Linear is not the last module of v_eta"
            )
        if output_linear.bias is None:
            raise RuntimeError("output Linear has no bias")
        print("Phase 2 v_eta module list:")
        print(v_eta)
        for child_index, child in enumerate(child_modules):
            if isinstance(child, torch.nn.Linear):
                child_bias = child.bias
                if child_bias is None:
                    raise RuntimeError(
                        f"v_eta[{child_index}] Linear has no bias"
                    )
                print(
                    f"  [{child_index}] Linear "
                    f"weight {tuple(child.weight.shape)} "
                    f"bias {tuple(child_bias.shape)}"
                )
            else:
                print(f"  [{child_index}] {type(child).__name__}")
        print(
            "Phase 2 four weight shapes: "
            + ", ".join(
                f"Linear[{linear_index}] {shape}"
                for linear_index, shape in enumerate(weight_shapes)
            )
        )
        print(
            "Phase 2 Sequential confirmed: 3->64->64->64->3 "
            "(three hidden Linear stages, each followed by GELU, "
            "then an output Linear with no activation)"
        )

        # Initial field vanishes, so the initial time-1 map is the identity.
        # In-place fill keeps the output Linear parameters as leaves.
        # Hidden Linear weights are left as build_mlp initialized them.
        with torch.no_grad():
            output_linear.weight.zero_()
            output_linear.bias.zero_()
        output_weight_max: float = float(
            torch.max(torch.abs(output_linear.weight)).item()
        )
        output_bias_max: float = float(
            torch.max(torch.abs(output_linear.bias)).item()
        )
        if output_weight_max != 0.0 or output_bias_max != 0.0:
            raise RuntimeError(
                "output Linear weight and bias were not set to zero: "
                f"weight maxabs {output_weight_max} "
                f"bias maxabs {output_bias_max}"
            )
        for hidden_index, hidden_linear in enumerate(linear_children[:-1]):
            if hidden_linear.bias is None:
                raise RuntimeError(
                    f"hidden Linear[{hidden_index}] has no bias"
                )
            hidden_weight_max: float = float(
                torch.max(torch.abs(hidden_linear.weight)).item()
            )
            if hidden_weight_max == 0.0:
                raise RuntimeError(
                    f"hidden Linear[{hidden_index}] weight is zero; "
                    "only the output Linear is initialized at 0"
                )
        print(
            "Phase 2 output Linear zeroed: "
            f"index {len(child_modules) - 1} "
            f"weight {tuple(output_linear.weight.shape)} "
            f"weight_maxabs {output_weight_max} "
            f"bias {tuple(output_linear.bias.shape)} "
            f"bias_maxabs {output_bias_max}"
        )

        all_float64: bool = (
            all(param.dtype == torch.float64 for param in v_eta.parameters())
            and all(buf.dtype == torch.float64 for buf in v_eta.buffers())
        )
        print("\n--- Verifying Model Dtype ---", flush=True)
        for name, param in v_eta.named_parameters():
            status: str = "V" if param.dtype == torch.float64 else "X"
            print(
                f"  [{status}] param  '{name}': {param.dtype} "
                f"device={param.device}",
                flush=True,
            )
            if param.device != device:
                raise RuntimeError(
                    f"param '{name}' device {param.device} != {device}"
                )
        for name, buf in v_eta.named_buffers():
            status = "V" if buf.dtype == torch.float64 else "X"
            print(
                f"  [{status}] buffer '{name}': {buf.dtype}",
                flush=True,
            )
        if not all_float64:
            raise RuntimeError(
                "MODEL DTYPE MISMATCH: not all parameters/buffers are float64."
            )
        print(
            "SUCCESS: All parameters and buffers verified as float64.\n",
            flush=True,
        )

        def rk4_step_count(dt_step: torch.Tensor) -> int:
            # Step count is 1/dt. Allowed steps are dyadic, so this is an integer.
            if dt_step.dtype != torch.float64:
                raise RuntimeError(
                    f"dt dtype {dt_step.dtype} != float64"
                )
            one: torch.Tensor = torch.tensor(
                np.float64(1.0), dtype=torch.float64, device=dt_step.device
            )
            reciprocal: torch.Tensor = one / dt_step
            nearest: torch.Tensor = torch.round(reciprocal)
            reciprocal_value: float = float(reciprocal.item())
            nearest_value: float = float(nearest.item())
            if reciprocal_value != nearest_value:
                raise ValueError(
                    f"1/dt = {reciprocal_value} is not an integer step count"
                )
            n_steps: int = int(nearest_value)
            if n_steps < 1:
                raise ValueError(f"RK4 step count {n_steps} < 1")
            return n_steps

        def rk4_time1(
            x0: torch.Tensor,
            field: torch.nn.Module,
            dt_step: torch.Tensor,
            negate_field: bool,
        ) -> torch.Tensor:
            # Classical RK4 from parameter 0 to 1 at fixed step dt_step.
            # Slopes: k1 = f(x), k2 = f(x + dt/2 k1), k3 = f(x + dt/2 k2),
            # k4 = f(x + dt k3). Advance dt*(k1 + 2 k2 + 2 k3 + k4)/6.
            # The factor 1/2 is the RK4 stage midpoint. dt_step is not cut here.
            # negate_field applies the same scheme to -field.
            if x0.dtype != torch.float64:
                raise RuntimeError(f"RK4 state dtype {x0.dtype} != float64")
            if int(x0.shape[-1]) != chart_dim:
                raise RuntimeError(
                    f"RK4 state last axis {int(x0.shape[-1])} != {chart_dim}"
                )
            if x0.device != dt_step.device:
                raise RuntimeError(
                    f"RK4 state device {x0.device} != dt device {dt_step.device}"
                )
            n_steps: int = rk4_step_count(dt_step)
            sign_value: np.float64 = (
                np.float64(-1.0) if negate_field else np.float64(1.0)
            )
            sign: torch.Tensor = torch.tensor(
                sign_value, dtype=torch.float64, device=x0.device
            )
            half: torch.Tensor = torch.tensor(
                np.float64(0.5), dtype=torch.float64, device=x0.device
            )
            two: torch.Tensor = torch.tensor(
                np.float64(2.0), dtype=torch.float64, device=x0.device
            )
            six: torch.Tensor = torch.tensor(
                np.float64(6.0), dtype=torch.float64, device=x0.device
            )
            state: torch.Tensor = x0
            for _step in range(n_steps):
                k1: torch.Tensor = sign * field(state)
                k2: torch.Tensor = sign * field(state + (half * dt_step) * k1)
                k3: torch.Tensor = sign * field(state + (half * dt_step) * k2)
                k4: torch.Tensor = sign * field(state + dt_step * k3)
                state = state + (
                    dt_step * (k1 + two * k2 + two * k3 + k4)
                ) / six
            return state

        def phi_eta(x: torch.Tensor, dt_step: torch.Tensor) -> torch.Tensor:
            # Time-1 flow of v_eta.
            return rk4_time1(x, v_eta, dt_step, negate_field=False)

        def phi_eta_inverse(
            theta: torch.Tensor,
            dt_step: torch.Tensor,
        ) -> torch.Tensor:
            # Training coordinate u = phi_eta^{-1}(theta):
            # the same RK4 applied to -v_eta, same dt, same step count.
            return rk4_time1(theta, v_eta, dt_step, negate_field=True)

        dt: torch.Tensor = torch.tensor(
            np.float64(DT_INIT), dtype=torch.float64, device=device
        )
        n_steps_init: int = rk4_step_count(dt)
        print(
            "Phase 2 flow: "
            f"seed={int(SEED)} before build_mlp "
            f"dt={float(dt.item())} n_steps={n_steps_init} "
            f"(n_steps = 1/dt) "
            f"phi_eta={phi_eta.__name__} "
            f"phi_eta_inverse={phi_eta_inverse.__name__} "
            "training coordinate u = phi_eta_inverse(theta) "
            "RK4 advance dt*(k1+2*k2+2*k3+k4)/6 "
            "inverse field = -v_eta"
        )

        # === TASK43 PHASE3 LOSS PROBES ===
        # Induced metric of f(u) = psi(phi_eta(u)).
        # J^a_mu = d f^a / d u^mu, ambient factor delta (Euclidean),
        # g = J^T delta J via einsum("ai,ab,bj->ij", jac, delta, jac).
        # Training loss: mean over the grid of ||g - I_3||_F.
        # No reconstruction term. No pairwise term.
        # rho = ||g - I_3||_F / sqrt(3), sqrt(3) = ||I_3||_F.
        # max rho is logged. It is not the scalar that is optimized.
        # Probes run before any optimizer step. A failure raises here.

        def induced_g_one(u_point: torch.Tensor) -> torch.Tensor:
            # Pullback at one chart point. jacrev differentiates the point.
            # v_eta is called by phi_eta; its weights stay the module leaves.
            if u_point.ndim != 1 or int(u_point.shape[0]) != chart_dim:
                raise RuntimeError(
                    f"induced_g_one expected shape ({chart_dim},), "
                    f"got {tuple(u_point.shape)}"
                )
            if u_point.dtype != torch.float64:
                raise RuntimeError(
                    f"induced_g_one dtype {u_point.dtype} != float64"
                )

            def chart_map(u_var: torch.Tensor) -> torch.Tensor:
                return psi(phi_eta(u_var, dt))

            jac: torch.Tensor = torch.func.jacrev(chart_map)(u_point)
            if jac.shape != (chart_dim, chart_dim):
                raise RuntimeError(
                    f"Jacobian shape {tuple(jac.shape)} != "
                    f"{(chart_dim, chart_dim)}"
                )
            if jac.dtype != torch.float64:
                raise RuntimeError(
                    f"Jacobian dtype {jac.dtype} != float64"
                )
            ambient_delta: torch.Tensor = torch.eye(
                chart_dim, dtype=torch.float64, device=u_point.device
            )
            g_point: torch.Tensor = torch.einsum(
                "ai,ab,bj->ij", jac, ambient_delta, jac
            )
            if g_point.dtype != torch.float64:
                raise RuntimeError(
                    f"g(u) dtype {g_point.dtype} != float64"
                )
            return g_point.contiguous()

        def induced_g_batch(u_batch: torch.Tensor) -> torch.Tensor:
            # vmap of the pointwise pullback. vmap is outside jacrev.
            if u_batch.ndim != 2 or int(u_batch.shape[1]) != chart_dim:
                raise RuntimeError(
                    f"induced_g_batch expected (N, {chart_dim}), "
                    f"got {tuple(u_batch.shape)}"
                )
            if u_batch.dtype != torch.float64:
                raise RuntimeError(
                    f"induced_g_batch dtype {u_batch.dtype} != float64"
                )
            g_batch: torch.Tensor = torch.func.vmap(
                induced_g_one, randomness="same"
            )(u_batch)
            if g_batch.shape != (int(u_batch.shape[0]), chart_dim, chart_dim):
                raise RuntimeError(
                    f"batched g shape {tuple(g_batch.shape)} != "
                    f"{(int(u_batch.shape[0]), chart_dim, chart_dim)}"
                )
            if g_batch.dtype != torch.float64:
                raise RuntimeError(
                    f"batched g dtype {g_batch.dtype} != float64"
                )
            return g_batch.contiguous()

        def frobenius_to_identity(g_cloud: torch.Tensor) -> torch.Tensor:
            # ||g - I_3||_F per point. Not divided by sqrt(3) and not by 3.
            eye: torch.Tensor = torch.eye(
                chart_dim, dtype=torch.float64, device=g_cloud.device
            )
            deviation: torch.Tensor = g_cloud - eye
            frobenius: torch.Tensor = torch.linalg.matrix_norm(
                deviation, ord="fro"
            )
            return frobenius

        def chart_state_on_grid() -> tuple[torch.Tensor, torch.Tensor]:
            # u = phi_eta^{-1}(theta), then g^{(u)} of f = psi o phi_eta.
            u_cloud: torch.Tensor = phi_eta_inverse(theta_grid, dt).contiguous()
            g_cloud: torch.Tensor = induced_g_batch(u_cloud)
            return g_cloud, u_cloud

        def mean_frobenius_loss() -> torch.Tensor:
            # Scalar Phase 4 differentiates. Mean over the grid, not max rho.
            g_cloud: torch.Tensor
            g_cloud, _u_cloud = chart_state_on_grid()
            loss: torch.Tensor = frobenius_to_identity(g_cloud).mean()
            if loss.ndim != 0:
                raise RuntimeError(
                    f"training loss ndim {loss.ndim} != 0"
                )
            if loss.dtype != torch.float64:
                raise RuntimeError(
                    f"training loss dtype {loss.dtype} != float64"
                )
            return loss

        def relative_inverse_closure(
            u_cloud: torch.Tensor,
            dt_step: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # r from the parent spec §5. Miss length is the Euclidean norm,
            # not the largest component. R is the root-mean-square Euclidean
            # distance of the cloud from its mean. Do not divide by |u|.
            if u_cloud.ndim != 2 or int(u_cloud.shape[1]) != chart_dim:
                raise RuntimeError(
                    f"closure cloud expected (N, {chart_dim}), "
                    f"got {tuple(u_cloud.shape)}"
                )
            if int(u_cloud.shape[0]) < 1:
                raise RuntimeError("closure cloud is empty")
            center: torch.Tensor = u_cloud.mean(dim=0, keepdim=True)
            offset: torch.Tensor = u_cloud - center
            dist_sq: torch.Tensor = torch.sum(offset * offset, dim=-1)
            radius: torch.Tensor = torch.sqrt(dist_sq.mean())
            if not bool(torch.isfinite(radius).all()):
                raise RuntimeError("closure radius R is not finite")
            if float(radius.detach().item()) == 0.0:
                raise RuntimeError(
                    "closure radius R is 0; relative inverse closure "
                    "is undefined and the check fails"
                )
            mapped: torch.Tensor = phi_eta_inverse(
                phi_eta(u_cloud, dt_step), dt_step
            )
            miss: torch.Tensor = mapped - u_cloud
            miss_len: torch.Tensor = torch.linalg.vector_norm(
                miss, ord=2, dim=-1
            )
            if not bool(torch.isfinite(miss_len).all()):
                raise RuntimeError("inverse-closure miss lengths are not finite")
            max_miss: torch.Tensor = miss_len.max()
            r_closure: torch.Tensor = max_miss / radius
            return r_closure, radius, max_miss

        print(
            "Phase 3 training loss: mean over the grid of "
            "||g^(u) - I_3||_F. "
            "rho = ||g^(u) - I_3||_F / sqrt(3), "
            "sqrt(3) = ||I_3||_F. "
            "max rho is logged and is not the training loss. "
            f"n_points={n_points}"
        )

        # --- Probes 1 and 2: zero-field values ---
        with torch.no_grad():
            g_num: torch.Tensor
            u_cloud: torch.Tensor
            g_num, u_cloud = chart_state_on_grid()
            loss_scalar: torch.Tensor = frobenius_to_identity(g_num).mean()

        if g_num.shape != g_on_grid.shape:
            raise RuntimeError(
                f"numerical g shape {tuple(g_num.shape)} != "
                f"analytic reference {tuple(g_on_grid.shape)}"
            )
        if g_num.dtype != torch.float64 or loss_scalar.dtype != torch.float64:
            raise RuntimeError(
                f"probe dtype g={g_num.dtype} loss={loss_scalar.dtype} "
                "!= float64"
            )
        if not bool(torch.isfinite(g_num).all()):
            raise RuntimeError("numerical g has a non-finite entry")
        if not bool(torch.isfinite(loss_scalar).all()):
            raise RuntimeError("training loss is not finite")

        loss_value: float = float(loss_scalar.detach().item())
        rho_num: torch.Tensor = rho_from_g(g_num)
        max_rho: float = float(rho_num.max().detach().item())
        g_abs_max: float = float((g_num - g_on_grid).abs().max().detach().item())
        u_vs_theta_max: float = float(
            (u_cloud - theta_grid).abs().max().detach().item()
        )
        print(
            "Phase 3 zero-field loss scalar "
            f"(mean ||g - I_3||_F) = {loss_value:.16e}"
        )
        print(
            "Phase 3 zero-field max rho = "
            f"{max_rho:.16e} "
            f"(gate > {float(RHO_MARK):.16e}; not the training loss)"
        )
        print(
            "Phase 3 zero-field max |u - theta| = "
            f"{u_vs_theta_max:.16e}"
        )

        # Probe 1. One max-abs over the matrix difference. No component split.
        print(
            "Phase 3 probe 1 analytic metric: "
            f"max |g_num - g_theta| = {g_abs_max:.16e} "
            f"gate < {float(G_ABS_ATOL):.16e}"
        )
        if not (g_abs_max < float(G_ABS_ATOL)):
            raise RuntimeError(
                "Phase 3 probe 1 failed: zero-field numerical g vs the "
                "Phase 1 analytic reference, "
                f"max abs entry {g_abs_max:.16e} "
                f">= {float(G_ABS_ATOL):.16e}"
            )
        print(
            "Phase 3 probe 1 max rho: "
            f"{max_rho:.16e} "
            f"gate > {float(RHO_MARK):.16e}"
        )
        if not (max_rho > float(RHO_MARK)):
            raise RuntimeError(
                "Phase 3 probe 1 failed: zero-field max rho "
                f"{max_rho:.16e} <= {float(RHO_MARK):.16e}"
            )

        # Probe 2. Every grid point, batched g against a one-point g.
        g_one_rows: list[torch.Tensor] = []
        with torch.no_grad():
            for point_index in range(n_points):
                u_point: torch.Tensor = u_cloud[point_index].contiguous()
                g_one_rows.append(induced_g_one(u_point))
        g_one_stack: torch.Tensor = torch.stack(g_one_rows, dim=0).contiguous()
        if g_one_stack.shape != g_num.shape:
            raise RuntimeError(
                f"one-point g stack {tuple(g_one_stack.shape)} != "
                f"batched g {tuple(g_num.shape)}"
            )
        per_point_abs: torch.Tensor = (g_num - g_one_stack).abs().amax(
            dim=(1, 2)
        )
        worst_point: int = int(torch.argmax(per_point_abs).item())
        batch_abs_max: float = float(per_point_abs[worst_point].detach().item())
        print(
            "Phase 3 probe 2 batching: "
            f"max over points of max |g_batch - g_one| = "
            f"{batch_abs_max:.16e} "
            f"worst point {worst_point} "
            f"gate < {float(G_ABS_ATOL):.16e} "
            f"on all {n_points} points"
        )
        if not (batch_abs_max < float(G_ABS_ATOL)):
            raise RuntimeError(
                "Phase 3 probe 2 failed: batched g vs one-point g, "
                f"max abs entry {batch_abs_max:.16e} at point {worst_point} "
                f">= {float(G_ABS_ATOL):.16e}"
            )

        # Probe 3. Central difference of the training loss on one
        # output-layer weight, then autograd of that same loss.
        # Restore the zero output layer before probe 4.
        if output_linear.weight.ndim != 2:
            raise RuntimeError(
                "output-layer weight is not a matrix, "
                f"shape {tuple(output_linear.weight.shape)}"
            )
        n_out_rows: int = int(output_linear.weight.shape[0])
        n_out_cols: int = int(output_linear.weight.shape[1])
        if n_out_rows != chart_dim or n_out_cols < 1:
            raise RuntimeError(
                "output-layer weight shape "
                f"{tuple(output_linear.weight.shape)} is not "
                f"({chart_dim}, width)"
            )
        output_bias = output_linear.bias
        if output_bias is None:
            raise RuntimeError("output Linear has no bias")
        output_weight_maxabs: float = float(
            output_linear.weight.detach().abs().max().item()
        )
        output_bias_maxabs: float = float(
            output_bias.detach().abs().max().item()
        )
        if output_weight_maxabs != 0.0 or output_bias_maxabs != 0.0:
            raise RuntimeError(
                "Probe 3 expected the zero field, "
                f"weight maxabs {output_weight_maxabs:.16e} "
                f"bias maxabs {output_bias_maxabs:.16e}"
            )
        output_weight_name: str = ""
        for param_name, param in v_eta.named_parameters():
            if param is output_linear.weight:
                output_weight_name = param_name
                break
        if output_weight_name == "":
            raise RuntimeError(
                "output-layer weight is not a named parameter of v_eta"
            )

        saved_output_weight: torch.Tensor = (
            output_linear.weight.detach().clone()
        )
        h_step: torch.Tensor = torch.tensor(
            np.float64(FD_STEP), dtype=torch.float64, device=device
        )
        two_step: torch.Tensor = torch.tensor(
            np.float64(2.0), dtype=torch.float64, device=device
        )
        # |L(w+h) - L(w-h)| at roundoff sits near a few ulps of L.
        # A live output entry is orders of magnitude above this floor.
        eps64: torch.Tensor = torch.tensor(
            np.float64(np.finfo(np.float64).eps),
            dtype=torch.float64,
            device=device,
        )
        ulp_factor: torch.Tensor = torch.tensor(
            np.float64(1000.0), dtype=torch.float64, device=device
        )
        loss_floor_ref: torch.Tensor = torch.maximum(
            loss_scalar.detach().abs(),
            torch.tensor(np.float64(1.0), dtype=torch.float64, device=device),
        )
        numerator_floor: torch.Tensor = ulp_factor * eps64 * loss_floor_ref
        floor_value: float = float(numerator_floor.detach().item())
        # Search order only. The entry that is probed is whichever output
        # weight first clears the roundoff floor. Row 1 is tried first
        # because the embedding shear sits in that chart component.
        row_order: list[int] = [1] + [
            row for row in range(n_out_rows) if row != 1
        ]
        print(
            "Phase 3 probe 3 tape: central difference step "
            f"h={float(h_step.detach().item()):.16e} "
            f"output parameter '{output_weight_name}' "
            f"shape {tuple(output_linear.weight.shape)} "
            f"scan rows {row_order} columns 0..{n_out_cols - 1} "
            f"roundoff floor on |L(w+h)-L(w-h)| = {floor_value:.16e} "
            f"(1000 * eps64 * max(|L|, 1))"
        )

        chosen_row: int = -1
        chosen_col: int = -1
        loss_plus: torch.Tensor = loss_scalar.detach().clone()
        loss_minus: torch.Tensor = loss_scalar.detach().clone()
        d_fd: torch.Tensor = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        numerator: torch.Tensor = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        found_output_entry: bool = False
        try:
            for out_row in row_order:
                for out_col in range(n_out_cols):
                    with torch.no_grad():
                        bumped_up: torch.Tensor = saved_output_weight.clone()
                        bumped_up[out_row, out_col] = (
                            bumped_up[out_row, out_col] + h_step
                        )
                        output_linear.weight.copy_(bumped_up)
                        loss_plus = mean_frobenius_loss()
                        bumped_down: torch.Tensor = saved_output_weight.clone()
                        bumped_down[out_row, out_col] = (
                            bumped_down[out_row, out_col] - h_step
                        )
                        output_linear.weight.copy_(bumped_down)
                        loss_minus = mean_frobenius_loss()
                        output_linear.weight.copy_(saved_output_weight)
                    numerator = (loss_plus - loss_minus).abs()
                    d_fd = (loss_plus - loss_minus) / (two_step * h_step)
                    numerator_value: float = float(numerator.detach().item())
                    d_fd_value: float = float(d_fd.detach().item())
                    if not (numerator_value > floor_value):
                        print(
                            "Phase 3 probe 3 skip roundoff entry "
                            f"'{output_weight_name}'[{out_row}, {out_col}]: "
                            f"L(w+h)={float(loss_plus.detach().item()):.16e} "
                            f"L(w-h)={float(loss_minus.detach().item()):.16e} "
                            f"|L+ - L-|={numerator_value:.16e} "
                            f"d_FD={d_fd_value:.16e} "
                            f"floor={floor_value:.16e}"
                        )
                        continue
                    chosen_row = out_row
                    chosen_col = out_col
                    found_output_entry = True
                    break
                if found_output_entry:
                    break
            if not found_output_entry:
                raise RuntimeError(
                    "Phase 3 probe 3 failed: every output-layer weight "
                    "entry has a finite difference at roundoff "
                    f"(floor {floor_value:.16e} on |L(w+h)-L(w-h)|)"
                )
        finally:
            with torch.no_grad():
                output_linear.weight.copy_(saved_output_weight)

        if not found_output_entry:
            raise RuntimeError(
                "Phase 3 probe 3 failed before an output-layer entry "
                "was selected"
            )
        restored_weight_maxabs: float = float(
            output_linear.weight.detach().abs().max().item()
        )
        restored_bias_maxabs: float = float(
            output_bias.detach().abs().max().item()
        )
        if restored_weight_maxabs != 0.0 or restored_bias_maxabs != 0.0:
            raise RuntimeError(
                "Phase 3 probe 3 failed to restore the zero field: "
                f"weight maxabs {restored_weight_maxabs:.16e} "
                f"bias maxabs {restored_bias_maxabs:.16e}"
            )

        loss_plus_value: float = float(loss_plus.detach().item())
        loss_minus_value: float = float(loss_minus.detach().item())
        d_fd_chosen: float = float(d_fd.detach().item())
        numerator_chosen: float = float(numerator.detach().item())
        print(
            "Phase 3 probe 3 probed output-layer entry "
            f"'{output_weight_name}'[{chosen_row}, {chosen_col}] "
            f"L(w+h)={loss_plus_value:.16e} "
            f"L(w-h)={loss_minus_value:.16e} "
            f"|L+ - L-|={numerator_chosen:.16e} "
            f"d_FD={d_fd_chosen:.16e} "
            f"h={float(h_step.detach().item()):.16e}"
        )

        d_auto: torch.Tensor
        tape_path: str
        try:
            v_eta.zero_grad(set_to_none=True)
            loss_tape: torch.Tensor = mean_frobenius_loss()
            loss_tape.backward()
            grad_w = output_linear.weight.grad
            if grad_w is None:
                raise RuntimeError(
                    "output-layer weight grad is None after backward()"
                )
            if not bool(torch.isfinite(grad_w).all()):
                raise RuntimeError(
                    "non-finite output-layer weight grad after backward()"
                )
            d_auto = grad_w[chosen_row, chosen_col].detach().clone()
            tape_path = "backward"
        except Exception as exc:
            print(
                "WARNING: primary tape path (backward) failed: "
                f"{exc}",
                flush=True,
            )
            print("Falling back to torch.autograd.grad ...", flush=True)
            v_eta.zero_grad(set_to_none=True)
            loss_tape_fallback: torch.Tensor = mean_frobenius_loss()
            grad_fallback: tuple[torch.Tensor, ...] = torch.autograd.grad(
                loss_tape_fallback, output_linear.weight
            )
            grad_w_fallback: torch.Tensor = grad_fallback[0]
            if not bool(torch.isfinite(grad_w_fallback).all()):
                raise RuntimeError(
                    "Fallback autograd.grad failed or produced a "
                    "non-finite gradient"
                )
            d_auto = grad_w_fallback[chosen_row, chosen_col].detach().clone()
            tape_path = "grad"
        finally:
            v_eta.zero_grad(set_to_none=True)

        if d_auto.ndim != 0 or d_auto.dtype != torch.float64:
            raise RuntimeError(
                f"d_auto shape {tuple(d_auto.shape)} dtype {d_auto.dtype} "
                "is not a float64 scalar"
            )
        d_auto_value: float = float(d_auto.detach().item())
        rel_gap: float = abs(d_auto_value - d_fd_chosen) / abs(d_fd_chosen)
        print(
            "Phase 3 probe 3 autograd: "
            f"path={tape_path} "
            f"entry '{output_weight_name}'[{chosen_row}, {chosen_col}] "
            f"d_auto={d_auto_value:.16e} "
            f"d_FD={d_fd_chosen:.16e} "
            f"|d_auto - d_FD| / |d_FD| = {rel_gap:.16e} "
            f"gate < {float(FD_REL_ATOL):.16e}"
        )
        print(
            "Phase 3 probe 3 zero field restored: "
            f"output weight maxabs {restored_weight_maxabs:.16e} "
            f"output bias maxabs {restored_bias_maxabs:.16e}"
        )
        if not (rel_gap < float(FD_REL_ATOL)):
            raise RuntimeError(
                "Phase 3 probe 3 failed: "
                f"entry '{output_weight_name}'[{chosen_row}, {chosen_col}] "
                f"d_auto={d_auto_value:.16e} "
                f"d_FD={d_fd_chosen:.16e} "
                f"rel {rel_gap:.16e} "
                f">= {float(FD_REL_ATOL):.16e}"
            )

        # Probe 4. Zero field, r from §5.
        with torch.no_grad():
            u_closure: torch.Tensor = phi_eta_inverse(
                theta_grid, dt
            ).contiguous()
            r_closure: torch.Tensor
            radius: torch.Tensor
            max_miss: torch.Tensor
            r_closure, radius, max_miss = relative_inverse_closure(
                u_closure, dt
            )
        r_value: float = float(r_closure.detach().item())
        radius_value: float = float(radius.detach().item())
        max_miss_value: float = float(max_miss.detach().item())
        closure_weight_maxabs: float = float(
            output_linear.weight.detach().abs().max().item()
        )
        closure_bias_maxabs: float = float(
            output_bias.detach().abs().max().item()
        )
        print(
            "Phase 3 probe 4 inverse closure: "
            f"r={r_value:.16e} "
            f"R={radius_value:.16e} "
            f"max ||phi_inv(phi(u)) - u||_2 = {max_miss_value:.16e} "
            f"gate r <= {float(R_ZERO_MAX):.16e} "
            f"output weight maxabs {closure_weight_maxabs:.16e} "
            f"output bias maxabs {closure_bias_maxabs:.16e}"
        )
        if closure_weight_maxabs != 0.0 or closure_bias_maxabs != 0.0:
            raise RuntimeError(
                "Phase 3 probe 4 expected the zero field, "
                f"weight maxabs {closure_weight_maxabs:.16e} "
                f"bias maxabs {closure_bias_maxabs:.16e}"
            )
        if not bool(torch.isfinite(r_closure).all()):
            raise RuntimeError(
                f"Phase 3 probe 4 failed: r is not finite ({r_value:.16e})"
            )
        if not (r_value <= float(R_ZERO_MAX)):
            raise RuntimeError(
                "Phase 3 probe 4 failed: zero-field relative inverse "
                f"closure r={r_value:.16e} "
                f"> {float(R_ZERO_MAX):.16e} "
                f"(R={radius_value:.16e}, "
                f"max miss={max_miss_value:.16e})"
            )

        print(
            "Phase 3 probes finished before any optimizer step. "
            f"loss={loss_value:.16e} "
            f"max_rho={max_rho:.16e} "
            f"probe1_max_abs={g_abs_max:.16e} "
            f"probe2_max_abs={batch_abs_max:.16e} "
            f"probe3_entry={output_weight_name}[{chosen_row}, {chosen_col}] "
            f"probe3_rel={rel_gap:.16e} "
            f"probe4_r={r_value:.16e}"
        )

        # === TASK43 PHASE4 TRAIN ===
        # AdamW on the Phase 3 mean Frobenius loss. The Phase 2 step
        # dt is the one already in scope. The only writes to dt are
        # the every-10-step closure check below, after optimizer.step,
        # with gradients off. The differentiated forward does not
        # assign dt. No reseed, no second network, no second RK4.

        if n_points != 125 or tuple(theta_grid.shape) != (125, 3):
            raise RuntimeError(
                "Phase 4 expected the full 5^3 grid, shape (125, 3), "
                f"got {tuple(theta_grid.shape)} n_points={n_points}"
            )
        if device.type != "cuda" or theta_grid.device != device:
            raise RuntimeError(
                f"Phase 4 is one GPU, cuda:0; device={device} "
                f"theta_grid={theta_grid.device}"
            )
        if dt.dtype != torch.float64:
            raise RuntimeError(f"Phase 4 dt dtype {dt.dtype} != float64")

        # mean_frobenius_loss discards g. Record that same tensor so the
        # logged training max rho is the forward AdamW differentiates,
        # not a second integration. The closure check never reads it.
        phase3_chart_state = chart_state_on_grid
        training_g_holder: dict[str, torch.Tensor] = {}

        def chart_state_on_grid() -> tuple[torch.Tensor, torch.Tensor]:
            g_cloud, u_cloud = phase3_chart_state()
            training_g_holder["g"] = g_cloud
            return g_cloud, u_cloud

        def describe_dt(dt_step: torch.Tensor) -> str:
            n_rk4: int = rk4_step_count(dt_step)
            return f"1/{n_rk4} ({float(dt_step.item()):.16e})"

        def integrate_check_cloud() -> tuple[bool, float, float, float]:
            # Fresh u = phi_eta^{-1}(theta) at the current dt, then g of
            # that cloud. Does not assign dt and does not read the
            # training forward's u. Gradients are off.
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "closure integration ran with gradients on"
                )
            u_cloud: torch.Tensor = phi_eta_inverse(
                theta_grid, dt
            ).contiguous()
            if u_cloud.shape != theta_grid.shape:
                raise RuntimeError(
                    f"check cloud shape {tuple(u_cloud.shape)} != "
                    f"grid {tuple(theta_grid.shape)}"
                )
            if u_cloud.dtype != torch.float64:
                raise RuntimeError(
                    f"check cloud dtype {u_cloud.dtype} != float64"
                )
            if int(u_cloud.shape[0]) != n_points:
                raise RuntimeError(
                    f"check cloud has {int(u_cloud.shape[0])} points, "
                    f"grid has {n_points}"
                )
            g_cloud: torch.Tensor = induced_g_batch(u_cloud)
            if g_cloud.dtype != torch.float64:
                raise RuntimeError(
                    f"check g dtype {g_cloud.dtype} != float64"
                )
            rho_cloud: torch.Tensor = rho_from_g(g_cloud)
            max_rho_now: float = float(rho_cloud.max().item())
            if not np.isfinite(max_rho_now):
                raise RuntimeError(
                    f"check max rho is not finite ({max_rho_now})"
                )
            try:
                r_closure, radius, _max_miss = relative_inverse_closure(
                    u_cloud, dt
                )
            except RuntimeError as exc:
                if "closure radius R is 0" in str(exc):
                    return (False, float("nan"), max_rho_now, 0.0)
                raise
            r_now: float = float(r_closure.item())
            radius_now: float = float(radius.item())
            if not np.isfinite(r_now) or not np.isfinite(radius_now):
                raise RuntimeError(
                    f"check closure is not finite r={r_now} R={radius_now}"
                )
            return (True, r_now, max_rho_now, radius_now)

        optimizer: torch.optim.AdamW = torch.optim.AdamW(
            v_eta.parameters(),
            lr=float(LR),
            weight_decay=float(WEIGHT_DECAY),
        )
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(FACTOR_SCHEDULER),
                patience=int(PATIENCE_SCHEDULER),
                cooldown=int(COOLDOWN_SCHEDULER),
                threshold=float(THRESHOLD_SCHEDULER),
                threshold_mode="rel",
                min_lr=float(LR_MIN),
            )
        )
        opt_lr: float = float(optimizer.param_groups[0]["lr"])
        opt_wd: float = float(optimizer.param_groups[0]["weight_decay"])
        if opt_lr != float(LR) or opt_wd != float(WEIGHT_DECAY):
            raise RuntimeError(
                f"AdamW lr={opt_lr} weight_decay={opt_wd} "
                f"!= {float(LR)} and {float(WEIGHT_DECAY)}"
            )
        dt_min_tensor: torch.Tensor = torch.tensor(
            np.float64(DT_MIN), dtype=torch.float64, device=device
        )
        dt_init_tensor: torch.Tensor = torch.tensor(
            np.float64(DT_INIT), dtype=torch.float64, device=device
        )
        n_dt_min: int = rk4_step_count(dt_min_tensor)
        n_dt_init: int = rk4_step_count(dt_init_tensor)
        if rk4_step_count(dt) != n_dt_init:
            raise RuntimeError(
                f"Phase 4 entered with dt={describe_dt(dt)}, "
                f"expected 1/{n_dt_init}"
            )
        print(
            "Phase 4 AdamW: "
            f"lr={opt_lr:.16e} weight_decay={opt_wd:.16e} "
            f"plateau factor={float(FACTOR_SCHEDULER):.16e} "
            f"patience={int(PATIENCE_SCHEDULER)} "
            f"cooldown={int(COOLDOWN_SCHEDULER)} "
            f"threshold={float(THRESHOLD_SCHEDULER):.16e} rel "
            f"min_lr={float(LR_MIN):.16e} "
            f"grid={tuple(theta_grid.shape)} "
            f"dt={describe_dt(dt)} "
            f"check_every={int(HALVE_EVERY_STEPS)} "
            f"max_steps={int(MAX_OPT_STEPS)} "
            f"dt_min=1/{n_dt_min} "
            f"r_bar={float(R_MAX):.16e} "
            f"rho_pass={float(RHO_PASS):.16e} "
            f"rho_progress={float(RHO_MARK):.16e} "
            "cuda:0, one process, full grid every step, "
            "no reseed, no rebuild of v_eta"
        )

        stop_reason: str = ""
        reported_r: float = float("nan")
        reported_max_rho: float = float("nan")
        have_reported_r: bool = False
        opt_step: int = 0
        r_bar: float = float(R_MAX)
        rho_pass: float = float(RHO_PASS)
        rho_mark: float = float(RHO_MARK)

        for opt_step in range(1, int(MAX_OPT_STEPS) + 1):
            if not torch.is_grad_enabled():
                raise RuntimeError(
                    "differentiated forward started with gradients off"
                )
            n_dt_forward: int = rk4_step_count(dt)
            dt_id_forward: int = id(dt)
            optimizer.zero_grad(set_to_none=True)
            training_g_holder.clear()
            loss: torch.Tensor = mean_frobenius_loss()
            if "g" not in training_g_holder:
                raise RuntimeError(
                    "training forward did not record g^(u) from the "
                    "Phase 3 chart state"
                )
            if loss.ndim != 0:
                raise RuntimeError(
                    f"training loss ndim {loss.ndim} != 0"
                )
            if loss.dtype != torch.float64:
                raise RuntimeError(
                    f"training loss dtype {loss.dtype} != float64"
                )
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError("training loss is not finite")
            loss_value: float = float(loss.item())
            g_train: torch.Tensor = training_g_holder["g"]
            if g_train.dtype != torch.float64:
                raise RuntimeError(
                    f"training g dtype {g_train.dtype} != float64"
                )
            max_rho_train: float = float(rho_from_g(g_train).max().item())
            training_g_holder.clear()
            if not np.isfinite(max_rho_train):
                raise RuntimeError(
                    f"training max rho is not finite ({max_rho_train})"
                )
            loss.backward()
            if id(dt) != dt_id_forward or rk4_step_count(dt) != n_dt_forward:
                raise RuntimeError(
                    "dt changed inside the differentiated forward"
                )
            saw_grad: bool = False
            for param in v_eta.parameters():
                grad = param.grad
                if grad is None:
                    continue
                saw_grad = True
                if grad.dtype != torch.float64:
                    raise RuntimeError(
                        f"gradient dtype {grad.dtype} != float64"
                    )
                if not bool(torch.isfinite(grad).all()):
                    raise RuntimeError(
                        "non-finite v_eta gradient; refusing the AdamW step"
                    )
            if not saw_grad:
                raise RuntimeError(
                    "mean Frobenius loss backward produced no v_eta gradient"
                )
            optimizer.step()
            lr_before: float = float(optimizer.param_groups[0]["lr"])
            scheduler.step(max_rho_train)
            lr_after: float = float(optimizer.param_groups[0]["lr"])
            if lr_after < lr_before:
                print(
                    "Phase 4 ReduceLROnPlateau: "
                    f"step={opt_step} "
                    f"lr {lr_before:.16e} -> {lr_after:.16e} "
                    f"max_rho={max_rho_train:.16e}"
                )
            del loss
            del g_train
            below_mark: bool = max_rho_train < rho_mark
            progress_note: str = ""
            if below_mark:
                progress_note = (
                    " progress_only max rho < "
                    f"{rho_mark:.16e} is not a pass and not a stop"
                )
            print(
                f"Phase 4 step {opt_step}: "
                f"mean_loss={loss_value:.16e} "
                f"max_rho={max_rho_train:.16e} "
                f"lr={lr_before:.16e} "
                f"dt={describe_dt(dt)}"
                f"{progress_note}"
            )

            is_check_step: bool = (
                (opt_step % int(HALVE_EVERY_STEPS) == 0)
                or (opt_step == int(MAX_OPT_STEPS))
            )
            if is_check_step:
                with torch.no_grad():
                    r_defined, r_now, max_rho_now, radius_now = (
                        integrate_check_cloud()
                    )
                    if not r_defined:
                        stop_reason = "R=0"
                        reported_max_rho = max_rho_now
                        print(
                            "Phase 4 check R=0: "
                            f"step={opt_step} dt={describe_dt(dt)} "
                            f"max_rho={max_rho_now:.16e} r=undefined"
                        )
                    else:
                        halve_count: int = 0
                        while (
                            r_now > r_bar
                            and rk4_step_count(dt) < n_dt_min
                        ):
                            halve_count = halve_count + 1
                            if halve_count > 8:
                                raise RuntimeError(
                                    "halving did not reach the closure "
                                    "bar or dt=1/512"
                                )
                            print(
                                "Phase 4 halve before: "
                                f"step={opt_step} dt={describe_dt(dt)} "
                                f"r={r_now:.16e} "
                                f"max_rho={max_rho_now:.16e}"
                            )
                            n_curr: int = rk4_step_count(dt)
                            n_next: int = n_curr * 2
                            if n_next > n_dt_min:
                                raise RuntimeError(
                                    "refusing to halve past 1/512 "
                                    f"(next step count {n_next})"
                                )
                            dt = torch.tensor(
                                np.float64(1.0) / np.float64(n_next),
                                dtype=torch.float64,
                                device=device,
                            )
                            if rk4_step_count(dt) != n_next:
                                raise RuntimeError(
                                    "halved dt is not 1 over the new "
                                    f"step count {n_next}"
                                )
                            (
                                r_defined,
                                r_now,
                                max_rho_now,
                                radius_now,
                            ) = integrate_check_cloud()
                            if not r_defined:
                                stop_reason = "R=0"
                                reported_max_rho = max_rho_now
                                print(
                                    "Phase 4 check R=0: "
                                    f"step={opt_step} dt={describe_dt(dt)} "
                                    f"max_rho={max_rho_now:.16e} "
                                    "r=undefined"
                                )
                                break
                            print(
                                "Phase 4 halve after: "
                                f"step={opt_step} dt={describe_dt(dt)} "
                                f"r={r_now:.16e} "
                                f"R={radius_now:.16e} "
                                f"max_rho={max_rho_now:.16e}"
                            )
                        if stop_reason != "R=0":
                            reported_r = r_now
                            have_reported_r = True
                            reported_max_rho = max_rho_now
                            if halve_count == 0:
                                print(
                                    "Phase 4 check integration: "
                                    f"step={opt_step} "
                                    f"dt={describe_dt(dt)} "
                                    f"r={r_now:.16e} "
                                    f"R={radius_now:.16e} "
                                    f"max_rho={max_rho_now:.16e}"
                                )
                            if r_now > r_bar:
                                if rk4_step_count(dt) != n_dt_min:
                                    raise RuntimeError(
                                        "closure r is above the bar but "
                                        f"dt={describe_dt(dt)} is not 1/512"
                                    )
                                stop_reason = (
                                    "closure failure at dt=1/512"
                                )
                                print(
                                    "Phase 4 closure failure before "
                                    "the metric stop: "
                                    f"step={opt_step} "
                                    f"dt={describe_dt(dt)} "
                                    f"r={r_now:.16e} "
                                    f"max_rho={max_rho_now:.16e}"
                                )
                            elif (max_rho_now < rho_pass) and (
                                r_now <= r_bar
                            ):
                                stop_reason = "joint pass"
                            else:
                                print(
                                    "Phase 4 check continue: "
                                    f"step={opt_step} "
                                    f"dt={describe_dt(dt)} "
                                    f"r={r_now:.16e} "
                                    f"max_rho={max_rho_now:.16e} "
                                    "bars not both met"
                                )
                if stop_reason != "":
                    break
            if opt_step == int(MAX_OPT_STEPS):
                stop_reason = f"{int(MAX_OPT_STEPS)}-step cap"
                break

        if stop_reason == "":
            raise RuntimeError(
                "training loop ended without a joint pass, "
                "closure failure at dt=1/512, R=0, or the "
                f"{int(MAX_OPT_STEPS)}-step cap"
            )
        if have_reported_r:
            r_text: str = f"{reported_r:.16e}"
        else:
            r_text = "undefined"
        print(f"Phase 4 stop: {stop_reason}")
        print(f"Phase 4 final dt={describe_dt(dt)}")
        print(f"Phase 4 final most_recent_r={r_text}")
        print(
            "Phase 4 final max_rho="
            f"{reported_max_rho:.16e}"
        )
        print(f"Phase 4 stopped_at_step={opt_step}")

    except Exception as exc:
        print(f"\n💥 UNHANDLED EXCEPTION in main: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_handle.close()
        error_file_handle.close()
        print(f"✅ Logs saved to: {out_err_folder}")


if __name__ == "__main__":
    main()
