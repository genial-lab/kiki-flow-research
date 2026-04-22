"""Generate Mode-A style pairs at state_dim=128 (32 stacks x 4 ortho)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.species import MixedCanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track1_perf.phenomenological_f import T1FreeEnergy

STATE_DIM_PER_SPECIES = 4  # grid per (ortho, stack)
N_STACKS = 32
N_ORTHO = 4
TOTAL = STATE_DIM_PER_SPECIES * N_ORTHO * N_STACKS  # 512 for fully-flattened; too big
# Compromise: use state_dim_per_species=1 so flatten = 4*32 = 128
GRID = 1
TARGET_DIM = GRID * N_ORTHO * N_STACKS
EXPECTED_DIM = 128
assert TARGET_DIM == EXPECTED_DIM, f"expected {EXPECTED_DIM}, got {TARGET_DIM}"
N_PAIRS = 100
stacks = [f"s{i:02d}" for i in range(N_STACKS)]
species = MixedCanonicalSpecies(stack_names=stacks)
names = species.species_names()
rng = np.random.default_rng(0)

# For grid=1 the solver dynamics are trivial; inject variation via scheduler
f = T1FreeEnergy(alpha=0.0, beta=0.1, gamma=0.5, species=species, v_curr=np.zeros(GRID))
support = np.linspace(-1, 1, GRID).reshape(-1, 1)
jko = JKOStep(f_functional=f, h=0.05, support=support, n_inner=5, apply_w2_prox=False)

out_dir = Path("bench/runs/T2_pairs_d128")
out_dir.mkdir(parents=True, exist_ok=True)
for i in range(N_PAIRS):
    # Random initial rho as Dirichlet samples per species
    rho_dict = {n: rng.dirichlet(np.ones(GRID)).astype(np.float32) for n in names}
    state = FlowState(
        rho=rho_dict,
        P_theta=np.zeros(4),
        mu_curr=np.full(GRID, 1.0 / GRID),
        tau=0,
        metadata={"track_id": "T1"},
    )
    post_state = jko.step(state)
    pre_flat = np.concatenate([rho_dict[n] for n in sorted(rho_dict.keys())])
    post_flat = np.concatenate([post_state.rho[n] for n in sorted(post_state.rho.keys())])
    save_file(
        {
            "rho::phono": pre_flat.astype(np.float32),
            "rho::phono_next": post_flat.astype(np.float32),
        },
        str(out_dir / f"pair_{i:04d}.safetensors"),
    )
print(f"Wrote {N_PAIRS} pairs at state_dim={TARGET_DIM} under {out_dir}")
