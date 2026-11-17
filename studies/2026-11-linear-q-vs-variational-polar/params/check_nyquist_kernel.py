"""Does the projection of B0 x u lose the shortest axial wave? For smooth velocity 2-forms
u = g(eta1) cos(2 pi (eta2 + n eta3)) e_c (one component c, m = 1, n = 1 .. N3/2) on the kink case, the norm of the
magnetic-field increment b = -curl P(B0 x u), relative to the norm of u (both in the M2 norm), with
- P = Pi^1 (commuting projector, q model: curl T2 u),
- P = L2 projection (variational model with transport_projection="L2": T0 u).

Run: mpirun -n 4 python check_nyquist_kernel.py [--nz 12]  (table also in nyquist_kernel_z<nz>.md)
"""

import argparse
import os

import numpy as np
from maybempi import MPI

from cases import domain_and_equil, make_model

ap = argparse.ArgumentParser()
ap.add_argument("--nz", type=int, default=12)
a = ap.parse_args()
rank = MPI.COMM_WORLD.Get_rank()

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, grids  # noqa: E402

ne = (16, 12, a.nz)
model = make_model("q")
domain, equil = domain_and_equil("kink", ne)
sim = Simulation(
    model=model,
    env=EnvironmentOptions(out_folders=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sims"),
                           sim_folder="nyq_kernel"),
    time_opts=Time(dt=0.5, Tend=0.5, split_algo="Strang"),
    domain=domain,
    equil=equil,
    grid=grids.TensorProductGrid(num_elements=ne),
    derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=(("free", "dirichlet"), None, None), polar_splines=True),
)
sim.allocate()
from struphy.feec.variational_utilities import L2TransportOperator  # noqa: E402
from struphy.propagators.base import Propagator  # noqa: E402

derham, mass_ops = Propagator.derham, Propagator.mass_ops
M2 = mass_ops.M2
T2 = Propagator.basis_ops.T2
curl = derham.curl
T0 = L2TransportOperator(derham, Propagator.domain, mass_ops, assemble=True)
T0.update_coeffs(Propagator.projected_equil.b2)
bop = derham.boundary_ops["2"]


def norm(v):
    return np.sqrt(M2.dot(v).inner(v))


def wave(c, n):
    u = derham.V2pol.zeros()
    t = u.tp.blocks[c]
    nr = u.space.n_rings[c]
    N = t.space.npts
    s = (np.arange(N[0]) + 0.5) / N[0]
    full = (np.sin(np.pi * s) ** 2)[:, None, None] * np.cos(
        2 * np.pi * (np.arange(N[1])[None, :, None] / N[1] + n * np.arange(N[2])[None, None, :] / N[2])
    )
    full[:nr] = 0.0
    sl = tuple(slice(st, en + 1) for st, en in zip(t.starts, t.ends))
    t[sl] = full[sl]
    u = bop.dot(u)
    u.update_ghost_regions()
    return u


rows = []
for c, name in ((0, "u^1 (radial flux)"), (1, "u^2 (poloidal flux)"), (2, "u^3 (axial flux)")):
    for n in range(1, a.nz // 2 + 1):
        u = wave(c, n)
        nu = norm(u)
        bq = curl.dot(T2.dot(u))
        bv = T0.dot(u)
        rows.append((name, n, norm(bq) / nu, norm(bv) / nu))
if rank == 0:
    with open(os.path.join(os.environ.get("STUDY_RUNS", os.path.dirname(os.path.abspath(__file__))),
                           f"nyquist_kernel_z{a.nz}.md"), "w") as f:
        f.write("| component | n | Pi^1 (q) | L2 (var) |\n|---|---|---|---|\n")
        f.writelines(f"| {name} | {n} | {q:.3e} | {v:.3e} |\n" for name, n, q, v in rows)
    print(f"N3 = {a.nz}: |b| / |u| (M2 norms), b = -curl P(B0 x u), u = smooth m = 1 wave in one component")
    print(f"{'component':22s} {'n':>3s} {'Pi^1 (q)':>11s} {'L2 (var)':>11s}")
    for name, n, q, v in rows:
        print(f"{name:22s} {n:3d} {q:11.3e} {v:11.3e}")
