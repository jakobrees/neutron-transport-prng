"""
Monte Carlo Neutron Transport Simulation
Godiva Benchmark: Bare HEU Sphere (HEU-MET-FAST-001)

Reference geometry: Frankle, S.C., LA-13594, LANL, 1999.
    - Radius:       8.7407 cm
    - Atom density: 4.7984e-2 atoms/(barn·cm)
    - Composition:  93.7% U-235, 5.2% U-238, 1.0% U-234 (atom fraction)
    - Benchmark k:  1.000 ± 0.001

Physics reference: Lux & Koblinger, "Monte Carlo Particle Transport Methods", 1991.
RNG reference:     Hayes, B., "The Middle of the Square", bit-player.org, 2022.

Physics model:
    - Multi-speed transport: cross sections recomputed after each scatter
    - Isotope sampling: weighted by n_i * sigma_t^(i)
    - Fission neutron energies sampled from Watt spectrum (U-235 parameters)
    - Inelastic scatter treated as elastic (neutron survives)
    - Elastic scatter: isotropic in CM frame -> uniform energy loss on [alpha*E, E]
"""

import numpy as np
import matplotlib.pyplot as plt
from cross_sections import load_all_cross_sections

# ================================================================== #
#  CONSTANTS AND GEOMETRY                                              #
# ================================================================== #

RADIUS       = 8.7407       # cm, Godiva critical radius (Frankle 1999)
ATOM_DENSITY = 4.7984e-2    # atoms/(barn·cm), total number density

# Atom fractions (Frankle 1999, LA-13594)
ATOM_FRACTIONS = {
    "U-235": 0.9377,
    "U-238": 0.0520,
    "U-234": 0.0103,
}

# Number densities: n_i = f_i * N_total  [atoms/(barn·cm)]
NUMBER_DENSITIES = {iso: f * ATOM_DENSITY
                    for iso, f in ATOM_FRACTIONS.items()}

# Mass numbers for scattering kinematics
MASS_NUMBERS = {"U-234": 234, "U-235": 235, "U-238": 238}

# Watt fission spectrum parameters for U-235 (Lux & Koblinger 1991)
# p(E) ~ exp(-E/a) * sinh(sqrt(b*E))
WATT_A = 0.988e6   # eV
WATT_B = 2.249e-6  # 1/eV

# Mean fission neutron yield U-235 at ~2 MeV
NU_BAR = 2.58

# Energy bounds for cross section interpolation [eV]
E_MIN = 1e-4
E_MAX = 2e7


# ================================================================== #
#  RANDOM NUMBER GENERATORS                                            #
# ================================================================== #

class MiddleSquare:
    """
    Von Neumann middle-square method, 8 decimal digits.
    Historically authentic to the ENIAC runs of spring 1948.

    Recurrence: x_{n+1} = (x_n^2 mod 10^12) // 10^4   (extract middle 8 digits)
    Median cycle length ~2700 (Hayes 2022).

    HISTORICAL USE: Hayes 2022 documents that the 1948 ENIAC simulations did
    not iterate the recurrence indefinitely — degenerate seeds and short
    cycles made that impractical.  Instead, a fixed block of ~2000 numbers
    was pre-generated from one seed and reused cyclically across the
    simulation.  We replicate that scheme here: pre-generate `block_size`
    numbers at construction time, then wrap (modular index) on each call.

    Block reuse means the SAME ~2000 numbers are seen many times within a
    cycle of the power iteration.  This is the key feature distinguishing
    middle-square from the modern PRNGs in this study and is the most
    plausible source of any elevated variance or systematic bias.

    Reference: Hayes, B. "The Middle of the Square", bit-player.org, 2022.
    """
    def __init__(self, seed=46831729, block_size=2000):
        assert 10_000_000 <= seed <= 99_999_999, "Seed must be exactly 8 digits"
        self.width      = 8
        self.modulus    = 10 ** (3 * self.width // 2)   # 10^12
        self.divisor    = 10 ** (self.width // 2)       # 10^4
        self._initial   = seed
        self.block_size = block_size

        # Pre-generate and validate the block at construction.
        self._block = self._generate_block(seed, block_size)
        self._pos   = 0

    def _generate_block(self, seed, n):
        """
        Iterate the recurrence n times from `seed` and return the normalized
        outputs.  Raises ValueError if the recurrence hits zero before n —
        the caller (rng_std_comparison) traps this and skips the seed.
        """
        block = []
        s = seed
        for i in range(n):
            s = (s ** 2 % self.modulus) // self.divisor
            if s == 0:
                raise ValueError(
                    f"Seed {seed} degenerates at step {i+1} — choose a different seed."
                )
            block.append(s / 1e8)
        return block

    def random(self):
        """Return next number in the pre-generated block (cyclic)."""
        val = self._block[self._pos]
        self._pos = (self._pos + 1) % self.block_size
        return val

    def detect_cycle(self, max_iter=500_000):
        """
        Returns cycle length of the underlying recurrence (independent of
        the block reuse) starting from the initial seed, or None if not
        found within max_iter steps.  Useful for the writeup statistic.
        """
        seen = {}
        s = self._initial
        for i in range(max_iter):
            if s in seen:
                return i - seen[s]
            seen[s] = i
            s = (s ** 2 % self.modulus) // self.divisor
            if s == 0:
                return i
        return None


class LCG:
    """
    Linear Congruential Generator.
    x_{n+1} = (a * x_n + c) mod m

    Parameters from Numerical Recipes satisfy Hull-Dobell full-period theorem:
        1. c and m coprime
        2. a-1 divisible by all prime factors of m
        3. a-1 divisible by 4 (since 4 | m)
    """
    def __init__(self, seed=12345, a=1664525, c=1013904223, m=2**32):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m

    def random(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m


class MersenneTwister:
    """Thin wrapper around Python stdlib random for uniform interface."""
    def __init__(self, seed=12345):
        import random
        self._rng = random.Random(seed)

    def random(self):
        return self._rng.random()


# ================================================================== #
#  CROSS SECTION SETUP                                                 #
# ================================================================== #

def build_sigma_tables(xs_data, E_eV):
    """
    Microscopic cross sections at energy E_eV for each isotope.
    Inelastic = total - elastic - fission - capture (residual, neutron survives).

    Returns dict:
        sigma[isotope] = {
            'total', 'elastic', 'fission', 'capture', 'inelastic'
        }  all in barns
    """
    E_eV  = float(np.clip(E_eV, E_MIN, E_MAX))
    sigma = {}

    for iso in ["U-234", "U-235", "U-238"]:
        xs  = xs_data[iso]
        tot = float(xs['total'   ]['interpolate'](E_eV)[0])
        el  = float(xs['elastic' ]['interpolate'](E_eV)[0])
        fis = float(xs['fission' ]['interpolate'](E_eV)[0])
        cap = float(xs['capture' ]['interpolate'](E_eV)[0])
        inel = max(0.0, tot - el - fis - cap)

        sigma[iso] = {
            'total':     tot,
            'elastic':   el,
            'fission':   fis,
            'capture':   cap,
            'inelastic': inel,
        }

    return sigma


def build_macro_sigma(sigma, number_densities=NUMBER_DENSITIES):
    """
    Macroscopic cross sections: Sigma_x = sum_i n_i * sigma_x^(i)
    Units: cm^-1
    """
    macro = {rxn: 0.0
             for rxn in ['total', 'elastic', 'fission', 'capture', 'inelastic']}
    for iso, n_i in number_densities.items():
        for rxn in macro:
            macro[rxn] += n_i * sigma[iso][rxn]
    return macro


# ================================================================== #
#  SAMPLING PRIMITIVES                                                 #
# ================================================================== #

def sample_path_length(sigma_t, rng):
    """
    d ~ Exponential(sigma_t).  Inverse CDF: d = -ln(U) / sigma_t.
    Mean free path = 1 / sigma_t.
    """
    return -np.log(rng.random()) / sigma_t


def sample_direction(rng):
    """
    Isotropic direction on unit sphere.
    mu = cos(theta) ~ Uniform(-1,1),  phi ~ Uniform(0, 2*pi).
    Returns unit vector (3,).
    """
    mu  = 2.0 * rng.random() - 1.0
    phi = 2.0 * np.pi * rng.random()
    s   = np.sqrt(1.0 - mu**2)
    return np.array([s * np.cos(phi), s * np.sin(phi), mu])


def sample_isotope(sigma, rng, number_densities=NUMBER_DENSITIES):
    """
    Which isotope was hit, weighted by n_i * sigma_t^(i).

    Correct weight: contribution to total macroscopic cross section.
    A rare isotope with large sigma is hit more than its atom fraction suggests.
    Returns isotope string: 'U-234', 'U-235', or 'U-238'.
    """
    weights = {iso: number_densities[iso] * sigma[iso]['total']
               for iso in sigma}
    total  = sum(weights.values())
    xi     = rng.random() * total
    cumsum = 0.0
    for iso, w in weights.items():
        cumsum += w
        if xi < cumsum:
            return iso
    return list(sigma.keys())[-1]


def sample_reaction_type_for_isotope(iso_sigma, rng):
    """
    Reaction type given a specific isotope was hit.

    Hierarchy (Lux & Koblinger 1991, Ch.2):
        fission | scatter (elastic + inelastic, neutron survives) | capture

    Inelastic scatter treated as elastic: neutron survives with
    approximate energy loss. Justified in multi-speed model since
    cross sections are recomputed at new energy after each scatter.
    """
    xi = rng.random() * iso_sigma['total']

    if xi < iso_sigma['fission']:
        return 'fission'
    elif xi < (iso_sigma['fission']
               + iso_sigma['elastic']
               + iso_sigma['inelastic']):
        return 'scatter'
    else:
        return 'capture'


def sample_watt_energy(rng, a=WATT_A, b=WATT_B):
    """
    Fission neutron energy from Watt spectrum:
        p(E) ~ exp(-E/a) * sinh(sqrt(b*E))

    Maxwell-with-correction method (used in MCNP and OpenMC):
        1. Sample w from Maxwell(a)
        2. E = w + a^2 b / 4 + (2 xi - 1) * sqrt(a^2 b w)

    Step 2 exactly recovers Watt because Watt(a,b) is the distribution of
    sqrt(E)^2 when sqrt(E) is normal with mean a*sqrt(b)/2 and variance a/2 —
    equivalently, a Maxwell convolved with a uniform shift on sqrt(E).
    Maxwell sampling: w = -a*(ln xi_1 + cos^2(pi xi_3 / 2) * ln xi_2).

    U-235 parameters: a=0.988 MeV, b=2.249 MeV^-1, stored in eV units.

    Reference: Forrest B. Brown, "Fundamentals of Monte Carlo Particle
    Transport", LA-UR-05-4983, Los Alamos National Laboratory (2005),
    section on sampling the Watt fission spectrum.
    """
    # Maxwell sample (no rejection)
    r1 = rng.random()
    r2 = rng.random()
    r3 = rng.random()
    c  = np.cos(0.5 * np.pi * r3)
    w  = -a * (np.log(r1) + np.log(r2) * c * c)

    # Watt correction
    E = w + 0.25 * a * a * b \
        + (2.0 * rng.random() - 1.0) * np.sqrt(a * a * b * w)

    return max(E, E_MIN)


def sample_nu(nu_bar, rng):
    """
    Integer fission neutron yield from mean nu_bar.
    floor(nu_bar) + Bernoulli(fractional part).
    """
    floor = int(nu_bar)
    return floor + (1 if rng.random() < (nu_bar - floor) else 0)


def sample_scatter_energy_direction(E, omega, A, rng):
    """
    Elastic scatter: isotropic in CM frame, transformed to lab frame.
    Post-scatter energy uniform on [alpha*E, E] (Lux & Koblinger 1991).

    alpha = ((A-1)/(A+1))^2 — fraction of energy retained at 180 deg.
    For U-235: alpha ~ 0.983 (barely slows down).
    For H-1:   alpha = 0    (full energy transfer possible).

    Returns: (E_new [eV], omega_new [unit vector])
    """
    mu0 = 2.0 * rng.random() - 1.0
    phi = 2.0 * np.pi * rng.random()

    alpha = ((A - 1.0) / (A + 1.0)) ** 2
    E_new = E * (1.0 + alpha + (1.0 - alpha) * mu0) / 2.0
    E_new = max(E_new, E_MIN)

    sin_theta = np.sqrt(max(0.0, 1.0 - mu0**2))

    if abs(omega[2]) < 0.9999:
        denom = np.sqrt(1.0 - omega[2]**2)
        ox = (sin_theta * (omega[0]*omega[2]*np.cos(phi)
              - omega[1]*np.sin(phi)) / denom + omega[0] * mu0)
        oy = (sin_theta * (omega[1]*omega[2]*np.cos(phi)
              + omega[0]*np.sin(phi)) / denom + omega[1] * mu0)
        oz = (-sin_theta * denom * np.cos(phi) + omega[2] * mu0)
    else:
        sign = np.sign(omega[2])
        ox   = sin_theta * np.cos(phi)
        oy   = sin_theta * np.sin(phi)
        oz   = sign * mu0

    omega_new = np.array([ox, oy, oz])
    norm = np.linalg.norm(omega_new)
    if norm > 0:
        omega_new /= norm

    return E_new, omega_new


# ================================================================== #
#  GEOMETRY                                                            #
# ================================================================== #

def distance_to_boundary(r, omega, radius=RADIUS):
    """
    Distance from r in direction omega to sphere surface.
    Solves |r + t*omega|^2 = R^2:
        t^2 + 2(r.omega)t + (|r|^2 - R^2) = 0
    Returns positive root.
    """
    b            = np.dot(r, omega)
    c            = np.dot(r, r) - radius**2
    discriminant = b**2 - c
    if discriminant < 0:
        return 0.0
    return -b + np.sqrt(discriminant)


# ================================================================== #
#  NEUTRON HISTORY                                                     #
# ================================================================== #

def simulate_neutron(r0, omega0, E0, xs_data, rng, nu_bar=NU_BAR):
    """
    Track a single neutron from birth to death (multi-speed).

    Sampling hierarchy at each collision (Lux & Koblinger 1991, Ch.2):
        1. Path length from Exponential(Sigma_t(E))
        2. Geometry check: escape if d_collision >= d_boundary
        3. Isotope: P(iso_i) = n_i*sigma_t^(i) / Sigma_t  [weighted by xs contribution]
        4. Reaction from that isotope's partial cross sections
        5. Scatter: update E and omega, recompute xs at new E
           Fission: spawn nu neutrons with Watt energies, terminate history

    Returns:
        fission_sites: list of (r, omega, E) for each fission-born neutron
        leaked:        bool — neutron escaped sphere
        absorbed:      bool — neutron captured
    """
    r     = r0.copy()
    omega = omega0.copy()
    E     = E0

    sigma = build_sigma_tables(xs_data, E_eV=E)
    macro = build_macro_sigma(sigma)

    fission_sites = []

    while True:
        # step 1: path length vs boundary
        d_collision = sample_path_length(macro['total'], rng)
        d_boundary  = distance_to_boundary(r, omega)

        if d_collision >= d_boundary:
            return fission_sites, True, False   # leaked

        # step 2: move to collision site
        r = r + d_collision * omega

        # step 3: which isotope
        hit_iso = sample_isotope(sigma, rng)
        A       = MASS_NUMBERS[hit_iso]

        # step 4: which reaction (for that isotope)
        reaction = sample_reaction_type_for_isotope(sigma[hit_iso], rng)

        if reaction == 'capture':
            return fission_sites, False, True   # absorbed

        elif reaction == 'scatter':
            E, omega = sample_scatter_energy_direction(E, omega, A, rng)
            # multi-speed: recompute cross sections at new energy
            sigma = build_sigma_tables(xs_data, E_eV=E)
            macro = build_macro_sigma(sigma)

        elif reaction == 'fission':
            nu = sample_nu(nu_bar, rng)
            for _ in range(nu):
                new_omega = sample_direction(rng)
                new_E     = sample_watt_energy(rng)
                fission_sites.append((r.copy(), new_omega, new_E))
            return fission_sites, False, False


# ================================================================== #
#  K-EFFECTIVE ESTIMATOR (POWER ITERATION)                             #
# ================================================================== #

def run_simulation(rng, xs_data,
                   n_histories=1000,
                   n_active_cycles=50,
                   n_inactive_cycles=20,
                   nu_bar=NU_BAR,
                   verbose=True,
                   track_flux=False):
    """
    Power iteration k-effective estimator.

    Inactive cycles: fission source converges to true spatial distribution
    (analogous to MCMC burn-in). Active cycles: collect k estimates.

    k_cycle = fission_neutrons_produced / neutrons_started_this_cycle

    Population renormalized to n_histories each cycle — we estimate
    the ratio k, not a dying/growing population.

    track_flux: if True, also return a list of fission-site positions
    (one (3,) array per fission neutron) accumulated across all active
    cycles.  After burn-in, these positions are samples from the
    fundamental-mode fission source distribution S(r), which (up to a
    nu*Sigma_f factor) is the spatial flux profile.

    Returns: k_mean, k_std (standard error), k_history (active cycles),
             [fission_positions if track_flux].
    """
    # initial source: all at origin, isotropic, Watt energy
    source = []
    for _ in range(n_histories):
        source.append((np.zeros(3), sample_direction(rng), sample_watt_energy(rng)))

    k_history         = []
    fission_positions = []                    # accumulated only during active cycles
    total_cycles      = n_inactive_cycles + n_active_cycles

    for cycle in range(total_cycles):
        new_source = []
        n_produced = 0

        for (r0, omega0, E0) in source:
            try:
                sites, leaked, absorbed = simulate_neutron(
                    r0, omega0, E0, xs_data, rng, nu_bar
                )
            except RuntimeError as e:
                if verbose:
                    print(f"  RNG error: {e}")
                break

            new_source.extend(sites)
            n_produced += len(sites)

        k_cycle   = n_produced / len(source) if len(source) > 0 else 0.0
        is_active = cycle >= n_inactive_cycles

        if verbose:
            tag = (f"  Cycle {cycle - n_inactive_cycles + 1:3d}/{n_active_cycles}"
                   if is_active else
                   f"  [inactive {cycle+1:2d}/{n_inactive_cycles}]")
            print(f"{tag}  k = {k_cycle:.5f}  fission sites = {n_produced}")

        if is_active:
            k_history.append(k_cycle)
            if track_flux:
                # Each entry of new_source is (r, omega, E); we keep r.
                for (r, _, _) in new_source:
                    fission_positions.append(r)

        if len(new_source) == 0:
            if verbose:
                print("  WARNING: fission source died out.")
            break

        # renormalize source back to n_histories
        n_new  = len(new_source)
        idx    = [int(rng.random() * n_new) for _ in range(n_histories)]
        source = [new_source[i] for i in idx]

    if not k_history:
        return (0.0, 0.0, []) if not track_flux else (0.0, 0.0, [], [])

    k_arr  = np.array(k_history)
    k_mean = float(np.mean(k_arr))
    k_std  = float(np.std(k_arr) / np.sqrt(len(k_arr)))

    if track_flux:
        return k_mean, k_std, k_history, fission_positions
    return k_mean, k_std, k_history


# ================================================================== #
#  RNG COMPARISON                                                      #
# ================================================================== #

def run_rng_comparison(xs_data, n_histories=100, n_active=30,
                       n_inactive=10, seed=46831729):
    """Run all three RNGs at von Neumann scale (N=100 histories)."""
    generators = [
        ("Middle Square", MiddleSquare(seed=seed)),
        ("LCG",           LCG(seed=seed)),
        ("Mersenne",      MersenneTwister(seed=seed)),
    ]
    results = {}
    for label, rng in generators:
        print(f"\n{'='*50}\n  {label}\n{'='*50}")
        try:
            k_mean, k_std, k_hist = run_simulation(
                rng, xs_data, n_histories=n_histories,
                n_active_cycles=n_active, n_inactive_cycles=n_inactive,
            )
            results[label] = {'k_mean': k_mean, 'k_std': k_std,
                               'k_history': k_hist}
            print(f"\n  RESULT: k_eff = {k_mean:.4f} ± {k_std:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[label] = None
    return results


# ================================================================== #
#  CONVERGENCE STUDY                                                   #
# ================================================================== #

def convergence_study(xs_data, history_counts=None, n_runs=5):
    """
    Mersenne simulation at increasing N. n_runs independent runs per N.
    Demonstrates 1/sqrt(N) convergence.
    """
    if history_counts is None:
        history_counts = [50, 100, 200, 500, 1000]

    k_means, k_stds = [], []
    for n in history_counts:
        run_ks = []
        for run in range(n_runs):
            rng = MersenneTwister(seed=42 + run)
            k_mean, _, _ = run_simulation(
                rng, xs_data, n_histories=n,
                n_active_cycles=20, n_inactive_cycles=20, verbose=False,
            )
            run_ks.append(k_mean)
        mean = float(np.mean(run_ks))
        std  = float(np.std(run_ks))
        k_means.append(mean)
        k_stds.append(std)
        print(f"  N={n:5d}  k={mean:.4f} ± {std:.4f}  ({n_runs} runs)")

    return history_counts, k_means, k_stds


# ================================================================== #
#  MULTI-SEED PRNG STUDY                                               #
# ================================================================== #

def rng_std_comparison(xs_data,
                       history_counts=None,
                       n_runs=10,
                       n_active=20,
                       n_inactive=20,
                       seed=46831729):
    """
    For each RNG and each history count N, run n_runs independent simulations
    (different seed per run) and collect mean(k) and std(k) ACROSS runs.

    Answers: does RNG choice produce statistically resolvable differences
    in either bias or run-to-run variance, and does any difference vanish
    with N?

    n_inactive=20 is used (not 10) so source convergence is not a
    confounder — for a bare critical sphere with R/lambda ~ 3, the
    relaxation timescale 1/(D*B^2) is roughly 8 cycles, so ~10 inactive
    cycles is borderline and ~20 is comfortable.

    Returns: dict mapping rng_label -> (history_counts, k_means, k_stds)
    """
    if history_counts is None:
        history_counts = [50, 100, 150, 200]

    generator_classes = [
        ("Middle Square", MiddleSquare,    {"seed": seed}),
        ("LCG",           LCG,             {"seed": seed}),
        ("Mersenne",      MersenneTwister, {"seed": seed}),
    ]

    all_results = {}

    for label, cls, kwargs in generator_classes:
        print(f"\n  === {label} ===")
        k_means_list, k_stds_list = [], []

        for n in history_counts:
            run_ks = []
            for run in range(n_runs):
                run_kwargs = dict(kwargs)
                run_kwargs["seed"] = seed + run * 1000
                if cls is MiddleSquare:
                    # 8-digit seed required
                    run_kwargs["seed"] = (seed + run * 1000) % 90_000_000 + 10_000_000

                try:
                    rng = cls(**run_kwargs)
                    k_mean, _, _ = run_simulation(
                        rng, xs_data,
                        n_histories=n,
                        n_active_cycles=n_active,
                        n_inactive_cycles=n_inactive,
                        verbose=False,
                    )
                    run_ks.append(k_mean)
                except Exception as e:
                    print(f"    N={n} run {run} failed: {e}")

            if run_ks:
                k_means_list.append(float(np.mean(run_ks)))
                k_stds_list.append(float(np.std(run_ks)))
                print(f"    N={n:4d}  k={k_means_list[-1]:.4f}  "
                      f"std={k_stds_list[-1]:.5f}  ({len(run_ks)} runs)")
            else:
                k_means_list.append(np.nan)
                k_stds_list.append(np.nan)

        all_results[label] = (history_counts, k_means_list, k_stds_list)

    return all_results


def plot_rng_std_comparison(all_results, save_path="rng_std_comparison.png",
                            n_runs=None):
    """
    Two-panel figure:
      Left:  std(k) vs N for each RNG — variance behavior
      Right: mean(k) vs N for each RNG — bias behavior

    n_runs: optional, used only for the suptitle annotation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    markers = ['o', 's', '^']
    colors  = ['steelblue', 'darkorange', 'green']

    for (label, (Ns, means, stds)), marker, color in zip(
            all_results.items(), markers, colors):
        Ns    = np.array(Ns,    dtype=float)
        means = np.array(means, dtype=float)
        stds  = np.array(stds,  dtype=float)

        axes[0].plot(Ns, stds, marker=marker, color=color,
                     label=label, linewidth=1.5, markersize=7)
        axes[1].plot(Ns, means, marker=marker, color=color,
                     label=label, linewidth=1.5, markersize=7)

    # 1/sqrt(N) reference, scaled to first Mersenne point
    if "Mersenne" in all_results:
        Ns_ref, _, stds_ref = all_results["Mersenne"]
        Ns_ref   = np.array(Ns_ref,   dtype=float)
        stds_ref = np.array(stds_ref, dtype=float)
        valid    = ~np.isnan(stds_ref)
        if valid.any():
            s0 = stds_ref[valid][0]
            N0 = Ns_ref[valid][0]
            axes[0].plot(Ns_ref, s0 * np.sqrt(N0 / Ns_ref), 'k--',
                         linewidth=1, label='1/√N reference', alpha=0.6)

    axes[0].set_xlabel("Neutron histories per cycle (N)")
    axes[0].set_ylabel("std(k) across independent runs")
    axes[0].set_title("k-effective variance by RNG")
    axes[0].legend(fontsize=9, loc='best')
    axes[0].grid(alpha=0.3)

    axes[1].axhline(1.000, color='black', linewidth=1.5,
                    linestyle='--', label='Benchmark k=1.000', alpha=0.7)
    axes[1].set_xlabel("Neutron histories per cycle (N)")
    axes[1].set_ylabel("mean k-effective across runs")
    axes[1].set_title("k-effective mean by RNG")
    axes[1].legend(fontsize=9, loc='best')
    axes[1].grid(alpha=0.3)

    suptitle_text = "Multi-seed PRNG comparison"
    if n_runs is not None:
        suptitle_text += f" (n_runs={n_runs} per N)"
    plt.suptitle(suptitle_text, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.show()
    plt.close(fig)


def plot_rng_convergence_large_n(xs_data,
                                  large_counts=None,
                                  n_runs=3,
                                  seed=46831729,
                                  save_path="rng_large_n.png"):
    """
    Same multi-seed analysis at larger N. Expectation: 1/sqrt(N) statistical
    convergence dominates any RNG-quality differences, all three generators
    overlap within error bars.
    """
    if large_counts is None:
        large_counts = [200, 500, 1000]

    all_results = rng_std_comparison(
        xs_data,
        history_counts=large_counts,
        n_runs=n_runs,
        n_active=20,
        n_inactive=20,
        seed=seed,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = ['o', 's', '^']
    colors  = ['steelblue', 'darkorange', 'green']

    for (label, (Ns, means, stds)), marker, color in zip(
            all_results.items(), markers, colors):
        Ns   = np.array(Ns,   dtype=float)
        stds = np.array(stds, dtype=float)
        ax.loglog(Ns, stds, marker=marker, color=color,
                  label=label, linewidth=1.5, markersize=8)

    # 1/sqrt(N) reference, scaled to the first Mersenne point so it lines
    # up with actual data instead of being arbitrarily placed.
    if "Mersenne" in all_results:
        Ns_ref, _, stds_ref = all_results["Mersenne"]
        Ns_ref   = np.array(Ns_ref,   dtype=float)
        stds_ref = np.array(stds_ref, dtype=float)
        valid    = ~np.isnan(stds_ref)
        if valid.any():
            s0 = stds_ref[valid][0]
            N0 = Ns_ref[valid][0]
            ax.loglog(Ns_ref, s0 * np.sqrt(N0 / Ns_ref), 'k--',
                      linewidth=1, label='1/√N reference', alpha=0.6)

    ax.set_xlabel("N histories per cycle")
    ax.set_ylabel("std(k) across independent runs")
    ax.set_title(f"std(k) vs N at larger scale  (n_runs={n_runs})")
    ax.legend(loc='best')
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.show()
    plt.close(fig)

    return all_results


# ================================================================== #
#  PLOTTING                                                            #
# ================================================================== #

def plot_k_convergence(results):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, res in results.items():
        if res is None:
            continue
        hist   = res['k_history']
        cycles = np.arange(1, len(hist) + 1)
        ax.plot(cycles, hist,
                label=f"{label} (mean={res['k_mean']:.4f})",
                alpha=0.85, linewidth=1.5)
    ax.axhline(1.000, color='black', linewidth=2.0, label='Benchmark k=1.000')
    ax.axhspan(0.999, 1.001, alpha=0.08, color='black', label='±0.001 band')
    ax.set_xlabel("Active cycle")
    ax.set_ylabel("k-effective")
    ax.set_title("k-effective per cycle by RNG (von Neumann scale, N=100)")
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("k_convergence.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_convergence_vs_histories(history_counts, k_means, k_stds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].errorbar(history_counts, k_means, yerr=k_stds,
                     fmt='o-', capsize=4, label='Simulation')
    axes[0].axhline(1.000, color='red', linestyle='--',
                    linewidth=1.5, label='Benchmark')
    axes[0].set_xlabel("Neutron histories per cycle")
    axes[0].set_ylabel("k-effective")
    axes[0].set_title("k-effective vs N histories")
    axes[0].legend(loc='best')
    axes[0].grid(alpha=0.3)

    N_arr = np.array(history_counts, dtype=float)
    s_arr = np.array(k_stds, dtype=float)
    ref   = s_arr[0] * np.sqrt(N_arr[0] / N_arr)
    axes[1].loglog(N_arr, s_arr, 'o-', label='Measured std(k)', linewidth=1.5)
    axes[1].loglog(N_arr, ref, '--', color='gray',
                   label='1/√N reference', linewidth=1.5)
    axes[1].set_xlabel("N histories")
    axes[1].set_ylabel("std(k)")
    axes[1].set_title("Convergence rate")
    axes[1].legend(loc='best')
    axes[1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_rng_uniformity(seed=46831729, n=10000):
    """Consecutive-pair scatter: reveals LCG lattice, middle-square cycles."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    generators = [
        ("Middle Square", MiddleSquare(seed=seed)),
        ("LCG",           LCG(seed=seed)),
        ("Mersenne",      MersenneTwister(seed=seed)),
    ]
    for ax, (label, rng) in zip(axes, generators):
        samples = []
        try:
            while len(samples) < n:
                samples.append(rng.random())
        except RuntimeError:
            pass
        s = np.array(samples)
        ax.scatter(s[:-1], s[1:], s=0.3, alpha=0.4, rasterized=True)
        ax.set_title(f"{label}  (n={len(s)})")
        ax.set_xlabel("$x_i$")
        ax.set_ylabel("$x_{i+1}$")
        ax.set_aspect('equal')
    plt.suptitle("Consecutive-pair scatter: RNG uniformity", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("rng_uniformity.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_watt_spectrum(n_samples=100000, rng=None):
    """Sampled Watt spectrum vs theoretical curve — validates sampler."""
    if rng is None:
        rng = MersenneTwister(42)

    samples_MeV = np.array([sample_watt_energy(rng) for _ in range(n_samples)]) / 1e6

    E_MeV  = np.linspace(0.01, 12, 500)
    a_MeV  = WATT_A / 1e6
    b_MeV  = WATT_B * 1e6
    theory = np.exp(-E_MeV / a_MeV) * np.sinh(np.sqrt(b_MeV * E_MeV))
    theory /= np.trapezoid(theory, E_MeV)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(samples_MeV, bins=100, density=True,
            alpha=0.6, label='Sampled', color='steelblue')
    ax.plot(E_MeV, theory, 'r-', linewidth=2, label='Watt (theory)')
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel("Probability density")
    ax.set_title("U-235 fission neutron energy spectrum (Watt)")
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("watt_spectrum.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_flux_profile(positions, radius=RADIUS, n_bins=20,
                      save_path="flux_profile.png"):
    """
    Spatial fission source density S(r) vs. radius, computed from positions
    of fission neutrons born during active cycles.

    After burn-in, the fission source has converged to the dominant
    eigenmode of the transport operator.  Histogramming positions by
    radius and dividing by shell volume 4*pi*r^2*dr gives an estimate
    of the fission source density.  This is a proxy for the spatial
    flux shape (related by S(r) = nu * integral( Sigma_f(E) * phi(r,E) dE )).

    For comparison we overlay the diffusion-theory ground state for a
    bare sphere, S(r) ~ sin(pi*r/R_tilde) / r, with extrapolated radius
    R_tilde = R + 0.7104*lambda from Milne's exact transport result
    (lambda is the transport mfp, ~2.87 cm here).  The shape match is
    qualitative: actual transport gives a similar center-peaked profile
    but corrects the diffusion-theory boundary behavior.
    """
    if not positions:
        print("plot_flux_profile: no positions provided.")
        return

    pos = np.asarray(positions)
    r   = np.linalg.norm(pos, axis=1)

    # Bin radially, divide by shell volume
    edges      = np.linspace(0.0, radius, n_bins + 1)
    counts, _  = np.histogram(r, bins=edges)
    r_centers  = 0.5 * (edges[:-1] + edges[1:])
    shell_vol  = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    density_raw = counts / shell_vol      # absolute density (events / cm^3)
    # Peak-normalized for plotting only
    density = density_raw / density_raw.max() if density_raw.max() > 0 else density_raw

    # Diffusion-theory reference: sin(pi*r/R_tilde)/r, R_tilde extrapolated
    # using Milne's exact transport result, R_tilde = R + 0.7104 * lambda_tr.
    # (NOT R + 2*lambda, which is wrong: that's the diffusion length for an
    # absorbing medium, not the extrapolation distance.)
    lam      = 2.87        # cm, transport mfp at 2 MeV
    R_tilde  = radius + 0.7104 * lam
    r_smooth = np.linspace(1e-3, radius, 300)
    diff_ref = np.sin(np.pi * r_smooth / R_tilde) / r_smooth
    diff_ref = diff_ref / diff_ref.max()
    # Diffusion reference at bin centers, peak-normalized to its max for direct comparison
    diff_at_centers = np.sin(np.pi * r_centers / R_tilde) / r_centers
    diff_at_centers = diff_at_centers / diff_at_centers.max()

    # ---- Diagnostic print: binned simulation vs. diffusion theory ----
    print("\n  Spatial fission source profile (peak-normalized):")
    print("  " + "-" * 60)
    print(f"  {'r [cm]':>8}  {'count':>7}  {'shell_vol':>10}  "
          f"{'sim':>7}  {'diff':>7}  {'sim-diff':>9}")
    print("  " + "-" * 60)
    for rc, n_evt, sv, ds, dd in zip(r_centers, counts, shell_vol,
                                     density, diff_at_centers):
        print(f"  {rc:8.3f}  {n_evt:7d}  {sv:10.3f}  "
              f"{ds:7.3f}  {dd:7.3f}  {ds-dd:+9.3f}")
    print("  " + "-" * 60)
    print(f"  Total positions: {len(positions)}")
    print(f"  Radius: R = {radius} cm,  R_tilde = R + 0.7104*lambda = {R_tilde:.2f} cm")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_centers, density, 'o-', linewidth=1.5, markersize=6,
            label=f"Simulation (n={len(positions)} sites)",
            color='steelblue')
    ax.plot(r_smooth, diff_ref, '--', linewidth=1.5,
            label=r"Diffusion theory $\propto \sin(\pi r / \tilde R)/r$",
            color='gray')
    ax.axvline(radius, color='black', linestyle=':', linewidth=1,
               label=f"R = {radius:.4f} cm")
    ax.set_xlabel("Radius r [cm]")
    ax.set_ylabel("Fission source density (peak-normalized)")
    ax.set_title("Spatial fission source profile (active cycles only)")
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {save_path}")
    plt.show()
    plt.close(fig)


# ================================================================== #
#  MAIN                                                                #
# ================================================================== #

if __name__ == "__main__":

    # 1. load cross sections
    print("Loading cross sections...\n")
    xs_data = load_all_cross_sections()

    sigma_2MeV = build_sigma_tables(xs_data, E_eV=2e6)
    macro_2MeV = build_macro_sigma(sigma_2MeV)
    print(f"\nMacroscopic cross sections at 2 MeV:")
    for rxn, val in macro_2MeV.items():
        print(f"  Sigma_{rxn:<10} = {val:.6f} cm^-1")
    print(f"  Mean free path  = {1/macro_2MeV['total']:.4f} cm")

    # 2. validate Watt sampler
    print("\nValidating Watt spectrum sampler...")
    plot_watt_spectrum()

    # 3. RNG uniformity
    print("\nRNG uniformity plots...")
    plot_rng_uniformity()

    # 4. middle-square cycle detection
    ms  = MiddleSquare(seed=46831729)
    cyc = ms.detect_cycle()
    print(f"\nMiddle-square cycle length (seed=46831729): {cyc}")
    print(f"  Hayes 2022: median ~2700 for 8-digit decimal")

    # 5. RNG comparison at von Neumann scale
    print("\nRNG comparison (N=100, von Neumann scale)...")
    results = run_rng_comparison(xs_data, n_histories=100,
                                 n_active=30, n_inactive=20)
    plot_k_convergence(results)

    # 6. *** Main PRNG study ***
    #    Multi-seed std(k) and mean(k) vs N at von Neumann scale.
    #    Question: does RNG choice produce statistically resolvable
    #    differences in either bias or run-to-run variance?
    #
    #    NB: this study covers the same ground as a Mersenne-only
    #    convergence_study (which we omit) AND adds the LCG/MS comparison.
    print("\n" + "="*50)
    print("Multi-seed PRNG study at von Neumann scale")
    print("="*50)
    rng_std_results = rng_std_comparison(
        xs_data,
        history_counts=[50, 100, 150, 200],
        n_runs=10,
        n_active=20,
        n_inactive=20,
        seed=46831729,
    )
    plot_rng_std_comparison(rng_std_results, n_runs=10)

    # 7. Same study at larger N: expect 1/sqrt(N) to dominate any
    #    RNG-quality differences for the modern PRNGs.  The middle-square
    #    plateau in std(k) is expected to persist at all N.
    print("\n" + "="*50)
    print("Multi-seed PRNG study at larger N")
    print("="*50)
    plot_rng_convergence_large_n(
        xs_data,
        large_counts=[200, 500, 1000],
        n_runs=10,
        seed=46831729,
    )

    # 8. final validation (with flux tracking)
    print("\n" + "="*50)
    print("Final validation (Mersenne, N=1000, with flux profile)")
    print("="*50)
    rng = MersenneTwister(seed=42)
    k_mean, k_std, _, fission_positions = run_simulation(
        rng, xs_data, n_histories=1000,
        n_active_cycles=50, n_inactive_cycles=20,
        track_flux=True,
    )
    print(f"\nResult:    k_eff = {k_mean:.4f} ± {k_std:.4f}")
    print(f"Benchmark: k_eff = 1.0000 ± 0.0010")
    print(f"Delta:     {abs(k_mean - 1.0)*100:.2f}%")
    print(f"Collected {len(fission_positions)} fission positions for flux plot.")

    plot_flux_profile(fission_positions)