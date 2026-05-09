"""
Cross section data loader for U-234, U-235, U-238.
Data source: ENDF/B-VIII.0 via NNDC BNL
Reactions: MT1 (total), MT2 (elastic), MT18 (fission), MT102 (radiative capture)
"""

import json
import numpy as np
import os
from scipy.interpolate import interp1d

# ------------------------------------------------------------------ #
#  File manifest: isotope -> reaction -> file ID                     #
# ------------------------------------------------------------------ #

BASE_DIR = os.path.join(os.path.dirname(__file__), "cross_section_data")

FILE_MANIFEST = {
    "U-234": {
        "total":    "13662201.json",
        "elastic":  "13662202.json",
        "fission":  "13662207.json",
        "capture":  "13662254.json",
    },
    "U-235": {
        "total":    "13662335.json",
        "elastic":  "13662336.json",
        "fission":  "13662341.json",
        "capture":  "13662382.json",
    },
    "U-238": {
        "total":    "13662978.json",
        "elastic":  "13662979.json",
        "fission":  "13662984.json",
        "capture":  "13663025.json",
    },
}

# MT numbers for reference
MT_CODES = {
    "total":   1,
    "elastic": 2,
    "fission": 18,
    "capture": 102,
}

# ------------------------------------------------------------------ #
#  Raw loader: returns dict with metadata + numpy arrays             #
# ------------------------------------------------------------------ #

def load_xs_file(filepath):
    """
    Load a single ENDF JSON cross section file.
    Returns a dict:
        {
            'id':        str,
            'library':   str,
            'target':    str,
            'reaction':  str,
            'MT':        int,
            'temp_K':    float,
            'n_pts':     int,
            'E':         np.ndarray  [eV],
            'sigma':     np.ndarray  [barns],
            'dsigma':    np.ndarray or None  [barns],  # uncertainty if present
        }
    """
    with open(filepath, 'r') as f:
        raw = json.load(f)

    ds = raw['datasets'][0]
    pts = ds['pts']

    E     = np.array([p['E']    for p in pts], dtype=np.float64)
    sigma = np.array([p['Sig']  for p in pts], dtype=np.float64)

    # uncertainty column only present in some files
    if 'dSig' in pts[0]:
        dsigma = np.array([p['dSig'] for p in pts], dtype=np.float64)
    else:
        dsigma = None

    return {
        'id':       ds['id'],
        'library':  ds['LIBRARY'],
        'target':   ds['TARGET'],
        'reaction': ds['REACTION'],
        'MT':       ds['MT'],
        'temp_K':   ds['TEMP'],
        'n_pts':    ds['nPts'],
        'E':        E,
        'sigma':    sigma,
        'dsigma':   dsigma,
    }

# ------------------------------------------------------------------ #
#  Build interpolators for a single dataset                          #
# ------------------------------------------------------------------ #

def build_interpolator(xs_data, scale='log-log'):
    """
    Build a scipy interpolator from a loaded cross section dataset.

    scale: 'log-log' (default, best for resonance regions)
           'lin-lin' (linear, fine for smooth regions)

    Returns a callable f(E_eV) -> sigma_barns.
    Extrapolation outside data range raises ValueError.
    """
    E     = xs_data['E']
    sigma = xs_data['sigma']

    # remove any zero or negative values before log (can appear at thresholds)
    valid = (E > 0) & (sigma > 0)
    E     = E[valid]
    sigma = sigma[valid]

    if scale == 'log-log':
        log_E     = np.log(E)
        log_sigma = np.log(sigma)
        interp = interp1d(log_E, log_sigma,
                          kind='linear',
                          bounds_error=True)
        def interpolator(E_query):
            E_query = np.atleast_1d(np.asarray(E_query, dtype=np.float64))
            return np.exp(interp(np.log(E_query)))

    elif scale == 'lin-lin':
        interp = interp1d(E, sigma,
                          kind='linear',
                          bounds_error=True)
        def interpolator(E_query):
            E_query = np.atleast_1d(np.asarray(E_query, dtype=np.float64))
            return interp(E_query)

    else:
        raise ValueError(f"Unknown scale '{scale}'. Use 'log-log' or 'lin-lin'.")

    # attach metadata
    interpolator.E_min = E.min()
    interpolator.E_max = E.max()
    interpolator.target   = xs_data['target']
    interpolator.reaction = xs_data['reaction']

    return interpolator

# ------------------------------------------------------------------ #
#  Load everything into a nested dict                                #
# ------------------------------------------------------------------ #

def load_all_cross_sections(base_dir=BASE_DIR, scale='log-log'):
    """
    Load all cross sections for U-234, U-235, U-238.

    Returns nested dict:
        cross_sections[isotope][reaction] = {
            'data':        raw dict from load_xs_file(),
            'interpolate': callable f(E_eV) -> sigma_barns,
        }

    Example usage:
        xs = load_all_cross_sections()
        sigma_f = xs['U-235']['fission']['interpolate'](2e6)  # at 2 MeV
    """
    cross_sections = {}

    for isotope, reactions in FILE_MANIFEST.items():
        cross_sections[isotope] = {}
        iso_dir = os.path.join(base_dir, isotope)

        for reaction, filename in reactions.items():
            filepath = os.path.join(iso_dir, filename)

            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Missing file: {filepath}\n"
                    f"Expected for {isotope} {reaction}"
                )

            data        = load_xs_file(filepath)
            interpolator = build_interpolator(data, scale=scale)

            cross_sections[isotope][reaction] = {
                'data':        data,
                'interpolate': interpolator,
            }

        print(f"Loaded {isotope}: "
              + ", ".join(f"{r} ({cross_sections[isotope][r]['data']['n_pts']} pts)"
                          for r in reactions))

    return cross_sections

# ------------------------------------------------------------------ #
#  Convenience: get all sigma values at a single energy              #
# ------------------------------------------------------------------ #

def get_sigma_at_energy(cross_sections, E_eV):
    """
    Query all isotopes and reaction types at a given energy E_eV.

    Returns dict:
        result[isotope][reaction] = sigma in barns
    """
    result = {}
    for isotope, reactions in cross_sections.items():
        result[isotope] = {}
        for reaction, xs in reactions.items():
            try:
                result[isotope][reaction] = float(
                    xs['interpolate'](E_eV)
                )
            except Exception as e:
                result[isotope][reaction] = None
                print(f"Warning: {isotope} {reaction} at {E_eV:.3e} eV failed: {e}")
    return result


# ------------------------------------------------------------------ #
#  Quick sanity check                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("Loading cross sections...\n")
    xs = load_all_cross_sections()

    # query at 2 MeV — our one-speed approximation energy
    E_test = 2e6  # eV
    print(f"\nCross sections at E = {E_test:.2e} eV (2 MeV):\n")
    print(f"{'Isotope':<10} {'Reaction':<12} {'sigma (barns)':>16}")
    print("-" * 42)

    sigma_at_2MeV = get_sigma_at_energy(xs, E_test)
    for isotope in ["U-234", "U-235", "U-238"]:
        for reaction in ["total", "elastic", "fission", "capture"]:
            val = sigma_at_2MeV[isotope][reaction]
            if val is not None:
                print(f"{isotope:<10} {reaction:<12} {val:>16.6f}")
            else:
                print(f"{isotope:<10} {reaction:<12} {'N/A':>16}")
        print()