# Monte Carlo Neutron Transport: Revisiting the ENIAC Era

A Monte Carlo simulation of neutron transport for the *Godiva* bare highly-enriched-uranium critical assembly (HEU-MET-FAST-001), with a side study comparing the pseudo-random number generator (PRNG) that von Neumann actually deployed on ENIAC in 1948 against modern alternatives.

The simulation tracks individual neutron histories through fission, scattering, and capture using a power-iteration estimator for the criticality multiplier $k$. Cross sections are taken from ENDF/B-VIII.0 with log-log interpolation; fission birth energies are sampled from the Watt spectrum.

## Results at a glance

- **Validation.** Against the Godiva benchmark $k_\text{eff} = 1.000 \pm 0.001$, the simulation yields $\hat{k} = 1.10 \pm 0.006$. The 10% overshoot is attributed to a simplified inelastic-scatter kinematics treatment.
- **PRNG comparison.** Von Neumann's middle-square method (8 decimal digits, the historically authentic implementation; see the write-up for details) shows a ~10% systematic bias and a run-to-run standard deviation that does **not** track the $1/\sqrt{N}$ scaling predicted by the central limit theorem. A modern linear congruential generator and the Mersenne Twister agree within statistical uncertainty at every $N$ tested.

The variance-scaling failure is attributable to cyclic reuse of the fixed 2,000-number block; the bias is empirically robust but its mechanism is not isolated.

## Repository contents

| File / folder | Purpose |
|---|---|
| `simulate.py` | Main simulation: geometry, transport, power iteration, multi-seed driver, plotting |
| `cross_sections.py` | ENDF/B-VIII.0 data loader with log-log interpolation |
| `prngs/` | Standalone PRNG implementations (`ms.py`, `lcg.py`, `mt.py`) |
| `cross_section_data.tar.gz` | Compressed ENDF/B-VIII.0 cross-section JSON files for U-234, U-235, U-238 |
| `Rees_Jakob_FinalReport.pdf` | Full write-up |

## Getting started

Unpack the cross-section data before running anything:

```bash
tar -xzf cross_section_data.tar.gz
```

This produces `cross_section_data/{U-234,U-235,U-238}/` containing the per-isotope JSON files referenced by `cross_sections.py`.

Run the simulation:

```bash
python simulate.py
```

Dependencies: `numpy`, `scipy`, `matplotlib`.

## Cross-section data

The bundled archive contains ENDF/B-VIII.0 evaluations for U-234, U-235, and U-238 with reactions MT1 (total), MT2 (elastic), MT18 (fission), and MT102 (radiative capture), retrieved from the National Nuclear Data Center at Brookhaven (https://www.nndc.bnl.gov/endf/).

## Author

Jakob Rees
