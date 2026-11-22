# Pore-connectivity parameter *l* of the van Genuchten–Mualem model

Data and scripts to reproduce the figures and numbers of the manuscript. The folders are numbered by
the section of the paper they belong to; each holds `scripts/`, `results/` and, where needed, `data/`.

| Folder | Paper | Content |
|---|---|---|
| `3.1_Numerical_experiments/` | Sect. 2.2, 3.1, Fig. 1 | COMSOL simulations of 1 m columns of sand, silt and clay |
| `3.2-3.3_Literature_review/` | Sect. 2.1, 3.2–3.3, Figs. 2–3 | Literature database: SA methods, treatment and reported values of *l* |
| `3.4_SA_SEN_PEST/` | Sect. 3.4, Fig. 4 | Morris screening of both lysimeters |
| `3.5_Interaction/` | Sect. 3.5, Fig. 5 | Pairwise 5 × 5 factorial interaction screen |
| `3.6_SWP_PEST/` | Sect. 3.6, Fig. 6, Tables S5–S6 | 1,000,000-run Monte Carlo sweeps, with *l* free and with *l* = 0.5 |
| `3.7_Decision_chart/` | Sect. 3.7, Fig. 7 | Decision chart |
| `HYDRUS_models/` | Sect. 2.3 | HYDRUS-1D input files and measured outflow of Ly1 and Ly2 |
| `bin/` | | HYDRUS-1D executables (`mac/hydrus`, `windows/H1D_CALC.exe`) |

## Setup
```
conda env create -f environment.yml
conda activate sensitivity_analysis
```
All figure scripts need only Python and read the archived data in the repository. Run them from their
own folder. Re-running HYDRUS-1D (interaction screen) writes working files to `tmp_copy/` and
`LEVEL_01.dir` at the repository root. The Morris runs and the sweeps were run with `pestpp-swp`
(https://github.com/usgs/pestpp) on an HTCondor pool; their run scripts are kept as a record.

## Reproduce the figures
| Figure | Command | Output |
|---|---|---|
| Fig. 1 | `cd 3.1_Numerical_experiments/scripts; python fig1_comsol.py` | `results/Figure1_Seff_gradH.png` |
| Fig. 2 | `cd 3.2-3.3_Literature_review/scripts; python plot_sa_methods_and_treatment_pies.py` | `results/fig2_SA_methods_and_l_treatment.png` |
| Fig. 3 | `python plotting.py` (same folder) | `results/fig3_l_by_soil_type.png` |
| Fig. 4 | `cd 3.4_SA_SEN_PEST/scripts; python fig4_morris_convergence.py` | `results/fig4_morris_convergence.png` |
| Fig. 5 | `cd 3.5_Interaction/scripts; python fig5_ddq.py` | `results/fig5_ddq.png` |
| Fig. 6 | `cd 3.6_SWP_PEST/scripts; python fig6_fit.py` | `results/fig6_fit.png` |
| Fig. 7 | `cd 3.7_Decision_chart; python fig7_decision_chart.py` | `fig7_decision_chart.png`, `.svg` |

## 3.1_Numerical_experiments
`data/` holds the COMSOL Multiphysics 6.4 exports of the 1 m columns (sand, silt, clay; *l* = −4 to 4;
100 and 500 mm yr⁻¹; `Output_tables/<Soil>_<flux>_<quantity>_lambda_<l>.csv` with quantity `Seff`,
`theta` or `Q_gradH`), the hydraulic gradient for *l* up to 8 (`lambda-gradH.xlsx`), the soil
parameters (`Soil_params.xlsx`, Text S2) and the model set-up (`Comsol_setup_unsat_VgLambda_v01.docx`).
Fig. 1a uses the 100 mm yr⁻¹ `Seff` tables, Fig. 1b `lambda-gradH.xlsx`. The COMSOL model file itself
is not included. For silt at 500 mm yr⁻¹ only part of the *l* values were exported.

## 3.2-3.3_Literature_review
Both scripts read `data/Literature Review_summary.xlsx` (sheet "Sensitivity analysis"; Table S1).
`results/fig2_counts.log` and `fig3_by_soil.log` give the counts quoted in Sects. 3.2–3.3. Fig. 3 uses
a linear axis from −4 to 6.5; the three study midpoints outside it are drawn as labelled arrows.

## 3.4_SA_SEN_PEST: Morris screening
Both lysimeters with the same design: SALib trajectories, r = 2000, p = 10 levels, 14,000 HYDRUS-1D
runs per lysimeter. Bounds (`results/ly*_morris_p10/design/morris_design.json`, Table S4): θr 0.001–0.1,
θs 0.15–0.4, n 1.05–2.3, *l* −2 to 5; α 0.005–0.5 and Ks 4,000–80,000 (Ly1), α 0.001–0.5 and Ks
10–80,000 (Ly2). Elementary effects that touch a failed run or a run taking in more than 1.01 × the
precipitation are removed (Ly1: 560 failed + 799 non-physical of 14,000 runs; Ly2: 553 + 968;
`results/ly*_morris_p10/logs/analysis.log`).

`fig4_morris_convergence.py` reads `results/ly*_morris_p10/results/model_ly*/morris_swp/analysis/elementary_effects.csv`
and writes Fig. 4 (KGE of daily outflow, the measure of the sweeps) and the same figure for the other
objectives (`figS_morris_convergence_{q,lq,mvol,cum}.png`: daily outflow error, log10 outflow, monthly and
cumulative volume).
μ\* and σ per objective are in `analysis/morris_<objective>.csv`.

Record of the runs (`scripts/morris_ly1/`, `morris_ly2/`): `design.py` (trajectories), `run_morris.py`
(pestpp-swp on HTCondor), `analyze.py` (elementary effects), `run_chain.sh`. They need the `hydrus_sa`
package of the cluster repository and the simulated outflow (`sweep_out.csv`, 95 MB per lysimeter),
neither included. Archived per lysimeter: `design/`, `sweep_in.csv` (parameter sets), `metrics.csv`
(scores and water balance per run), `failed_runs.csv`, `run_manifest.json`, `logs/`.

## 3.5_Interaction: pairwise factorial screen
```
cd 3.5_Interaction/scripts
python fig5_ddq.py              # Fig. 5 -> results/fig5_ddq.png, ddq_table.csv, excluded_runs.csv (Text S5, Table S9)
python interaction_test_ly1.py  # re-run the Ly1 grid (HYDRUS-1D, 375 runs) -> results/interaction_test_model_ly1/pairwise_grid.csv
python interaction_test_ly2.py
```
Every parameter has 5 levels; for *l* −2, −0.25, 1.5, 3.25, 5. The other four parameters stay at the
calibrated values. `pairwise_grid.csv`: p1, p2, v1, v2, q_sim (cumulative outflow, cm), t_end (d). Runs
that did not finish or whose outflow exceeds the 290.4 cm of precipitation are dropped (1 for Ly1, 6
for Ly2). Where a corner run is missing, ΔΔQ is taken from the widest rectangle of valid levels
(`reduced_range = True`, marked * in Fig. 5). The grid scripts skip points already in
`pairwise_grid.csv`; `linking_test_ly1.py` provides the HYDRUS-1D call.

## 3.6_SWP_PEST: Monte Carlo sweeps
```
cd 3.6_SWP_PEST/scripts
python fig6_fit.py              # Fig. 6
python numbers_sec3_6.py        # numbers of Sect. 3.6, Tables S5-S6 -> results/numbers_sec3_6.log
python compare_l05.py           # Fig. S3: l free against l = 0.5 (Sect. 3.6) -> results/figS3_l05_paired.png
python figS_l_planes.py         # Fig. S2: best KGE in the planes of l with the other parameters
python figS_corner.py           # Figs. S4, S5: corner plots, all usable vs best 1 % / 0.1 %
python figS_corner_l05.py       # Figs. S6, S7: corner plots, best 1 % with l free vs l = 0.5
```
`sweeps.py` holds the shared paths and helpers. Usable: not flagged by PEST++, all 1,216 days present
and, for Ly2, not non-physical (infiltration > 1.01 × precipitation, checked for the best 2,000 runs).

Archived data:
- `results/ly1_1M/`: Ly1 sweep, *l* −2 to 5. `sweep_in.csv.gz` (1,000,000 parameter sets),
  `sweep_scores_5pct.csv.gz` (per run: `failed_flag`, `n_failed_obs`, `kge`, `r2`), PEST++ control,
  template and instruction files, `run_manifest.json`.
- `results/ly2_1M/`: Ly2 sweep, *l* −2 to 5, α 0.001–0.5, n 1.05–4. `sweep_in.csv.gz`,
  `sweep_scores.csv.gz` (`failed_flag`, `kge`, `nonphysical`), `top2000_water_balance.csv`.
- `results/l05_1M/`: the same parameter sets run with *l* = 0.5; `ly*/paired.csv.gz` holds both KGE
  values per row; `run_chain.sh` and `compare_l.py` record the runs.

`SWP_setup_parallel.py` and `obs_config.py` set up a sweep for local PEST++ workers;
`run_chain_ly1_1M.sh` records how the Ly1 sweep was run on HTCondor (needs the `hydrus_sa` package, not
included). Not included because of size: the simulated outflow of every run (`sweep_out.csv`, several GB).
