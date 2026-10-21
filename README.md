# Does the Gate Use the Regime?

Code and data for **"Does the Gate Use the Regime? Placebo Inference and Pre-Tests for Regime-Gated Model Selection"** by Michael Zhu Ning.

The scripts reproduce every table and figure in the paper and its Online Appendix from the public monthly predictor data of Welch and Goyal (2008), 1926–2020, included here as `PredictorData1926-2020.csv`.

## Setup

Python 3.12. Install the dependencies with

```bash
pip install -r requirements.txt
```

Run every script from inside this folder; all paths are relative to it. Intermediate results are saved as `.npz` and `.json` files in the same folder.

## Main results (five experts, equity premium)

```bash
python prep.py                          # out-of-sample expert forecasts, 1947-2020 -> streams.npz
python select2.py dma                   # dynamic model averaging
python select2.py msnoz                 # Markov switching without indicators
PEN=2 SUF= python select2.py gate       # primary specification: linear gate, lambda = 2 -> W_gate.npz
PEN=2 SUF= python select2.py filt       # filtered gate, lambda = 2 -> W_filt.npz
PEN=0 SUF=_pen0 python select2.py gate  # unshrunk linear gate -> W_gate_pen0.npz
PEN=0 SUF=_pen0 python select2.py filt  # unshrunk filtered gate -> W_filt_pen0.npz
PEN=0.5 SUF=_p0.5 python select2.py gate ; PEN=8 SUF=_p8 python select2.py gate
PEN=0.5 SUF=_p0.5 python select2.py filt ; PEN=8 SUF=_p8 python select2.py filt
PEN=0 SUF=_p0 python select2.py mlp     # unshrunk MLP gate
PEN=2 SUF=_mlp2 python select2.py mlp   # shrunk MLP gate
python analysis.py                      # main performance tables and both figures
```

## Placebo-gate test

```bash
python ceiling2.py                      # fits the indicator VAR (var_fit.npz); hindsight ceilings with VAR and rotation placebos
python placebo_batch.py real            # reproduces the primary gate (R2 = 0.9333)
python placebo_batch.py var 0 999       # 999 VAR placebo gates, lambda = 2
python placebo_batch.py rot 0 999       # 999 rotation placebo gates, lambda = 2
PEN=0 python placebo_batch.py var 0 199 # unshrunk gate
PEN=0.5 python placebo_batch.py var 0 250 ; PEN=8 python placebo_batch.py var 0 400
python placebo_batch.py const           # constant-weight benchmark
python init_check.py ; PEN=0 python init_check.py   # sensitivity to random starting values
```

Long runs can be split into batches (`var 0 250`, `var 250 500`, ...); each batch saves its own JSON file.

## Pre-test Monte Carlo

```bash
python mc_power2.py <beta> 200 <seed> <dispersion>        # beta in {0, 2, 8}, dispersion in {1, 3, 10}
python mc_mis.py <thresh|interact|latent> <g> <dispersion> 200 <seed>   # g in {0.5, 1}, dispersion in {1, 3}
```

As in the paper, run `python var_pre.py` first and set `VARFILE=var_fit_pre.npz` to use the indicator VAR estimated on 1947-1988 data.

## Size of the placebo test (Online Appendix)

```bash
python size_sim.py <start> <stop> 39    # t(5)-innovation indicator VAR, VAR re-estimated, 39 Gaussian placebos
python size_corr.py <start> <stop> 19   # real indicators and experts, outcome = equal weights + block-permuted residuals
```

## Eighteen experts and realized volatility

```bash
python prep_big.py                                          # -> streams_big.npz
STREAMS=streams_big.npz PEN=2 SUF=_big_p2 python select3.py gate
STREAMS=streams_big.npz TAG=big python placebo_batch.py var 0 499
python eval_big.py

python prep_vol.py                                          # -> streams_vol.npz (log monthly realized variance)
STREAMS=streams_vol.npz PEN=2 SUF=_vol_p2 python select3.py gate
STREAMS=streams_vol.npz STAT=mse_ew TAG=vol python placebo_batch.py var 0 499
python eval_vol.py
```

## Other files

- `select3.py`: the same estimator as `select2.py`, with options to read alternative forecast streams (`STREAMS`) and indicator paths (`ZFILE`).
- `ceiling.py`, `placebo.py`, `mc_power.py`: earlier versions of the ceiling, placebo and Monte Carlo code, kept for reference.
- `select_original.py`: the first estimation script, which set the primary penalty `PEN = 2.0` before any evaluation-sample result existed; kept to document the chronology of the primary specification.

## License

MIT. See `LICENSE`.
