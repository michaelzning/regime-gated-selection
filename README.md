Install dependencies with: pip install -r requirements.txt (Python 3.12).

Reproduce all tables and figures of Regime_Gated_Selection.tex (Python 3.12, numpy, scipy, jax, matplotlib).
Run every script from inside this folder; all paths are relative to it.

python prep.py                                   # expert streams 1947-2020
python select2.py dma ; python select2.py msnoz
for p in 0 0.5 2 8: PEN=$p SUF=_p$p python select2.py gate ; PEN=$p SUF=_p$p python select2.py filt
PEN=0 SUF=_p0 python select2.py mlp ; PEN=2 SUF=_mlp2 python select2.py mlp
python analysis.py                               # tables (paper_results.json) and figures
python placebo.py                                # hindsight-ceiling placebo test

File-name note: analysis.py expects the lambda=2 runs as W_gate.npz / W_filt.npz and the
lambda=0 runs as W_gate_pen0.npz / W_filt_pen0.npz; rename the _p2 / _p0 outputs accordingly.

Hindsight ceilings and Monte Carlo experiments
python ceiling.py                                # hindsight ceilings, matched placebos, family-wise p, fineness, combination ceiling
# placebo-gate test: build zplac_k.npy (AR(1) with each indicator's persistence), then
ZFILE=zplac_0.npy PEN=2 SUF=_plac0_p2 python select3.py gate   # repeat for k = 0..19, and PEN=0 for k = 0..7
python mc_power.py <beta> <reps> <seed> <dispersion>           # Monte Carlo cells: beta in {0,2,8}, dispersion in {1,3,10}, 150 reps

VAR and rotation placebo tests
python ceiling2.py        # VAR(1) and rotation placebos, (1+#)/(B+1) p-values, family-wise tests, 1957-2020 and 1957-1988
                          # writes var_fit.npz used below; the ex ante 1957-1988 table rows refit the VAR on 1947-1988 data
# placebo-gate test: zvar_k.npy = sim_var(...) paths from var_fit.npz, then for k = 0..38 (lambda=2) and 0..18 (lambda=0):
ZFILE=zvar_k.npy PEN=2 SUF=_vark_p2 python select3.py gate
python mc_power2.py <beta> 200 <seed> <dispersion>   # pre-test on 1957-1988, gate trained 1947-1988, tested 1989-2020

Eighteen-expert library (Section "A Larger Expert Library")
python prep_big.py                                         # writes streams_big.npz
STREAMS=streams_big.npz PEN=2 SUF=_big_p2 python select3.py gate   # also PEN=0 SUF=_big_p0 gate; PEN=2 SUF=_big_p2 filt
STREAMS=streams_big.npz ZFILE=zvar_k.npy PEN=2 SUF=_bigvark_p2 python select3.py gate   # placebo gates, k = 0..18
python eval_big.py                                         # performance table and pre-test diagnostics

Realized-volatility application (Section "A Volatility Application")
python prep_vol.py                                          # writes streams_vol.npz (target: log monthly realized variance)
STREAMS=streams_vol.npz PEN=2 SUF=_vol_p2 python select3.py gate   # also PEN=0 SUF=_vol_p0 gate; PEN=2 SUF=_vol_p2 filt
STREAMS=streams_vol.npz ZFILE=zvar_k.npy PEN=2 SUF=_volvark_p2 python select3.py gate   # placebo gates, k = 0..18
python eval_vol.py                                          # table and pre-test diagnostics

Misspecified Monte Carlo designs (subsection "Misspecified selection mechanisms")
python mc_mis.py <thresh|interact|latent> <g in {0.5,1}> <dispersion in {1,3}> 200 <seed>

Fast placebo-gate test (masked likelihood, compiled once; ~0.4 s per placebo gate)
python placebo_batch.py real                                   # reproduces the real shrunk gate (R2 0.9333)
python placebo_batch.py var 0 999 ; python placebo_batch.py rot 0 999      # 5 experts, lambda = 2
PEN=0 python placebo_batch.py var 0 199                        # unshrunk
STREAMS=streams_big.npz TAG=big python placebo_batch.py var 0 499
STREAMS=streams_vol.npz STAT=mse_ew TAG=vol python placebo_batch.py var 0 499
(split long runs into batches; results are saved per batch as JSON)

Null-calibration (size) experiment for the full placebo-gate procedure
python size_sim.py <start> <stop> 39      # replications start..stop-1; t(5)-innovation VAR indicators, VAR re-estimated, 39 Gaussian placebos
(115 replications were run in batches of about 16; results are saved per batch as JSON)

Size under a correlated null
python size_corr.py <start> <stop> 19   # real indicators and experts, outcome = fbar + block-permuted residuals
