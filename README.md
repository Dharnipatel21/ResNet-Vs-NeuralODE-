# ResNet-Vs-NeuralODE-
Parameter Efficiency and Irregular Sampling in Neural ODEs: An Empirical Study on PhysioNet ICU Data
A comparison of ResNet and Neural ODE architectures for next-step vital-sign prediction on ICU patient data, testing whether Neural ODE's theoretical advantages — parameter efficiency and native handling of irregular time sampling — hold up empirically on real, irregularly-sampled clinical data.

Built on top of "Neural Ordinary Differential Equations" (Chen, Rubanova, Bettencourt & Duvenaud, NeurIPS 2018, Best Paper).

Motivation
ResNets update their hidden state layer-by-layer: h(t+1) = h(t) + f(h(t)). This is mathematically one step of Euler's method for solving a differential equation. Chen et al. (2018) made this literal: replace the discrete stack of layers with an ODE solver, treating "depth" as continuous. The paper claims two main benefits:

Parameter efficiency — a continuous-depth model can match a discrete ResNet's accuracy with substantially fewer parameters.
Native handling of irregular sampling — because the model integrates over continuous time, it can naturally use real, unevenly-spaced timestamps, unlike a ResNet which assumes uniform steps.
This project tests both claims on the PhysioNet Challenge 2012 ICU dataset, which has genuinely irregular real-world vital sign timestamps — a natural fit for testing claim (2).

Method
Task: predict the next set of vital signs (HR, blood pressure, temp, GCS, etc.) from a 10-step window of prior readings.
Two data regimes:
Pre-binned — timestamps discarded, readings treated as uniformly spaced.
Real-TS — actual irregular timestamps preserved and used.
Two architectures, each built for both regimes:
ResNetModel — stack of independently-weighted residual blocks.
NeuralODE — single reused block, integrated via torchdiffeq.odeint (dopri5 adaptive solver). For Real-TS, each sample integrates over its own real elapsed time gap, via a rescaled-time trick: the vector field is scaled by each sample's individual gap so the whole batch can still be solved in one call over a shared [0, 1] interval.
Two ResNet sizes per regime, to separate architecture effects from capacity effects:
ResNet-4block — a normal-sized ResNet (~37–38k params).
ResNet-1block — capacity-matched to Neural ODE (~13k params, same as the ODE model) for a fair architecture-only comparison.
Patient-level train/test split (80/20) with scalers fit on training patients only, to avoid data leakage.
External validation on PhysioNet Set B — a fully separate patient cohort — to check generalization beyond the original split.
Multi-seed replication (6 runs) of the capacity-matched comparison, to check whether any observed advantage is a stable effect or random noise.
Results
Full-size ResNet (4-block) vs Neural ODE
Regime	ResNet MAE	ODE MAE	ODE advantage
Pre-binned	0.0207	0.0218	-5.2% (ResNet wins)
Real-TS	0.0196	0.0202	-2.9% (ResNet wins)
Capacity-matched (ResNet-1block, same parameter count as ODE)
Regime	ResNet-1block MAE	ODE MAE	ODE advantage
Pre-binned	0.0213	0.0218	-2.2% (ResNet wins)
Real-TS	0.0231	0.0202	+12.3% (ODE wins)
(ODE advantage = (ResNet_MAE − ODE_MAE) / ResNet_MAE × 100; positive = ODE has lower error.)

Multi-seed replication of the capacity-matched Real-TS result
A single run showed a +12.3% ODE advantage on real timestamps. Re-running the same comparison across 6 random seeds:

Seed	ODE advantage
42	+3.3%
123	−8.8%
2026	−7.6%
7	+2.6%
555	−9.5%
314	+6.0%
Mean: −2.3% ± 6.4% — the range spans both positive and negative values, i.e. it crosses zero. This means the +12.3% single-run result was not a stable effect; across repeated trials neither architecture reliably outperforms the other on this task once parameter count is matched.

Parameter efficiency
Regime	ResNet params	ODE params	Ratio
Pre-binned	37,574	12,614	2.98x
Real-TS	38,279	13,319	2.87x
The full 4-block ResNet's accuracy edge over ODE is small (2.9–5.2%) despite using ~3x the parameters — and once parameters are matched exactly, that edge disappears (and the direction becomes statistically uncertain, per the multi-seed result above).

Adaptive computation (solver function evaluations)
The paper predicts the ODE solver should adapt its number of integration steps (NFE) to how difficult the input is to integrate, rather than doing a fixed amount of work like a discrete network.

Model	NFE, epoch 1	NFE, epoch 60
ODE (Pre-binned)	39.7	44.0 (varies 38–44 across training)
ODE (Real-TS)	26.0	26.3 (essentially flat)
Supported for pre-binned; not observed for Real-TS, likely because the rescaled-time integration trick normalizes all samples onto the same [0,1] interval, reducing the variation in integration difficulty the solver would otherwise adapt to. Noted as a limitation rather than treated as refuting the claim.

External generalization (PhysioNet Set B)
Model	Set A test MAE	Set B (external) MAE	Gap
ResNet (Pre-binned)	0.0207	0.0203	−2.2%
Neural ODE (Pre-binned)	0.0218	0.0214	−1.8%
ResNet (Real-TS)	0.0196	0.0194	−1.4%
Neural ODE (Real-TS)	0.0202	0.0199	−1.8%
All models generalize cleanly to an entirely unseen patient cohort — no signs of overfitting to Set A for either architecture.

Conclusions
Parameter efficiency: supported. Neural ODE matches ResNet's accuracy within a few percent using roughly one-third the parameters.
Adaptive computation: partially supported. Clearly observed for pre-binned data; not observed for real timestamps under the current rescaled-time integration design.
Irregular-timestamp advantage: not conclusively supported. A single run suggested a large (+12.3%) ODE advantage on real timestamps, but 6-seed replication shows this is within noise (−2.3% ± 6.4%, crossing zero). The defensible claim is that Neural ODE achieves comparable accuracy to a parameter-matched ResNet on irregular data, not a reliably superior one.
Generalization: both architectures generalize consistently to unseen patients.
Limitations / future work
ResNet-4block (Pre-binned) had not fully converged at 60 epochs (17.2% loss drop in the final 5 epochs) — results for that specific configuration may shift with further training.
The Real-TS Neural ODE only uses the total window time span (first-to-last timestamp), not the irregular spacing between the 10 individual readings within a window. A stepwise, ODE-RNN-style design (Rubanova et al., 2019) that integrates between each real observation could show a larger and more stable effect.
6 seeds is a modest sample for the multi-seed test; more seeds would narrow the confidence interval further.
Repository structure
resnet-vs-ode-FIXED.ipynb   — full notebook: data loading, models, training,
                               evaluation, multi-seed test, all figures
README.md                   — this file
Running it
Upload the notebook to Kaggle (or any environment with GPU access). 2. Add the PhysioNet Challenge 2012 Set A dataset as a data source (used here via a public Kaggle mirror; the notebook also downloads Set B directly from physionet.org for external validation). 3. Run all cells top to bottom. Requires torchdiffeq (installed by the first cell).

Data
Uses the PhysioNet Challenge 2012 ICU dataset (Set A and Set B). Raw patient data is not included in this repository — see the link above for access and licensing terms.

Reference
Chen, T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. NeurIPS 2018. arXiv:1806.07366

