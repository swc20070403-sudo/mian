@@TITLE Supplementary material for "Fusing short-pulse voltage waveforms with handcrafted features for rapid state-of-health estimation of lithium iron phosphate batteries: a controlled comparison of encoder pairings"

@@AUTHORS [First Author]^a^, [Second Author]^a^, [Corresponding Author]^a,^\*

@@AFFIL ^a^ [Department, Institution, City Postcode, Country]

@@HEAD Contents

@@TOC Supplementary Note S1. Handcrafted feature definitions

@@TOC Supplementary Note S2. PatchTST encoder equations

@@TOC Supplementary Note S3. Hyperparameter grids of the conventional regressors

@@TOC Supplementary Note S4. Statistical test families

@@TOC Supplementary Note S5. Waveform increment over the GBDT backbone

@@TOC Supplementary Note S6. Re-implementation used for the generalisation, robustness and deployment experiments

@@TOC Supplementary Algorithm S1; Supplementary Tables S1–S7; Supplementary Figs. S1–S5

# Supplementary Note S1. Handcrafted feature definitions

This note defines the 143 waveform-derived features of Table 2; the seven context quantities are given in Eq. (3) of the main text. The segments follow the publisher's pulse timing and are defined, together with the first and second differences, as

$$S^{(1)}=(V_1,\ldots,V_{30}),\quad S^{(2)}=(V_{31},\ldots,V_{40}),\quad S^{(3)}=(V_{41},\ldots,V_{100}),\qquad d^{(1)}_t=V_{t+1}-V_t,\quad d^{(2)}_t=d^{(1)}_{t+1}-d^{(1)}_t$$

{{EQ:S1}}

**Basic operators.** For a window $x$ of length $n$,

$$\overline{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,\quad \sigma(x)=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i-\overline{x})^2},\quad \mathrm{RMS}(x)=\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2},\quad E(x)=\sum_{i=1}^{n}x_i^2$$

{{EQ:S2}}

$$\begin{aligned}&R(x)=\max x-\min x,\qquad \mathrm{IQR}(x)=Q_{0.75}-Q_{0.25},\\&\mathrm{MAD}(x)=\operatorname{median}\left(|x-\operatorname{median}(x)|\right),\qquad \mathrm{AUC}(x)=\sum_{i=1}^{n-1}\frac{x_i+x_{i+1}}{2}\end{aligned}$$

{{EQ:S3}}

The standard deviation uses the population form ($1/n$), except for the global voltage sequence, which follows the publisher's supplementary definition with the sample form ($1/(n-1)$); both are labelled in the feature list.

**Shape.** Bias-corrected Fisher–Pearson skewness and excess kurtosis are used, with $m_k$ the $k$-th central moment:

$$\begin{aligned}&m_k=\frac{1}{n}\sum_{i=1}^{n}(x_i-\overline{x})^k,\qquad g_1=\frac{m_3}{m_2^{3/2}},\qquad g_2=\frac{m_4}{m_2^{2}}-3,\\&G_1=\frac{\sqrt{n(n-1)}}{n-2}g_1,\qquad G_2=\frac{n-1}{(n-2)(n-3)}[(n+1)g_2+6]\end{aligned}$$

{{EQ:S4}}

**Histogram entropy.** The range of the window is divided into 16 equal bins and $p_i$ is the fraction of samples in bin $i$:

$$H(x)=-\frac{1}{\ln 16}\sum_{p_i>0}p_i\ln p_i$$

{{EQ:S5}}

**Linear trend.** A least-squares line is fitted against the sample index $t_i$:

$$a=\frac{\sum_i(x_i-\overline{x})(t_i-\overline{t})}{\sum_i(t_i-\overline{t})^2},\qquad b=\overline{x}-a\overline{t},\qquad R^2=1-\frac{\sum_i(x_i-b-at_i)^2}{\sum_i(x_i-\overline{x})^2}$$

{{EQ:S6}}

**Zero crossings and total variation.** Zero crossings of a difference sequence are counted after removing exact zeros, giving $y$:

$$Z(y)=\text{\#}\{i:y_iy_{i+1}<0\},\qquad \mathrm{TV}(x)=\sum_{i=1}^{n-1}|x_{i+1}-x_i|$$

{{EQ:S7}}

The 15 operators of Eqs. (S2)–(S7) applied to $S^{(1)}$, $S^{(2)}$ and $S^{(3)}$ give the 45 segment statistics; applied to the full sequence and to $d^{(1)}$ and $d^{(2)}$, together with end points and quantiles, they give the 24 global and 22 difference-dynamics features.

**Spectral features.** The demeaned sequence is Hann-windowed and transformed by a real fast Fourier transform ($N=101$, frequency in cycles per sample):

$$\begin{aligned}&w_t=0.5-0.5\cos\left(\frac{2\pi t}{N-1}\right),\qquad x_t=(V_t-\overline{V})\,w_t,\\&X_k=\sum_{t=0}^{N-1}x_t\,e^{-\mathrm{j}2\pi kt/N},\qquad P_k=|X_k|^2,\qquad f_k=\frac{k}{N}\end{aligned}$$

{{EQ:S8}}

The amplitudes of the first eight non-DC lines and the energy shares of four bands are

$$\begin{aligned}&A_k=\frac{2|X_k|}{\sum_t w_t}\ (k=1,\ldots,8),\\&B_1=\frac{\sum_{k=1}^{3}P_k}{\sum_{k=1}^{50}P_k},\ B_2=\frac{\sum_{k=4}^{7}P_k}{\sum_{k=1}^{50}P_k},\ B_3=\frac{\sum_{k=8}^{15}P_k}{\sum_{k=1}^{50}P_k},\ B_4=\frac{\sum_{k=16}^{50}P_k}{\sum_{k=1}^{50}P_k}\end{aligned}$$

{{EQ:S9}}

Spectral-shape features exclude the DC component: dominant frequency and amplitude, centroid, spread, flatness, entropy and 85% roll-off,

$$\begin{aligned}&k^{\star}=\arg\max_{k\ge1}P_k,\qquad f_{\mathrm{dom}}=f_{k^{\star}},\qquad A_{\mathrm{dom}}=A_{k^{\star}},\\&C=\frac{\sum_k f_kP_k}{\sum_kP_k},\qquad S=\sqrt{\frac{\sum_kP_k(f_k-C)^2}{\sum_kP_k}},\qquad F=\frac{\left(\prod_kP_k\right)^{1/K}}{\frac{1}{K}\sum_kP_k}\end{aligned}$$

{{EQ:S10}}

$$H_f=-\frac{1}{\ln 50}\sum_k p_k\ln p_k,\quad p_k=\frac{P_k}{\sum_{\kappa}P_{\kappa}},\quad f_{85}=\min\left\{f_k:\sum_{\kappa\le k}P_{\kappa}\ge0.85\sum_{\kappa}P_{\kappa}\right\}$$

{{EQ:S11}}

**Wavelet features.** A four-level db2 decomposition with symmetric extension gives coefficient sets $c$, and their energy fractions are

$$\varepsilon_c=\sum_i c_i^2,\qquad \mathrm{EF}_c=\frac{\varepsilon_c}{\sum_{c'}\varepsilon_{c'}},\qquad c\in\{\mathrm{cA}_4,\mathrm{cD}_4,\mathrm{cD}_3,\mathrm{cD}_2,\mathrm{cD}_1\}$$

{{EQ:S12}}

Eqs. (S8)–(S12) yield 24 frequency and wavelet features (8 amplitudes, 4 band shares, 7 spectral-shape features and 5 wavelet energy fractions). The dominant frequency takes a single value in the whole dataset and the roll-off frequency takes two values and is constant in the training cells of fold 0; their rank correlations are undefined in the affected folds and they are skipped by the finiteness condition of Section 2.3.3, which is why the feature-count ladder ends at 144 rather than 146.

**Nonlinear features.** Lagged autocorrelation ($\ell\in\{1,5,10\}$) and the three Hjorth parameters are

$$\begin{aligned}&\mathbf a=(V_0,\ldots,V_{100-\ell}),\qquad \mathbf b=(V_{\ell},\ldots,V_{100}),\qquad \rho_{\ell}=\frac{\sum_i(a_i-\overline{a})(b_i-\overline{b})}{\sqrt{\sum_i(a_i-\overline{a})^2\sum_i(b_i-\overline{b})^2}},\\&\mathrm{Act}=\sigma^2(V),\qquad \mathrm{Mob}(V)=\sqrt{\frac{\sigma^2(d^{(1)})}{\sigma^2(V)}},\qquad \mathrm{Com}=\frac{\mathrm{Mob}(d^{(1)})}{\mathrm{Mob}(V)}\end{aligned}$$

{{EQ:S13}}

Approximate and sample entropy use embedding dimension $m=2$ (and $m+1=3$), tolerance $r=0.2\sigma(V)$ and the Chebyshev distance; approximate entropy counts self-matches, whereas sample entropy counts only pairs $i<j$:

$$\begin{aligned}&C_i^m(r)=\frac{\text{\#}\{j:\max_{0\le k<m}|V_{i+k}-V_{j+k}|\le r\}}{N-m+1},\qquad \Phi^m(r)=\frac{1}{N-m+1}\sum_i\ln C_i^m(r),\\&\mathrm{ApEn}=\Phi^2(r)-\Phi^3(r),\qquad \mathrm{SampEn}=-\ln\frac{A}{B}\end{aligned}$$

{{EQ:S14}}

Permutation entropy uses order 3 and delay 1, and the Hurst exponent is the slope of a variogram regression over lags $\ell\in\{1,2,4,8,16\}$:

$$\begin{aligned}&\mathrm{PE}=-\frac{1}{\ln 6}\sum_{\pi\in\Pi_3}p(\pi)\ln p(\pi),\qquad \tau(\ell)=\sqrt{\frac{1}{101-\ell}\sum_i\left(V_{i+\ell}-V_i\right)^2},\\&H=\text{OLS slope of }\ln\tau(\ell)\text{ on }\ln\ell\end{aligned}$$

{{EQ:S15}}

**Transient-boundary features.** Fifteen voltage differences between specified samples, including the steps at current changes ($V_1-V_0$, $V_{31}-V_{30}$, $V_{41}-V_{40}$) and the net changes within segments ($V_{30}-V_1$, $V_{40}-V_{31}$, $V_{100}-V_{41}$), describe the transitions between segments. Three response times are defined on the rest segment $S^{(3)}$, with linear interpolation between samples and $\alpha\in\{0.10, 0.50, 0.90\}$:

$$p_t=\operatorname{sgn}(V_{100}-V_{41})\,(V_t-V_{41}),\quad t=41,\ldots,100,\qquad t_{\alpha}=\min\left\{t:p_t\ge\alpha|V_{100}-V_{41}|\right\}$$

{{EQ:S16}}

# Supplementary Note S2. PatchTST encoder equations

Each patch $\mathbf p_m$ of Eq. (12) is embedded by a shared linear map and a learnable positional embedding $\mathbf e_m$, initialised from $\mathcal N(0,0.02^2)$ and trained with the network:

$$\mathbf z_m^{(0)}=\mathbf W_p\mathbf p_m+\mathbf b_p+\mathbf e_m,\qquad \mathbf W_p\in\mathbb{R}^{D\times P},\qquad \mathbf Z^{(0)}=[\mathbf z_1^{(0)},\ldots,\mathbf z_M^{(0)}]^{\mathrm T}\in\mathbb{R}^{M\times D}$$

{{EQ:S17}}

The $M$ tokens enter one Transformer encoder layer whose multi-head self-attention ($h=4$ heads, $d_k=D/h=8$) acts between patches:

$$\begin{aligned}&\mathbf Q_i=\mathbf Z\mathbf W_i^{Q},\qquad \mathbf K_i=\mathbf Z\mathbf W_i^{K},\qquad \mathbf V_i=\mathbf Z\mathbf W_i^{V},\\&\mathrm{head}_i=\operatorname{softmax}\left(\frac{\mathbf Q_i\mathbf K_i^{\mathrm T}}{\sqrt{d_k}}\right)\mathbf V_i,\qquad \mathrm{MHA}(\mathbf Z)=[\mathrm{head}_1;\ldots;\mathrm{head}_h]\mathbf W^{O}\end{aligned}$$

{{EQ:S18}}

The layer uses post-normalisation residual connections and a GELU feed-forward sublayer of width $2D=64$ with dropout 0.2:

$$\begin{aligned}&\mathbf Z'=\mathrm{LN}\left(\mathbf Z^{(0)}+\mathrm{MHA}(\mathbf Z^{(0)})\right),\qquad \mathbf Z^{(1)}=\mathrm{LN}\left(\mathbf Z'+\mathrm{FFN}(\mathbf Z')\right),\\&\mathrm{FFN}(\mathbf u)=\mathbf W_2\,\mathrm{GELU}(\mathbf W_1\mathbf u+\mathbf b_1)+\mathbf b_2\end{aligned}$$

{{EQ:S19}}

Layer normalisation, mean pooling over tokens and a linear projection give the 32-dimensional waveform embedding:

$$E_{\mathrm v}(\tilde{\mathbf v})=\mathrm{Drop}\left(\mathrm{ReLU}\left(\mathbf W_o\cdot\frac{1}{M}\sum_{m=1}^{M}\mathrm{LN}\left(\mathbf z_m^{(1)}\right)+\mathbf b_o\right)\right)\in\mathbb{R}^{32}$$

{{EQ:S20}}

# Supplementary Note S3. Hyperparameter grids of the conventional regressors

Hyperparameters were selected on the validation cells of each fold by cell-macro RMSE, using the same validation cells as neural early stopping, and the selected models were not refitted on training plus validation cells. Ridge: regularisation coefficient from 10^−4^ to 10^3^. Random forest: 300 trees, maximum feature fraction ∈ {0.33, 1.0}, minimum leaf size ∈ {1, 5, 20}, seeds 0, 1, 2. GBDT (scikit-learn histogram gradient boosting): learning rate ∈ {0.05, 0.1}, maximum leaf nodes ∈ {15, 31}, minimum leaf size ∈ {20, 100}, iterations ∈ {300, 1,000}. RBF-SVR: *C* ∈ {1, 10, 100}, insensitive-band width ε ∈ {0.05, 0.1} on the standardised target, kernel width by the scikit-learn "scale" rule. GPR: product of a constant and an RBF kernel plus a white-noise kernel; kernel hyperparameters were optimised by maximum likelihood on a fixed subsample of 2,000 training records (two random restarts), after which the model was conditioned on all training records. All algorithms except random forest are deterministic.

# Supplementary Note S4. Statistical test families

Holm correction was applied within the following prespecified families at α = 0.05: (1) the 14 encoder pairings versus MLP + CNN; (2) the five waveform complements versus A2; (3) the confirmatory ablations R0 − A1, R0 − A2, R0 − A3 and A3 − A4; (4) the three patch configurations versus R0; (5) the three adaptation ablations versus R0; (6) the six feature-count levels versus *K* = 50, separately for RMSE and MAE; (7) the conventional regressors, information-source controls, reproduced reference model and post hoc configurations versus A2 and versus R0 (30 comparisons); (8) the four residual comparisons on the GBDT backbone; and (9) the two post hoc raw-waveform augmentations. A1 − A2 was a single prespecified comparison without correction. Marginal effects, difference-in-differences, stratified increments, residual complementarity and low-SOH bias are descriptive. Confidence intervals are conditional on the fixed split and the trained models and do not cover the variability of new data collection or repartitioning.

# Supplementary Note S5. Waveform increment over the GBDT backbone

With the GBDT selected in Supplementary Note S3 as the backbone, networks with the structures of R0, A2 and A1 learned its residuals (main text, Section 3.5.4). The waveform increment (G+R0) − (G+A2) was −0.008 pp with 36 of 64 cells improved (Holm *p* = 0.35), and none of the residual learners improved significantly on the GBDT alone. Appending the 101 standardised voltages directly as additional input columns of GBDT (extended grid with up to 63 leaf nodes and 3,000 iterations) and of RBF-SVR, followed by re-tuning, likewise produced no detectable gain. These analyses were defined after part of the GBDT results had been seen and are reported as post hoc.

# Supplementary Note S6. Re-implementation used for the generalisation, robustness and deployment experiments

The experiments of Section 4.6 were run with an independent re-implementation of the protocol written in Python (NumPy, scikit-learn 1.9 and PyTorch 2.14 on CPU, one thread per run), starting from the publisher's processed pulse files [31]. Cleaning reproduced the record counts exactly: 17,130 records in 2,855 groups before cleaning, 192 records removed at the 42nd RPT, and 16,938 records in 2,823 complete groups (SOH 72.34–100%) after cleaning. The 143 waveform features were implemented from the definitions in Supplementary Note S1; details not fixed by those definitions, such as the quantile levels of the global family and the nine voltage differences not listed in Note S1, may differ from the original code, and the response-time levels were set to 0.5, 0.632 and 0.9. Feature selection was nonetheless similar: in fold 0 it selected 16 frequency and wavelet, 14 segment, 11 difference-dynamics, 6 global, 1 boundary and 1 nonlinear feature together with the DCIR, which ranked 13th; 44 features were selected in every fold and 57 in at least one. The split permutation used NumPy's default generator with seed 42, which reproduces the fold sizes but not necessarily the fold membership of the original runs. Architectures, loss, optimiser, scheduler, early stopping and seeds follow Sections 3.2–3.4 of the main text, with positional embeddings initialised from U(−0.02, 0.02).

Under the original five-fold protocol the re-implementation gives R0 1.427, A2 1.474 and GBDT 1.028 pp (reported: 1.377, 1.500 and 1.022 pp). The waveform increment keeps its sign and significance (−0.046 pp, −0.070 to −0.023; *p* < 0.001; 42 of 64 cells), and cells 12 and 3 are again the best- and worst-estimated cells. Table S7 lists all re-implementation results. Among the conventional baselines, ridge regression, RBF-SVR, the *τ*-only controls and the without-*τ* ridge and GBDT were re-run (Table S5); where reported values exist, they agree closely (RBF-SVR 1.007 vs 1.026 pp, without-*τ* GBDT 1.698 vs 1.663 pp, *τ*-only models about 3 pp). Random forest and GPR on the full input and the without-*τ* random forest were not re-run because of their computational and memory cost.

*NMC/graphite dataset.* The UConn-ILCC NMC dataset [31] records the same 100-s pulses at nine nominal SOCs; the three SOCs used for LFP (20%, 50% and 90%) were retained, giving six conditions per RPT, 8,016 records in 1,336 cell–RPT groups of 44 cells and SOH from 41.2% to 100%. Because NMC cells have a higher resistance, the voltage-step and range limits of Eq. (2) were relaxed to 0.3 and 0.8 V; no record was removed. The pipeline was otherwise unchanged. On these data the two neural models perform alike (R0 2.030, A2 2.026 pp; R0 − A2 = +0.004 pp, −0.042 to 0.047; *p* = 0.90), both clearly better than A1 (2.317 pp), and GBDT reaches 0.935 pp; GBDT is more accurate than the neural models in every SOH range.

*Noise not seen in training.* When Gaussian noise is added only to the test voltages of models trained on clean data, all feature-based models deteriorate strongly (σ~noise~ = 0.5 mV: R0 4.37, A2 4.60, GBDT 1.48 pp), because difference- and spectrum-based features computed on the smooth, 0.1-mV-resolution resampled records are sensitive to noise absent from training. Models should therefore be trained on data from the target sensor, or with matching noise augmentation, as in the matched-noise experiment of Section 4.6.

# Supplementary Algorithm S1

@@ALG

# Supplementary Tables

: Supplementary Table S1. Ageing protocols of the cycling-condition groups.

| Group | CC–CV charge rate (C) | CC discharge rate (C) | DOD (%) | Cells (IDs) |
|-----:|----------:|----------:|-------:|:------------|
| 1 | 2 | 1.5 | 60 | 6 (2–7) |
| 2 | 2.42 | 1.5 | 60 | 6 (8–13) |
| 3 | 2 | 2 | 60 | 6 (14–19) |
| 4 | 2 | 1.5 | 80 | 6 (20–25) |
| 5 | 2.83 | 1.5 | 60 | 6 (26–31) |
| 6 | 2.42 | 2 | 60 | 6 (32–37) |
| 7 | 2.42 | 1.5 | 80 | 6 (38–43) |
| 8 | 3.25 | 1.5 | 60 | 6 (44–49) |
| 9 | 2 | 1.5 | 100 | 6 (50–55) |
| 10 | 2 | 2 | 80 | 6 (56–61) |
| 11 | 2.83 | 1.5 | 80 | 4 (62–65) |

TN: From the cell test log in the publisher's public repository [28]. Charging is CC–CV and discharging CC; rates refer to the nominal 1.2 Ah, so 2.42C, 2.83C and 3.25C correspond to 2.9 A, 3.4 A and 3.9 A.

: Supplementary Table S2. Input information of the two branches.

| Input | Dimension | Waveform branch | Backbone branch | Source |
|:----------------------|------:|:-------:|:----------:|:--------------|
| First-sample-referenced voltage sequence | 101 | ✓ | — | Single pulse |
| Waveform-derived features (train-fold selected) | 49 | — | ✓ | Single pulse |
| Matched-condition DCIR (selected) | 1 | — | ✓ | Single pulse |
| Standardised cumulative cycle count *τ* | 1 | — | ✓ | Cycle record |
| Nominal SOC, pulse direction | — | — | Candidate, not selected | Test setting |
| Coulomb-counted SOC, SOC deviation | — | — | Excluded | Requires reference capacity |
| Other condition records, past capacities, cell identity | — | — | — | — |

TN: ✓, input enters the branch; —, not used or not applicable.

: Supplementary Table S3. Ablation configurations and reference models.

| Category | Configuration | Handcrafted backbone | Waveform complement | *τ* | *λ* | Trainings |
|:---------|:------------|:----------------|:------------------|:---:|:---:|:----------|
| Main | R0 | MLP, 50 features | PatchTST (9, 4) | ✓ | 0.1 | 15 |
| Encoder matrix | 15 pairings | MLP / ResNet / FT-Transformer | CNN / MLP / TCN / PW-Trans. / PatchTST | ✓ | 0.1 | 225 (incl. R0) |
| Branch ablation | A2 | MLP | — | ✓ | 0.1 | 15 |
| Branch ablation | A1 | DCIR only (concatenated) | PatchTST (9, 4) | ✓ | 0.1 | 15 |
| Factor ablation | A3 | MLP | PatchTST (9, 4) | ✓ | 0 | 15 |
| Factor ablation | A4 | MLP | PatchTST (9, 4) | — | 0 | 15 |
| Patch parameters | P1, P2, P3 | MLP | PatchTST (5, 2), (5, 4), (9, 2) | ✓ | 0.1 | 45 |
| Adaptation | D1, D2, D3 | MLP | Flatten head, + RevIN, raw voltage | ✓ | 0.1 | 45 |
| Feature count | *K* = 10, 20, 30, 75, 100, 144 | MLP, *K* features | PatchTST (9, 4) | ✓ | 0.1 | 90 (*K* = 50 reuses R0) |
| Conventional | Ridge, RF, GBDT, SVR, GPR | Same 51-dim input as A2 | — | ✓ | — | 1 per fold (RF 3) |
| Information source | *τ* only | *τ* only | — | ✓ | — | 1 per fold (RF 3) |
| Information source | Without *τ* | 50 features | — | — | — | 1 per fold (RF 3) |
| Published model | Nowacki et al. [28], reproduced | — | Raw ΔV, 5 × 100 MLP | — | — | 30 models |
| Strong backbone | G+R0, G+A2, G+A1 | GBDT, network learns residual | As R0, A2, A1 | ✓ | 0.1 | 45 |

TN: Neural configurations use 5 folds × 3 seeds (510 trainings in total, plus 30 reproduced models). Post hoc GBDT/SVR raw-voltage configurations are not listed.

: Supplementary Table S4. Stratified waveform increment (R0 − A2, SOH pp).

| Stratum | Level | A2 RMSE | R0 RMSE | ΔRMSE | 95% CI | Relative change | Improved cells |
|:--------|:----------|------:|------:|-------:|:-------------|--------:|-------:|
| True SOH | <80% | 2.404 | 1.991 | −0.413 | −0.460 to −0.363 | −17.2% | 62/64 |
| True SOH | 80–85% | 1.742 | 1.601 | −0.141 | −0.200 to −0.082 | −8.1% | 47/64 |
| True SOH | 85–90% | 1.974 | 1.861 | −0.113 | −0.151 to −0.074 | −5.7% | 50/64 |
| True SOH | 90–95% | 1.150 | 1.123 | −0.027 | −0.046 to −0.009 | −2.3% | 36/64 |
| True SOH | ≥95% | 0.956 | 0.808 | −0.148 | −0.163 to −0.134 | −15.5% | 61/64 |
| Nominal SOC | 20% | 1.546 | 1.452 | −0.094 | −0.116 to −0.072 | −6.1% | 55/64 |
| Nominal SOC | 50% | 1.550 | 1.414 | −0.136 | −0.161 to −0.111 | −8.8% | 58/64 |
| Nominal SOC | 90% | 1.579 | 1.474 | −0.104 | −0.127 to −0.080 | −6.6% | 56/64 |
| Direction | Charge | 1.546 | 1.434 | −0.112 | −0.136 to −0.088 | −7.2% | 55/64 |
| Direction | Discharge | 1.584 | 1.477 | −0.107 | −0.125 to −0.088 | −6.7% | 57/64 |
| Condition | 20% charge | 1.533 | 1.431 | −0.102 | −0.128 to −0.076 | −6.7% | 47/64 |
| Condition | 20% discharge | 1.556 | 1.471 | −0.085 | −0.109 to −0.061 | −5.5% | 49/64 |
| Condition | 50% charge | 1.512 | 1.380 | −0.132 | −0.161 to −0.104 | −8.7% | 57/64 |
| Condition | 50% discharge | 1.578 | 1.439 | −0.139 | −0.167 to −0.110 | −8.8% | 54/64 |
| Condition | 90% charge | 1.567 | 1.458 | −0.109 | −0.141 to −0.078 | −7.0% | 52/64 |
| Condition | 90% discharge | 1.584 | 1.483 | −0.101 | −0.124 to −0.077 | −6.4% | 54/64 |

TN: True-SOH strata are formed at the cell–RPT level after averaging the six conditions, and each cell's RMSE is computed within the stratum; SOC, direction and condition strata use single records. Predictions are averaged over the three seeds. Descriptive analysis without multiplicity correction.

: Supplementary Table S5. Conventional regressors, information-source controls and reference models (SOH pp).

|Model|Input|Reported|Re-implemented|
|:--------------------------|:----------------------|--------:|------------:|
| R0 (dual-branch) | Features + *τ* + ΔV | 1.377 | 1.427 |
| A2 (backbone only) | Features + *τ* | 1.500 | 1.474 |
| Ridge | Features + *τ* | n.r. | 1.879 |
| Random forest | Features + *τ* | n.r. | — |
| GBDT | Features + *τ* | 1.022 | 1.028 |
| RBF-SVR | Features + *τ* | 1.026 | 1.007 |
| GPR | Features + *τ* | n.r. | — |
| Ridge | *τ* only | ≈3^a^ | 3.083 |
| Random forest | *τ* only | ≈3^a^ | 3.095 |
| GBDT | *τ* only | ≈3^a^ | 2.881 |
| Ridge | Features without *τ* | n.r. | 2.593 |
| Random forest | Features without *τ* | n.r. | — |
| GBDT | Features without *τ* | 1.663 | 1.698 |
| Nowacki et al. [28], reproduced | Raw ΔV | 1.628 | — |
| GBDT + raw voltage (post hoc) | Features + *τ* + 101 voltages | no gain^b^ | — |
| RBF-SVR + raw voltage (post hoc) | Features + *τ* + 101 voltages | no gain^b^ | — |

TN: Cell-macro RMSE over the 64 test cells. "Reported" values are those of the original runs on which the main text is based; n.r., not given in the source draft (to be inserted from the original result files). "Re-implemented" values come from the CPU re-implementation of Supplementary Note S6 with the same grids (random forest averaged over seeds 0–2); —, not run in the re-implementation. ^a^ About 3 pp in the original runs. ^b^ No detectable gain over GBDT and RBF-SVR, respectively (Supplementary Note S5).

: Supplementary Table S6. Sensitivity to the number of handcrafted features (SOH pp).

| *K* | Backbone input | Parameters | RMSE | MAE | ΔRMSE | 95% CI | Holm *p* | Improved / worse cells |
|---:|------:|------:|-----:|-----:|-------:|:------------|------:|:--------|
| 10 | 11 | 16,481 | 1.516 | 1.249 | +0.139 | 0.089–0.188 | 0.001 | 16/48 |
| 20 | 21 | 17,121 | 1.499 | 1.226 | +0.122 | 0.077–0.169 | 0.001 | 14/50 |
| 30 | 31 | 17,761 | 1.468 | 1.210 | +0.091 | 0.057–0.125 | 0.001 | 14/50 |
| 50 (R0) | 51 | 19,041 | 1.377 | 1.141 | — | — | — | — |
| 75 | 76 | 20,641 | 1.414 | 1.165 | +0.037 | 0.018–0.057 | 0.002 | 24/40 |
| 100 | 101 | 22,241 | 1.418 | 1.176 | +0.041 | 0.021–0.062 | 0.002 | 15/49 |
| 144 | 145 | 25,057 | 1.376 | 1.128 | −0.002 | −0.019 to 0.015 | 0.838 | 32/32 |

TN: *K* excludes the separately appended cycle count, so the backbone input has *K* + 1 dimensions. ΔRMSE is the level minus *K* = 50; Holm correction covers the six RMSE comparisons. Improved (worse) cells have lower (higher) error than *K* = 50.

: Supplementary Table S7. Results of the re-implementation (cell-macro RMSE, SOH pp).

|Scenario|Data|R0|A2|A1|GBDT|R0 − A2|95% CI|*p*|Improved cells|
|:-----------------|:----|-----:|-----:|-----:|-----:|-------:|:-------------|-----:|--------:|
| Five-fold, reported (main text) | LFP | 1.377 | 1.500 | 1.494 | 1.022 | −0.123 | −0.141 to −0.103 | <0.001 | 58/64 |
| Five-fold, re-implementation | LFP | 1.427 | 1.474 | — | 1.028 | −0.046 | −0.070 to −0.023 | <0.001 | 42/64 |
| Leave-one-group-out | LFP | 1.560 | 1.594 | — | 1.401 | −0.035 | −0.053 to −0.015 | 0.008 | 39/64 |
| Cycle count +10% (test) | LFP | 1.449 | 1.497 | — | 1.117 | −0.048 | −0.072 to −0.024 | <0.001 | 43/64 |
| Cycle count −10% (test) | LFP | 1.477 | 1.518 | — | 1.176 | −0.041 | −0.063 to −0.018 | 0.002 | 39/64 |
| Matched noise, σ~noise~ = 1 mV | LFP | 1.620 | 1.678 | — | 1.372 | −0.058 | −0.077 to −0.040 | <0.001 | 50/64 |
| Matched noise, σ~noise~ = 2 mV | LFP | 1.715 | 1.760 | — | 1.424 | −0.045 | −0.069 to −0.022 | 0.002 | 46/64 |
| Test-only noise, σ~noise~ = 0.5 mV^a^ | LFP | 4.374 | 4.595 | — | 1.485 | — | — | — | — |
| Five-fold cell-disjoint | NMC | 2.030 | 2.026 | 2.317 | 0.935 | +0.004 | −0.042 to 0.047 | 0.90 | 21/44 |

TN: Paired comparisons use 5,000 fold- (or group-) stratified cell bootstraps and two-sided sign-flip tests, without multiplicity correction. ^a^ Models trained on clean data with noise added to the test voltages only. Group-level leave-one-group-out errors are shown in Fig. 8a of the main text.

# Supplementary Figures

![**Fig. S1.** Cumulative error distributions of the waveform-branch design variants. (a–c) Patch configurations P1 (5, 2), P2 (5, 4) and P3 (9, 2) versus R0 (9, 4); (d–f) adaptation ablations D1 (flatten head), D2 (with RevIN) and D3 (raw voltage). Solid blue, R0; dashed brown, variant. Each curve contains the 192 cell × seed RMSEs (not 192 independent cells); insets magnify RMSE 0.8–1.8 pp and cumulative proportion 15–75%. Cell-level tests are given in Table 8.](figures/figS1_patch_cdf.png){width=6.3in}

![**Fig. S2.** Sensitivity to the number of handcrafted features *K*. Difference in cell-macro RMSE relative to *K* = 50 with 95% fold-stratified cell bootstrap intervals; labels give the total parameter count. Hollow markers denote intervals that include zero.](figures/figS2_K.png){width=4.6in}

![**Fig. S3.** SOH-dependent mean bias and overall residual distribution of R0 (blue, solid) and GBDT (teal, dashed). Residuals are the seed-averaged prediction minus the measured SOH of each record. Left: 1-pp bins of measured SOH with at least 20 records; shading marks SOH < 80% and ≥ 95%; the inset magnifies 89–97% SOH. Right: kernel density of all 16,938 residuals, normalised to the common maximum.](figures/figS3_bias_soh.png){width=6.3in}

![**Fig. S4.** Record-level predictions and residuals of representative cells 12, 6, 44 and 3 (ranks 1, 22, 43 and 64 by R0 error). Each RPT contributes 18 records (three SOCs × two directions × three seeds), so the horizontal axis is the record index rather than the cycle count; the grey staircase repeats the measured SOH of each RPT. Blue, R0; orange dashed, A2.](figures/figS4_cells.png){width=6.3in}

![**Fig. S5.** SOH estimated by R0 versus measured SOH for the 11 cycling-condition groups. Each point is one cell–RPT (predictions averaged over seeds and conditions, all from the cell's test fold); colour denotes absolute error (saturating at 3 pp) and the red dashed line is *y* = *x*. The number in each panel is the group's cell-macro RMSE following Eq. (14).](figures/figS5_groups.png){width=6.3in}


