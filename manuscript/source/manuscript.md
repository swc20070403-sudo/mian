@@TITLE Fusing short-pulse voltage waveforms with handcrafted features for rapid state-of-health estimation of lithium iron phosphate batteries: a controlled comparison of encoder pairings

@@AUTHORS [First Author]^a^, [Second Author]^a^, [Corresponding Author]^a,^\*

@@AFFIL ^a^ [Department, Institution, City Postcode, Country]

@@AFFIL \* Corresponding author. E-mail address: [ ]

@@HEAD Highlights

@@HL Fifteen encoder pairings are benchmarked under one cell-disjoint protocol.

@@HL A patch-based waveform branch cuts the MLP-backbone error by 8.2% in 58/64 cells.

@@HL Encoder rankings depend on the pairing rather than on architectural complexity.

@@HL The waveform gain persists for unseen cycling conditions and matched sensor noise.

@@HL Three charge pulses and a 19k-parameter model suffice for rapid SOH estimation.

@@HEAD Abstract

@@ABS Short current pulses offer a rapid diagnostic for the state of health (SOH) of lithium-ion batteries, but whether a learned waveform representation adds information to handcrafted pulse features has not been tested under a controlled protocol. We build a dual-branch model in which handcrafted features and the cumulative cycle count form the backbone and the 101-point pulse-voltage response forms a complementary branch, and evaluate all 15 pairings of three backbone and five waveform encoders on 16,938 pulse records from 64 public lithium iron phosphate (LFP)/graphite cells with five-fold cell-disjoint cross-validation. Among the pairings, a multilayer perceptron (MLP) backbone with a patch-based Transformer (PatchTST) complement attains the lowest cell-level root-mean-square error, 1.377 SOH percentage points (95% CI 1.247–1.513). Relative to the same backbone without the waveform, the error falls by 8.2% and 58 of 64 cells improve, with the largest gain (17.2%) below 80% SOH, where the waveform partly corrects the shrinkage bias of the backbone. Encoder rankings depend on the pairing: PatchTST ranks first with the MLP backbone but last with the other two, and the pairing term explains up to 47.6% of the between-pairing variation. In further tests, the waveform gain persists for held-out cycling conditions and under matched sensor noise, ±10% cycle-count errors raise the error by at most 3.5%, three charge pulses match the six-pulse protocol, and inference needs 19,041 parameters and 0.42 MFLOPs per pulse. Encoders for pulse-based SOH estimation should therefore be selected as pairings on controlled evidence rather than by architectural complexity.

@@KEY **Keywords:** Lithium iron phosphate battery; State of health; Short current pulse; Feature fusion; Patch-based Transformer; Cell-disjoint validation

@@HEAD Nomenclature

| Abbreviation | Definition | Symbol | Definition |
|:---------|:------------------------|:------------|:--------------------------|
| BMS | Battery management system | $E_{\mathrm h}$, $E_{\mathrm v}$ | Backbone and complement encoders |
| CNN | Convolutional neural network | $g$ | Regression head |
| DCIR | Direct-current internal resistance | $K$ | Number of selected handcrafted features |
| DOD | Depth of discharge | $L$, $P$, $S$ | Sequence length, patch length, patch stride |
| GBDT | Gradient-boosting decision tree | $M$ | Number of patches |
| GPR | Gaussian process regression | $Q_{ir}$ | Discharge capacity of cell $i$ at RPT $r$ |
| LFP | Lithium iron phosphate | $y_{ir}$ | Measured SOH (%) |
| MAE | Mean absolute error | $\hat y$, $\hat y^{\text{*}}$ | Predicted SOH and its standardised value |
| MLP | Multilayer perceptron | $\Delta v_t$, $\tilde v_t$ | Referenced and standardised pulse voltage |
| NMC | Nickel–manganese–cobalt oxide | $\mathbf h$ | Backbone input vector |
| PW-Trans. | Point-wise Transformer | $n_{i,r}$, $\tau$ | Cumulative cycle count and its standardised value |
| RBF-SVR | Radial-basis-function support vector regression | $\lambda$ | Weight of the monotonicity penalty |
| RevIN | Reversible instance normalisation | $\rho_j$ | Spearman correlation of feature $j$ with SOH |
| RMSE | Root-mean-square error | $\sigma_n$ | Standard deviation of added voltage noise |
| RPT | Reference performance test | R0 | Reference configuration (MLP + PatchTST) |
| SOC, SOH | State of charge, state of health | A1–A4, D1–D3, P1–P3 | Ablation configurations (Section 3.5.1) |
| TCN | Temporal convolutional network | pp | Percentage points of SOH |

# 1. Introduction

Lithium-ion batteries in electric vehicles and stationary storage degrade throughout service, and their state of health (SOH) governs estimates of available energy and power, maintenance scheduling and second-life decisions [1–3]. Capacity-based SOH, the ratio of present to initial capacity, requires a controlled full charge–discharge that takes hours and interrupts operation [4]; even multi-year field studies rely on periodic capacity tests as ground truth [5]. Faster surrogate signals therefore have to be used, but each carries its own prerequisites [1,6]. Incremental-capacity and differential-voltage analyses resolve degradation modes but need controlled, complete cycles [6], and their peaks are blurred by sensor noise and by the flat voltage plateau of lithium iron phosphate (LFP) cells [7]. Partial-charging methods require the charge to traverse a specific voltage window [8], relaxation methods require long rests at defined states [4], and electrochemical impedance spectroscopy (EIS), although informative [9], needs dedicated excitation hardware and a relaxed cell [10,11]. A diagnostic that completes within seconds to minutes, needs only a controlled current and still carries capacity information is therefore desirable.

Short current pulses meet these requirements. A pulse is a time-domain impedance measurement: over a pulse lasting seconds, electron transfer, ion transfer and solid-state diffusion all shape the voltage response, and pulse- and EIS-derived resistances agree when their time scales are matched [12]. Pulse responses can even be used to reconstruct impedance spectra in a fraction of the EIS test time [13]. Loss of lithium inventory, loss of active material and interphase growth all alter these processes [14], so the response evolves with ageing, and the pulse resistance itself defines power capability and is a standard SOH indicator [12,15]. Pulses are also routine in battery testing: standard power characterisation is pulse-based [15], and pulse injection during vehicle charging has been demonstrated on cells [16]. Whereas a full hybrid pulse power characterisation (HPPC) sequence can exceed 12 h [17], a single pulse lasts seconds to minutes and passes little charge. For LFP cells the response is dominated by kinetic overpotentials, so its change with ageing can remain measurable even where the equilibrium voltage barely moves, making pulses particularly attractive for this chemistry.

Deploying data-driven SOH models in battery management systems (BMSs) adds engineering constraints: inference must run in real time on resource-limited hardware, and black-box behaviour undermines trust in safety-relevant decisions [18,19]. Reviews list the lack of common standards, poor feature choice, limited data and limited interpretability as the main obstacles [20]. Existing methods fall into two families [21,22]. The first extracts handcrafted health indicators from voltage, current, temperature or incremental-capacity curves and feeds them to a regressor [3]; these descriptors are physically interpretable and suit lightweight models [21], but they depend on expert design and transfer poorly across operating conditions [22]. The second learns representations directly from raw sequences; it avoids feature design but is harder to interpret [18], and its accuracy depends on how well the architecture fits the task [22]. Multi-branch models that fuse several representations have been proposed [23–25], yet their branch encoders are fixed at design time and the reported comparisons target baselines and fusion schemes [24,25] rather than the encoders themselves. More broadly, most studies emphasise algorithmic novelty over a systematic examination of health indicators [18], comparisons under consistent protocols and rigorous cross-validation remain scarce [26], and some benchmarks consider only a single input type [27].

For short-pulse SOH estimation specifically, existing studies use a single representation: the raw pulse response [16,28] or features extracted from it [17,29,30], and transfer across states of charge (SOC) has also been examined [31]. These works establish that short pulses carry SOH information, but leave three questions open for models that use both representations. First, what is the main effect of each encoder type, for example whether a more elaborate handcrafted-feature encoder helps? Second, does the merit of a waveform encoder depend on the handcrafted encoder it is paired with? Comparisons that replace only one branch cannot reveal such pairing dependence. Third, because inputs [32–34], data splits and preprocessing [35,36] and training settings [26] differ between studies and no common benchmark exists [37], reported differences cannot be attributed to the encoders. The answers matter for model selection: if more complex encoders bring no benefit, lightweight structures should be preferred for resource-limited BMSs.

Handcrafted features are a natural backbone for this task. Health indicators extracted from charge–discharge curves, incremental-capacity curves or pulse responses are widely used [3,17,29,30], they are interpretable and inexpensive [21], and context that the raw waveform lacks, such as usage history, can be added as explicit features. Handcrafted features nevertheless compress the response: each describes one predefined property, their choice depends on experience [22], and because they are computed item by item they struggle to capture joint changes across segments of the transient. For LFP cells, capacity-related changes in the response may be subtle and distributed over the whole transient. A learned encoder can recover such shape information, but on its own it is less interpretable [18] and blind to usage history, so it is better used as a complement than as a replacement. Previous fusion studies combined signals from different measurements [42,43] or different transforms of one signal [24,25] without distinguishing primary and complementary branches [23–25]; the increment of a waveform branch over a handcrafted backbone derived from the same pulse has rarely been tested. Patch-based encoders such as PatchTST tokenise contiguous segments and relate them through self-attention [40], which could complement item-wise features, but they were designed for long-horizon forecasting [40] and have been applied to batteries only for SOH estimation, lifetime prediction and voltage forecasting [44–46]; adapting them to single-value regression from one short pulse remains to be done.

This work addresses these gaps with a dual-branch model in which handcrafted features and the cumulative cycle count form the backbone and the pulse-voltage waveform forms the complement. Unlike approaches built on degradation dynamics, mechanistic features or digital twins [47–49], a soft monotonicity penalty with respect to the cycle count introduces the prior that capacity fades with use without requiring a mechanistic model. The contributions are threefold:

(1) A complete 3 × 5 matrix of handcrafted-backbone and waveform-complement encoders is evaluated under one protocol with identical splits, preprocessing and training budgets, separating the main effects of each encoder type from their pairing dependence.

(2) A dual-branch model with a PatchTST encoder adapted to single-value regression from one short pulse is developed, and ablations, cell-level paired statistics and strong conventional learners quantify the contribution of each design factor and the conditions under which the waveform complement helps.

(3) The scope of the model is probed on held-out cycling conditions, under cycle-count errors and sensor noise, with reduced pulse protocols, and in terms of attribution and computational cost; an independent nickel–manganese–cobalt (NMC)/graphite dataset recorded with the same pulse protocol is also examined.

# 2. Data and unified protocol

All models share one data protocol, so that error differences between encoders can be attributed to architecture: the cleaning rules, cell split, input policy of both branches and the train-fold feature selection and scaling are identical across the 15 encoder pairings and all supplementary experiments (Table 1).

## 2.1. Dataset, pulse protocol and SOH definition

The public UConn-ISU-ILCC LFP/Gr battery ageing dataset [28] contains 64 LFP/graphite power cells (nominal capacity 1.2 Ah) aged under 11 cycling conditions that differ in constant-current–constant-voltage (CC–CV) charge rate (2C–3.25C), constant-current discharge rate (1.5C or 2C) and depth of discharge (DOD, 60–100%) (Supplementary Table S1). Cycling was interrupted roughly every 100 cycles for a reference performance test (RPT) that measured the remaining capacity and applied short pulses [28,31]. All ageing tests and RPTs were performed by the data publisher; this study reanalyses the data.

Each RPT applies one charge and one discharge pulse at each of three SOCs. A pulse consists of 30 s at 0.2C, 10 s at 1C and a 60 s rest (Fig. 1a) [31]; it lasts 100 s and passes about 0.44% of the nominal capacity. Pulses are triggered at preset voltage cut-offs, so 20%, 50% and 90% are nominal SOCs at beginning of life and the actual SOC drifts as capacity fades [31]. Each RPT therefore yields records for six conditions (three nominal SOCs × two directions). Within the 1 Hz record, the voltage step at each current change mainly reflects ohmic resistance and fast charge transfer, the continued change during constant current adds diffusion, and the slow recovery during rest is dominated by diffusion relaxation [12] (Fig. 1b,c). The voltage difference between the ends of the 0.2C and 1C segments gives the 10-s direct-current internal resistance (DCIR) used in Section 2.3.2.

SOH is normalised by the discharge capacity at the first RPT of the same cell:

$$y_{ir}=100\times\frac{Q_{ir}}{Q_{i0}}$$

{{EQ:1}}

where $Q_{ir}$ is the discharge capacity of cell $i$ at RPT $r$. SOH thus denotes capacity retention relative to the initial measurement rather than to the rated capacity. The capacity comes from the full reference discharge of the RPT, i.e. the hour-long test that pulse diagnostics aim to replace; it serves only as the label, and no quantity derived from it enters the model (Section 2.3.2).

Each record contains 101 voltage samples: sample 0 is the last voltage before current is applied, and the 30, 10 and 60 s segments were resampled to 1 Hz by piecewise cubic Hermite interpolation in the publisher's processing code [31]. Interpolated points are not treated as independent samples. Each inference uses one pulse record, i.e. one 100-s test. The primary metric averages the six single-record predictions of an RPT, which corresponds to a joint diagnosis at three nominal SOCs whose duration also includes bringing the cell to each SOC; single-condition errors are reported separately (Section 4.3).

![**Fig. 1.** Dataset and pulse response. (a) Programmed current of the charge (solid) and discharge (dashed) pulses: 30 s at C/5, 10 s at 1C and 60 s rest. (b, c) Mean voltage response ΔV relative to the pre-pulse voltage at 50% nominal SOC, grouped by measured SOH (darker = lower SOH); insets magnify the end of the 1C segment and the onset of relaxation, where the overpotential grows with ageing. (d) Measured SOH of the 64 cells versus cumulative cycle count, coloured by depth of discharge; lines connect measured RPTs without interpolation.](figures/fig1_dataset.png){width=6.5in}

## 2.2. Data cleaning and cell-disjoint split

Before cleaning, the data contain 17,130 records in 2,855 cell–RPT groups. A record is retained only if it passes all predefined checks:

$$\mathcal{A}=\left\{(V,x,y):\begin{aligned}&V_t\in[2.5,4.5]\ \mathrm{V}\ \forall t;\ \ \max_t V_t-\min_t V_t\in[0.005,0.5]\ \mathrm{V};\ \ \max_t|V_{t+1}-V_t|\le0.1\ \mathrm{V}\\&y\in[0,110];\ \ q_{\mathrm{miss}}\le0.20;\ \ |G_{i,r}|=6;\ \ \max_{g\in G_{i,r}}y_g-\min_{g\in G_{i,r}}y_g\le10^{-5}\end{aligned}\right\}$$

{{EQ:2}}

where $V_t$ is the $t$-th voltage sample, $y$ the SOH in percent, $q_{\mathrm{miss}}$ the fraction of missing handcrafted features and $G_{i,r}$ the six condition records of cell $i$ at RPT $r$. All 192 removed records failed because their cycle count was not finite: the publisher's data lack the cycle count at the 42nd RPT, which affects all 32 cells that reached it (32 complete cell–RPT groups). The remaining 16,938 records form 2,823 complete groups covering all 64 cells, with SOH from 72.34% to 100% (Fig. 1d). Thresholds were fixed in advance and logged, not tuned on test errors.

Five outer test folds were built with split seed 42: the 64 cell identifiers were permuted and divided into five disjoint test sets. Within the non-test cells of each fold, validation cells were drawn with the fold index as offset and a stride of five, and the remaining cells were used for training, giving 40/11/13, 41/10/13, 41/10/13, 41/10/13 and 42/10/12 training/validation/test cells. All RPTs and conditions of a cell are assigned together, because random record-level splits inflate accuracy [36]. The split is identical for all experiments, and the MLP backbone with the PatchTST complement serves as the reference configuration R0.

: Data, inputs and experimental design.

| Item | Setting |
|:--------|:------------------------------|
| Data | Public UConn-ISU-ILCC LFP/Gr dataset; 64 cells, 16,938 records, 2,823 cell–RPT groups |
| Pulse conditions | Nominal SOC 20%, 50%, 90% × charge, discharge; 100 s per pulse |
| Waveform branch | 101-point ΔV, per-position train-fold standardisation |
| Backbone branch | 50 train-fold-selected features (49 waveform-derived + DCIR) + cumulative cycle count *τ* |
| Excluded inputs | Coulomb-counted SOC and its deviation (derived from reference capacity); other records of the same RPT; past capacities; cell identity |
| Validation | Five cell-disjoint outer folds; initialisation seeds 0, 1, 2 |
| Encoder matrix | 5 waveform × 3 backbone encoders × 5 folds × 3 seeds = 225 trainings |
| Further experiments | 19 neural configurations × 15 = 285 trainings; 30 reproduced reference models; conventional regressors (Section 3.5) |
| Primary metric | Cell-macro RMSE after averaging the six condition predictions of each RPT |

## 2.3. Handcrafted backbone inputs

### 2.3.1. Candidate feature families

The handcrafted branch is the backbone of the model. Of 150 candidate features, 143 are computed from the voltage of a single pulse and 7 describe condition and age (Table 2). Waveform-derived features follow the publisher's pulse timing and act on the 30-s low-current segment, the 10-s high-current segment, the 60-s rest and the full sequence. They form six families, each feature having a fixed window and formula (Supplementary Note S1). All are computed item by item: transient-boundary features span adjacent segments but compare only predefined sample pairs and do not represent joint shape changes across the transient, which motivates the waveform complement (Section 2.4).

: Candidate handcrafted feature families and their selection on the training cells of fold 0.

| Family | Candidates | Window | Content | Selected |
|:------------|------:|:------------|:--------------------|------:|
| Segment statistics | 45 | 30 s, 10 s, 60 s segments | Level, dispersion, shape, entropy, linear trend | 9 |
| Global waveform | 24 | Full sequence | Global statistics, end points, quantiles | 7 |
| Difference dynamics | 22 | First and second differences | Rate of voltage change and its change | 14 |
| Transient boundary | 18 | Predefined sample pairs | Steps at current changes, net changes, response times | 2 |
| Frequency and wavelet | 24 | Demeaned, windowed sequence | Spectral amplitudes, band energies, spectral shape, wavelet energies | 16 |
| Nonlinear complexity | 10 | Full sequence | Autocorrelation, Hjorth parameters, entropies, Hurst exponent | 1 |
| Condition and age context | 7 | Record metadata | Nominal and coulomb-counted SOC, deviation, direction, DCIR, cycle count, RPT index | 1^a^ |
| Total | 150 | | | 50 |

TN: ^a^ Matched-condition DCIR. Across the five folds, 47 features were selected in every fold and 54 in at least one. Coulomb-counted SOC, SOC deviation, cumulative cycle count and RPT index do not enter selection (Section 2.3.2).

### 2.3.2. Condition context and cumulative cycle count

A single pulse carries no record of cumulative use, and its starting SOC drifts with ageing. The backbone therefore considers seven context candidates,

$$[\frac{\mathrm{SOC_{nom}}}{100},\ \frac{\mathrm{SOC_{coul}}}{100},\ \frac{\mathrm{SOC_{coul}}-\mathrm{SOC_{nom}}}{100},\ \pm1,\ \mathrm{DCIR}_{\mathrm{pulse,SOC}},\ n_{i,r},\ r]$$

{{EQ:3}}

namely nominal SOC, coulomb-counted SOC, their deviation, pulse direction, matched-condition DCIR, cumulative cycle count and RPT index. The publisher computes the DCIR from the voltage difference between the two current levels of the same pulse, $R=|V_{40}-V_{30}|/(1.2\ \mathrm{A}-0.24\ \mathrm{A})$ [31], where $V_{30}$ and $V_{40}$ are the last samples of the 0.2C and 1C segments. It depends only on the pulse, represents the power-related resistance discussed in Section 1, and enters feature selection.

Coulomb-counted SOC and its deviation from the nominal SOC are excluded. The publisher's code divides the charge accumulated at the pulse start by the capacity measured in the reference discharge of the same RPT [31], which is the numerator of the SOH label in Eq. (1); both quantities are therefore label-derived and unavailable without a full capacity test. Nominal SOC and pulse direction remain candidates, but since every RPT contains all six conditions their rank correlation with SOH is zero and they were never selected; the direction is also evident from the sign of the waveform input.

The cumulative cycle count does not enter selection; it is standardised separately as the input $\tau$ (Section 2.5), and the RPT index is removed to avoid duplicating age information. The count $n_{i,r}$ is accumulated from the cycle intervals between successive RPTs. Because the 42nd-RPT interval is missing, the 32 affected cells undercount one interval (67–118 cycles, median 85, i.e. 1.3–2.3% of life) at their 494 subsequent cell–RPT points; Fig. 1d and the model use the same rule. Using $\tau$ requires the cell's cycle record, which defines the application scope of the model.

### 2.3.3. Train-fold feature selection

Removing cumulative cycle count, RPT index and the two label-derived SOC quantities leaves 146 selectable features. In each fold, the Spearman rank correlation between each feature and SOH is computed on the training records only,

$$\rho_j=\frac{\sum_i(r_{ij}-\overline{r}_j)(r_i^{y}-\overline{r}^{y})}{\sqrt{\sum_i(r_{ij}-\overline{r}_j)^2\sum_i(r_i^{y}-\overline{r}^{y})^2}}$$

{{EQ:4}}

where $r_{ij}$ and $r_i^{y}$ are the ranks of feature $j$ and of the label. Features are taken greedily in descending $|\rho_j|$, skipping any candidate whose absolute rank correlation with an already selected feature reaches 0.98:

$$\mathcal{S}\leftarrow\mathcal{S}\cup\{j\}\quad\text{if and only if}\quad\max_{i\in\mathcal{S}}\left|\rho^{\mathrm{rank}}_{ji}\right|<0.98\ \ \text{and}\ \ |\mathcal{S}|<K$$

{{EQ:5}}

If fewer than $K$ features pass, the remaining finite-correlation features are added in descending $|\rho_j|$ without the redundancy constraint. Each fold selected 50 features, 47 of them in all five folds (54 in the union), so the selection is stable across training cells. In fold 0, frequency and wavelet (16) and difference-dynamics features (14) dominate (Table 2), and the DCIR ranks 13th–14th in every fold. The dominant frequency and the 85% spectral roll-off are constant within the training cells of some folds and are skipped, so at most 144 features can be ranked. The choice $K=50$ is justified in Section 4.5.3.

## 2.4. Waveform complement input

The waveform branch receives only the voltage sequence of the pulse, without condition or age information. The sequence $V_0,\ldots,V_{100}$ is referenced to its first sample,

$$\Delta v_t=V_t-V_0,\quad t=0,1,\ldots,100$$

{{EQ:6}}

which removes the starting voltage so that the sequence represents the pulse-induced polarisation, the kinetically controlled part that remains measurable on the flat LFP plateau. Each position is then standardised with training-fold statistics,

$$\tilde v_t=\frac{\Delta v_t-\mu_t^{\mathrm{w}}}{\sigma_t^{\mathrm{w}}}$$

{{EQ:7}}

so the full 101-point transient is retained for the waveform encoder to learn inter-segment shape information not covered by the handcrafted features.

## 2.5. Imputation, scaling and input audit

Missing values are imputed by the training-fold median of finite values and then standardised:

$$m_j=\operatorname{median}\{x_{ij}:x_{ij}\in\mathbb{R}\},\qquad \tilde x_{ij}=\begin{cases}x_{ij}, & x_{ij}\in\mathbb{R}\\ m_j, & x_{ij}\notin\mathbb{R}\end{cases},\qquad z_{ij}=\frac{\tilde x_{ij}-\mu_j}{\sigma_j}$$

{{EQ:8}}

The cumulative cycle count is standardised and appended to the $K$ selected features,

$$n_{i,r}=\sum_{q\le r}\Delta c_{i,q},\qquad \tau=\frac{n_{i,r}-\mu_n}{\sigma_n},\qquad \mathbf h=[z_1,\ldots,z_K,\tau]^{\mathrm T}\in\mathbb{R}^{K+1}$$

{{EQ:9}}

giving a 51-dimensional backbone input for $K=50$. Labels are divided by 100 and standardised, and predictions are transformed back:

$$y^{\text{*}}=\frac{y/100-\mu_y}{\sigma_y},\qquad \hat y=100\left(\sigma_y\hat y^{\text{*}}+\mu_y\right)$$

{{EQ:10}}

All statistics in Eqs. (4), (5) and (7)–(10) are fitted on the training fold only. Apart from the cycle count, every input is obtained from the single pulse; other condition records of the same RPT, past capacities, cell identity and reference-capacity-derived quantities never enter the model (Supplementary Table S2).

# 3. Model and experimental design

## 3.1. Framework

The model consists of a handcrafted backbone encoder $E_{\mathrm h}$, a waveform complement encoder $E_{\mathrm v}$ and a regression head $g$ (Fig. 2). The backbone receives the 51-dimensional input $\mathbf h$ of Eq. (9) and the complement receives $\tilde{\mathbf v}$ of Eq. (7); both are derived from the same pulse record, and $\tau$ additionally needs the cycle record. Each encoder outputs a 32-dimensional embedding, and the concatenation is mapped to the standardised SOH:

$$\hat y^{\text{*}}=g\left([E_{\mathrm v}(\tilde{\mathbf v});\,E_{\mathrm h}(\mathbf h)]\right)$$

{{EQ:11}}

The two branches differ in information: the backbone carries pulse-derived physical descriptors such as the DCIR together with usage information and can estimate SOH on its own, whereas the complement sees only the shape of the response. The complement is therefore evaluated by its increment over the backbone (Section 4.2). In the encoder matrix, three backbone and five complement candidates are crossed while everything else is held fixed.

![**Fig. 2.** Dual-branch SOH estimation model (configuration R0). Top left: handcrafted backbone; 50 train-fold-selected features and the standardised cumulative cycle count *τ* form a 51-dimensional input encoded by an MLP into a 32-dimensional embedding. Bottom left: waveform complement; the 101-point pulse response is referenced to its first sample, standardised per position and split into 24 patches (length *P* = 9, stride *S* = 4) that a single-layer PatchTST encodes into a 32-dimensional embedding. Right: the concatenated 64-dimensional embedding is regressed to the standardised SOH *ŷ*\* and inverse-transformed. Training minimises the mean squared error plus a penalty on positive ∂*ŷ*\*/∂*τ* (*λ* = 0.1); the soft constraint acts only during training.](figures/fig2_framework.png){width=6.5in}

## 3.2. Backbone encoders

Three backbone encoders are compared. The MLP maps the 51 inputs to 64 and then 32 units, each layer followed by layer normalisation and ReLU, with dropout 0.2 after the first layer (5,600 parameters). ResNet and FT-Transformer follow their tabular reference implementations [41], adjusted to a 32-dimensional output at a similar budget: ResNet uses two residual blocks of width 28 (5,800 parameters), and FT-Transformer tokenises each input to width 20 and uses one Transformer block with four heads and a ReGLU feed-forward layer of hidden width 26 (6,124 parameters), both with dropout 0.2. All three receive the same input; $\tau$ is passed as a separate tensor and concatenated so that the partial derivative of the prediction with respect to $\tau$ can be computed (Section 3.4).

## 3.3. Waveform encoders and PatchTST adaptation

Five complement encoders map the 101-point standardised sequence to a 32-dimensional embedding, each ending with a linear projection, ReLU and dropout 0.2 (Table 3). The convolutional neural network (CNN) has three convolutional layers (channels 1→16→32→64, kernels 7, 5, 3, each with batch normalisation and ReLU, max pooling after the first two) followed by global average pooling; the MLP maps 101→64→48→32; the temporal convolutional network (TCN) [38] has width 24 and three residual blocks with dilations 1, 2 and 4, using symmetric padding over the complete pulse rather than strictly causal convolution; and the point-wise Transformer (PW-Trans.) [39] treats each of the 101 samples as a token (width 24, one layer, four heads) with mean pooling over tokens.

PatchTST was introduced for long-horizon forecasting [40]; here its patching idea is retained. The sequence of length $L=101$ is split into overlapping patches of length $P$ and stride $S$, whose number is

$$M=\operatorname{floor}\left(\frac{L-P}{S}\right)+1,\qquad \mathbf p_m=\left(\tilde v_{(m-1)S},\,\tilde v_{(m-1)S+1},\ldots,\tilde v_{(m-1)S+P-1}\right)^{\mathrm T}\in\mathbb{R}^{P}$$

{{EQ:12}}

With $P=9$ and $S=4$, $M=24$ and the last patch ends exactly at the 101st sample, so no padding or truncation is needed. Each patch is linearly embedded to $D=32$, a learnable positional embedding is added, and a single post-normalisation Transformer encoder layer with four heads, a GELU feed-forward width of 64 and dropout 0.2 models patch-level dependencies; layer normalisation, mean pooling over tokens and a linear projection give the 32-dimensional embedding (Supplementary Note S2). Patching first represents neighbouring samples jointly within a local window and then relates segments through attention, reducing the token count from 101 to 24; the branch has 10,752 parameters.

Relative to the original PatchTST [40], four changes adapt it to single-value regression from one short pulse. (i) Mean pooling plus a linear projection replaces the flatten head, making the embedding size independent of the patch count; a flatten head would raise the branch to 34,304 parameters (42,593 in total). (ii) Reversible instance normalisation (RevIN) is not used, preserving the response amplitude, which scales with internal resistance and may carry ageing information. (iii) The input is the first-sample-referenced, position-standardised single channel of Section 2.4 rather than the raw voltage. (iv) Only one encoder layer is kept, without the multi-step forecasting head or self-supervised pre-training. Changes (i)–(iii) are tested by reverting each in turn (Section 4.5.2).

: Architectures and parameter counts of the backbone and complement encoders.

| Module | Encoder | Main settings | Parameters |
|:-------|:---------|:------------------------------|-------:|
| Backbone | MLP | 51→64→32; layer norm, ReLU; dropout 0.2 after layer 1 | 5,600 |
| Backbone | ResNet | 2 residual blocks; width 28; hidden width 28 | 5,800 |
| Backbone | FT-Transformer | 1 block; token width 20; 4 heads; ReGLU width 26 | 6,124 |
| Complement | CNN | Channels 1→16→32→64; kernels 7, 5, 3; pooled projection | 11,232 |
| Complement | MLP | 101→64→48→32; layer norm, ReLU, dropout | 11,440 |
| Complement | TCN | Width 24; 3 residual blocks; dilations 1, 2, 4; symmetric padding | 11,744 |
| Complement | PW-Trans. | 101 tokens; width 24; 4 heads; 1 layer; FFN 48 | 8,192 |
| Complement | PatchTST | 24 patches; width 32; 4 heads; 1 layer; FFN 64; mean-pooled projection | 10,752 |
| Head | MLP | Concatenated 2 × 32; 64→32→16→1 | 2,689 |

TN: Total parameters of a pairing are the sum of backbone, complement and head: 16,481–20,557 across the 15 pairings and 19,041 for R0.

## 3.4. Regression head, loss and training

The head maps the 64-dimensional concatenation through 64→32→16→1 fully connected layers, with layer normalisation, ReLU and dropout 0.2 after the first layer and ReLU after the second (2,689 parameters). R0 thus has 5,600 + 10,752 + 2,689 = 19,041 parameters, and the similar budgets of all 15 pairings (16,481–20,557) limit the influence of model size on the comparison. Linear weights use Xavier uniform initialisation with zero biases.

The training objective combines the mean squared error of the standardised SOH with a penalty on positive partial derivatives with respect to $\tau$:

$$\mathcal{L}=\mathrm{MSE}\left(\hat y^{\text{*}},y^{\text{*}}\right)+\frac{\lambda}{B}\sum_{b=1}^{B}[\partial\hat y_b^{\text{*}}/\partial\tau_b]_{\text{+}}^{2}$$

{{EQ:13}}

where $B$ is the mini-batch size, $[x]_{\text{+}}=\max(0,x)$ and $\lambda=0.1$. The term discourages predictions that rise with $\tau$ when other inputs are held fixed, encoding the prior that capacity fades with cycling without a mechanistic degradation model. Because the voltage response and handcrafted features also change with ageing, a non-positive partial derivative does not guarantee monotone predictions along a measured trajectory, so the penalty is a soft constraint whose trajectory-level effect is examined in Section 4.5.1.

All networks use AdamW (learning rate 10^−3^, weight decay 10^−4^), batch size 256, at most 100 epochs and gradient-norm clipping at 5. ReduceLROnPlateau (patience 5, factor 0.5), early stopping with patience 15 and best-epoch selection all monitor the cell-macro root-mean-square error (RMSE) on the validation cells. Each configuration is trained on the five outer folds with initialisation seeds 0, 1 and 2; mini-batch shuffling uses a separate generator (seed + 10,000) and deterministic CUDA kernels, so every run is bitwise reproducible. Training used an NVIDIA GeForce RTX 5060 Laptop GPU; the full procedure is given in Supplementary Algorithm S1.

## 3.5. Ablations and reference models

### 3.5.1. Ablation and sensitivity configurations of R0

Branch ablations test the necessity of each branch (Supplementary Table S3). A2 keeps only the backbone (with $\tau$ and the monotonicity penalty) and feeds its 32-dimensional embedding to a 32→32→16→1 head (7,265 parameters); it is the direct reference for the waveform increment. A1 keeps only the PatchTST branch and concatenates the DCIR and $\tau$ directly to a 34→32→16→1 head (12,481 parameters). Factor ablations remove the penalty ($\lambda=0$, A3) and additionally remove $\tau$ (A4, 50-dimensional backbone input). Patch parameters are compared on a 2 × 2 grid: P1 (5, 2), P2 (5, 4), P3 (9, 2) and R0 (9, 4). Adaptation ablations revert one change of Section 3.3 at a time: D1 uses a flatten head, D2 adds RevIN with learnable affine parameters before patching, and D3 uses the raw voltage instead of $\Delta v$ (keeping position-wise standardisation). Finally, $K$ is varied over {10, 20, 30, 50, 75, 100, 144}, where 144 is the number of rankable candidates.

### 3.5.2. Conventional regressors and information-source controls

To test whether the backbone must be neural, five conventional regressors are trained on exactly the 51-dimensional input of A2: ridge regression, random forest, a histogram gradient-boosting decision tree (GBDT; scikit-learn, the same algorithm family as LightGBM), support vector regression with a radial basis function kernel (RBF-SVR) and Gaussian process regression (GPR). Hyperparameters are chosen on the validation cells from predefined grids (Supplementary Note S3) and are not refitted on training plus validation cells. To separate information sources, ridge, random forest and GBDT are also trained on $\tau$ alone and on the 50 features without $\tau$.

### 3.5.3. Reproduction of a published pulse model

The raw-pulse model of Nowacki et al. [28] is reproduced with the defaults of the public code under the present split and metric: the input is the 101-point $\Delta v$, inputs and targets are standardised with training statistics, and an MLP with five hidden layers of 100 tanh units is trained with Adam (learning rate 0.0015, batch 50, at most 500 epochs, early stopping on training loss with patience 5). One model per pulse direction covers all SOCs, and only the SOH output of the original multi-output model is kept (30 models in total).

### 3.5.4. Waveform increment over a strong backbone

Because the increment may depend on backbone strength, the GBDT selected in Section 3.5.2 is used as a backbone and networks learn its residuals. Training-cell residuals come from four-fold cell-disjoint out-of-fold predictions, so that they are not shrunk by in-sample fitting; validation and test residuals use a GBDT fitted on all training cells. The residual learners G+R0, G+A2 and G+A1 share the structures and training of R0, A2 and A1, with the residual standardised as the target, and the final prediction is the sum of the GBDT and residual predictions. (G+R0) − (G+A2) is thus the waveform increment over the GBDT backbone. As a post hoc addition, the 101 standardised voltages were also appended directly to the GBDT and SVR inputs and re-tuned.

### 3.5.5. Generalisation, robustness and deployment experiments

Five further experiments probe the scope of the model. (i) Unseen cycling conditions: each of the 11 cycling groups is held out in turn (leave-one-group-out), with validation cells drawn from the remaining groups. (ii) Independent chemistry: the UConn-ILCC NMC/Gr dataset [31] (44 cells, 11 cycling conditions) uses the same 100-s pulse protocol; the three nominal SOCs 20%, 50% and 90% are used for parity, giving 8,016 records in 1,336 cell–RPT groups (SOH 41.2–100%), and the identical pipeline is applied with five cell-disjoint folds (seed 42). Only the voltage-step and range limits of Eq. (2) are relaxed to 0.3 V and 0.8 V for the higher-resistance NMC cells, which removes no record. (iii) Robustness: the cycle count of the test cells is biased by ±10% with the trained models fixed, and zero-mean Gaussian noise with σ~n~ = 1 or 2 mV is added to all voltages before feature extraction, with every model retrained on the noisy data (matched sensor noise). (iv) The number of pulses averaged per RPT is varied from one to six over all condition combinations. (v) Integrated gradients [50] attribute R0's prediction to the waveform samples, and parameters, floating-point operations (FLOPs) and single-thread CPU latency quantify deployment cost. These experiments were run with an independent CPU re-implementation of the protocol. Under the original five-fold split, the re-implementation gives RMSEs of 1.427, 1.474 and 1.028 pp for R0, A2 and GBDT (versus 1.377, 1.500 and 1.022 pp), and every comparison in Section 4.6 is made within the re-implementation (Supplementary Note S6).

## 3.6. Metrics and statistical tests

The six condition records of an RPT share one capacity label and cannot be treated as independent samples. For cell $i$, RPT $r$ and initialisation $s$, the six condition predictions are first averaged to $\overline{y}_{irs}$; the RMSE of each cell over its RPTs is then averaged over seeds and cells:

$$e_{is}=\sqrt{\frac{1}{R_i}\sum_{r=1}^{R_i}\left(\overline{y}_{irs}-y_{ir}\right)^{2}},\qquad \mathrm{RMSE}=\frac{1}{NS}\sum_{i=1}^{N}\sum_{s=1}^{S}e_{is}$$

{{EQ:14}}

where $R_i$ is the number of RPTs of cell $i$, $N=64$ and $S=3$. The mean absolute error (MAE) is aggregated in the same way, and errors are expressed in SOH percentage points (pp). Uncertainty is estimated by 5,000 fold-stratified cell bootstrap resamples that preserve the pairing of cells between models. Paired differences are defined as ΔRMSE = RMSE~candidate~ − RMSE~reference~, so negative values favour the candidate, and *p* values come from 5,000 two-sided paired sign-flip randomisations. Holm correction is applied within nine prespecified test families at α = 0.05 (Supplementary Note S4); stratified, bias, marginal and post hoc analyses are descriptive. Intervals are conditional on the fixed split and trained models.

# 4. Results and discussion

Unless stated otherwise, RMSE and MAE follow Eq. (14), intervals are 95% fold-stratified cell bootstrap intervals, and "improved cells" counts cells with lower error in a paired comparison.

## 4.1. Encoder pairings and their dependence

Among the 15 pairings, R0 (MLP backbone + PatchTST complement) attains the lowest RMSE, 1.377 pp (1.247–1.513), with an MAE of 1.141 pp (Table 4, Fig. 3a). Relative to the MLP + CNN pairing it reduces RMSE by 0.131 pp (−0.160 to −0.101; Holm *p* = 0.003), or 8.7%, improving 51 of 64 cells. The runner-up, ResNet + MLP, reaches 1.397 pp, 0.020 pp above R0; because not all pairs were tested against each other, the lowest observed mean is not claimed to be significantly better than every alternative. R0 also ranks first on the validation cells (1.310 pp), and the top three pairings and their order coincide between validation and test; across the five test folds R0 ranks 8th, 1st, 1st, 1st and 3rd.

At the level of single factors, the FT-Transformer backbone raises the error by 0.147 pp on average relative to the MLP (0.107–0.191), whereas ResNet and MLP are indistinguishable; the average effects of the waveform encoders are small (−0.041 to −0.011 pp relative to the CNN). The ranking of complement encoders, however, depends on the backbone (Fig. 3b). PatchTST is best with the MLP backbone but last with ResNet and FT-Transformer, which prefer the MLP and point-wise Transformer complements, respectively. Taking MLP + CNN as the double reference, all eight difference-in-differences are positive (ResNet 0.061–0.140 pp, FT-Transformer 0.098–0.174 pp; uncorrected *p* ≤ 0.0022), so the advantage of a complement over the CNN shrinks or reverses when the backbone changes.

A sum-of-squares decomposition of the 3 × 5 RMSE table attributes 84.4% of the between-pairing variation to the backbone, 3.8% to the complement and 11.8% to their pairing (Fig. 3c). Without the FT-Transformer row the shares become 0.1%, 52.3% and 47.6%, so the apparent backbone dominance reflects the higher error of FT-Transformer, while the pairing term is substantial in both cases. Encoders should therefore be selected as pairings on controlled evidence, not by architectural complexity; this agrees with the BMS preference for lightweight, interpretable models [18,27] and supports the choice of an MLP backbone with a PatchTST complement.

![**Fig. 3.** Main effects and pairing dependence in the encoder matrix. (a) Cell-macro RMSE of the 15 pairings; the diverging scale is centred on the backbone-only model A2 (1.500 pp), blue denoting lower and red higher error; the frame marks R0. (b) Interaction plot; non-parallel lines indicate pairing dependence; the dashed line marks A2. (c) Share of the between-pairing sum of squares due to backbone, complement and pairing for all 15 pairings (95% intervals: 73.1–90.2%, 1.9–8.1%, 7.3–20.2%) and without FT-Transformer (0.0–4.6%, 40.4–65.0%, 34.1–58.4%), from 5,000 fold-stratified cell bootstraps. Values are seed-averaged cell metrics from 225 trainings.](figures/fig3_matrix.png){width=6.5in}

: Errors of all 15 encoder pairings (SOH pp).

| Backbone | Complement | Rank | RMSE | 95% CI | MAE | Val. RMSE (rank) | ΔRMSE^a^ | Holm *p* |
|:------------|:---------|----:|------:|:----------|------:|:----------|-------:|------:|
| MLP | PatchTST | 1 | **1.377** | 1.247–1.513 | 1.141 | 1.310 (1) | −0.131 | 0.003 |
| MLP | MLP | 3 | 1.408 | 1.271–1.553 | 1.154 | 1.321 (3) | −0.100 | 0.003 |
| MLP | PW-Trans. | 4 | 1.411 | 1.276–1.546 | 1.164 | 1.341 (5) | −0.098 | 0.003 |
| MLP | TCN | 6 | 1.437 | 1.294–1.585 | 1.175 | 1.356 (8) | −0.072 | 0.003 |
| MLP | CNN | 10 | 1.509 | 1.368–1.655 | 1.249 | 1.408 (10) | — | — |
| ResNet | MLP | 2 | 1.397 | 1.258–1.548 | 1.152 | 1.312 (2) | −0.111 | 0.003 |
| ResNet | TCN | 5 | 1.433 | 1.296–1.581 | 1.186 | 1.332 (4) | −0.075 | 0.003 |
| ResNet | CNN | 7 | 1.437 | 1.304–1.578 | 1.200 | 1.367 (9) | −0.072 | 0.003 |
| ResNet | PW-Trans. | 8 | 1.441 | 1.303–1.583 | 1.199 | 1.355 (7) | −0.068 | 0.003 |
| ResNet | PatchTST | 9 | 1.445 | 1.309–1.592 | 1.200 | 1.352 (6) | −0.064 | 0.003 |
| FT-Transformer | PW-Trans. | 11 | 1.555 | 1.406–1.709 | 1.303 | 1.495 (12) | +0.046 | 0.103 |
| FT-Transformer | CNN | 12 | 1.555 | 1.404–1.710 | 1.305 | 1.482 (11) | +0.046 | 0.093 |
| FT-Transformer | MLP | 13 | 1.573 | 1.420–1.731 | 1.315 | 1.495 (13) | +0.065 | 0.038 |
| FT-Transformer | TCN | 14 | 1.597 | 1.448–1.758 | 1.333 | 1.518 (14) | +0.089 | 0.003 |
| FT-Transformer | PatchTST | 15 | 1.598 | 1.446–1.757 | 1.340 | 1.525 (15) | +0.089 | 0.003 |

TN: ^a^ Pairing minus MLP + CNN; negative values denote lower error. Holm *p* values refer to the 14 comparisons against MLP + CNN, not to comparisons between pairings. Validation RMSE is the mean best-epoch error on validation cells over 15 runs.

## 4.2. Waveform increment over the handcrafted backbone

Adding the PatchTST branch to the MLP backbone lowers the error on the same test cells (Table 5, Fig. 4a). R0 improves on A2 by 0.123 pp (−0.141 to −0.103; Holm *p* < 0.001), with 58 of 64 cells improved and a paired effect size *d*~z~ = −1.55. Relative to A1, which lacks the handcrafted backbone, R0 improves by 0.117 pp (−0.187 to −0.033; Holm *p* = 0.002) in 50 of 64 cells, with a larger spread between cells (*d*~z~ = −0.36, Fig. 4b). A1 and A2 do not differ (−0.006 pp, −0.091 to 0.065; *p* = 0.91); because A1 still contains the DCIR and $\tau$, this is not a pure waveform-versus-feature comparison. The two representations are thus complementary on average.

: Ablation of the backbone, waveform complement, cycle count and monotonicity penalty (SOH pp).

| Config. | Inputs and constraint | RMSE | MAE | Contrast | ΔRMSE | 95% CI | Holm *p* | Improved cells |
|:------|:----------------------|------:|------:|:--------|-------:|:-------------|------:|--------:|
| A2 | Backbone + *τ*, *λ* = 0.1 | 1.500 | 1.244 | — | — | — | — | — |
| R0 | Backbone + PatchTST + *τ*, *λ* = 0.1 | 1.377 | 1.141 | R0 − A2 | −0.123 | −0.141 to −0.103 | <0.001 | 58/64 |
| A1 | PatchTST + DCIR + *τ*, *λ* = 0.1 | 1.494 | 1.259 | R0 − A1 | −0.117 | −0.187 to −0.033 | 0.002 | 50/64 |
| A3 | As R0, *λ* = 0 | 1.406 | 1.164 | R0 − A3 | −0.028 | −0.038 to −0.018 | <0.001 | 46/64 |
| A4 | As A3, without *τ* | 1.983 | 1.718 | A3 − A4 | −0.578 | −0.731 to −0.421 | <0.001 | 52/64 |

TN: ΔRMSE is the first minus the second configuration of each contrast; improved cells are those where the first has the lower error. The four contrasts are prespecified confirmatory tests with Holm correction among them.

Not every waveform encoder delivers this increment (Table 6, Fig. 4c). With the MLP backbone fixed, PatchTST, MLP, point-wise Transformer and TCN complements lower RMSE by 0.123, 0.092, 0.089 and 0.063 pp (Holm *p* < 0.001 for all), whereas the CNN complement changes it by +0.009 pp (−0.012 to 0.028; Holm *p* = 0.491). PatchTST gives the largest observed reduction, consistent with its rank in Section 4.1.

: Increment of each waveform complement over the MLP backbone A2 (SOH pp).

| Complement | Dual-branch RMSE | ΔRMSE | 95% CI | Relative change | Holm *p* | Improved cells | *d*~z~ |
|:---------|--------:|-------:|:-------------|--------:|------:|--------:|------:|
| PatchTST | 1.377 | −0.123 | −0.141 to −0.104 | −8.2% | <0.001 | 58/64 | −1.55 |
| MLP | 1.408 | −0.092 | −0.109 to −0.074 | −6.1% | <0.001 | 55/64 | −1.14 |
| PW-Trans. | 1.411 | −0.089 | −0.109 to −0.070 | −5.9% | <0.001 | 54/64 | −0.97 |
| TCN | 1.437 | −0.063 | −0.089 to −0.037 | −4.2% | <0.001 | 45/64 | −0.56 |
| CNN | 1.509 | +0.009 | −0.012 to 0.028 | +0.6% | 0.491 | 28/64 | +0.09 |

TN: ΔRMSE is the dual-branch model minus A2 (1.500 pp); *d*~z~ is the mean per-cell difference divided by its standard deviation. Holm correction covers these five comparisons.

![**Fig. 4.** Effects of the handcrafted backbone and the waveform complement. (a) Per-cell RMSE difference R0 − A2 (seed-averaged) for the 64 test cells sorted in ascending order; blue bars denote lower and red bars higher error for R0; the dashed line and grey band show the mean difference and its 95% interval. (b) As (a) for R0 − A1; the rightmost bar is cell 3. (c) Left: numbers of cells with lower (blue) or higher (red) error after adding each complement to the MLP backbone, and for A1 versus A2 (below the dashed line); pale colours mark non-significant differences. Right: violin and box plots of the 64 paired differences (box, interquartile range; line, median; whiskers, 1.5 × IQR; diamond, mean); one value outside the axis (A1 − A2, −1.72 pp, cell 3) is marked at the left edge.](figures/fig4_increment.png){width=6.5in}

## 4.3. Distribution of the increment and bias correction

The increment varies in size across SOH ranges but has the same sign in all six pulse conditions (Supplementary Table S4, Fig. 5b). Below 80% SOH, R0 lowers the error relative to A2 by 0.413 pp (17.2%, 62/64 cells), and above 95% by 0.148 pp (15.5%, 61/64); the 90–95% range shows the smallest gain (0.027 pp, 36/64). Across the six conditions the reduction ranges from 0.085 to 0.139 pp; 313 of the 384 cell–condition units (81.5%) improve, 29 cells improve in all six conditions, and all 11 cycling groups show negative mean differences (−0.029 to −0.186 pp). Each single pulse condition therefore benefits from the complement, which matters for deployments that can afford only one pulse.

Binned bias analysis shows that the increment partly corrects a shrinkage bias of the backbone (Fig. 5a). Over the 2,823 cell–RPT points, A2 overestimates SOH by 3.25 pp below 78% and underestimates it by 1.41 pp at 98–100%; R0 reduces these to 2.74 and 1.16 pp. The change in prediction from A2 to R0 increases with true SOH (slope 0.034 pp per pp, 0.031–0.036) and crosses zero near 92.1%: predictions are lowered at low SOH and raised at high SOH, opposite to the bias of A2. As a result, 68.0% of the points move closer to the measurement, the slope of predicted on true SOH rises from 0.848 to 0.882, and the total squared error falls by 14.4%.

![**Fig. 5.** Bias correction by the waveform complement and its distribution across conditions and cells. (a) Top: mean bias (prediction minus truth) in 2-pp bins of measured SOH (bins below 78% merged) for A2 (grey) and R0 (blue); the shading marks the corrected bias. Bottom: R0 − A2 prediction change at the 2,823 cell–RPT points (blue, closer to the measurement; red, farther) with the least-squares fit and its 95% bootstrap band. Predictions are averaged over seeds and conditions. (b) Left: per-cell R0 − A2 RMSE difference for the six conditions and all records (cells ordered by cycling group; colours saturate beyond ±0.3 pp). Right: cell-macro mean difference per row; all 95% intervals exclude zero.](figures/fig5_bias.png){width=6.5in}

## 4.4. Learner dependence and comparison with reference methods

The comparison with reference methods delimits where the dual-branch design helps (Table 7; all results in Supplementary Table S5). The reproduced raw-pulse model of Nowacki et al. [28] reaches 1.628 pp, and R0 is 0.251 pp lower (Holm *p* = 0.019); since that model uses only the raw pulse without $\tau$, the gap cannot be attributed to the dual-branch structure alone. On the A2 input, GBDT and RBF-SVR reach 1.022 and 1.026 pp. A2 and GBDT share the same input yet differ by about 0.48 pp, so the backbone learner matters on these data. Models using $\tau$ alone err by about 3 pp and GBDT without $\tau$ reaches 1.663 pp, so the cycle record and the pulse features both contribute.

: Inputs and cell-macro errors of representative methods (SOH pp).

| Model | Input | RMSE | 95% CI | ΔRMSE vs. A2 | ΔRMSE vs. R0 | Holm *p*^a^ | Cells better than R0 |
|:----------------|:-------------|------:|:----------|--------:|--------:|------:|--------:|
| R0 (this work) | Features + *τ* + ΔV | 1.377 | 1.247–1.513 | −0.123 | — | — | — |
| A2 (backbone only) | Features + *τ* | 1.500 | 1.365–1.637 | — | +0.123 | <0.001 | 6/64 |
| Nowacki et al. [28], reproduced | Raw ΔV | 1.628 | 1.454–1.820 | +0.128 | +0.251 | 0.019 | 17/64 |
| GBDT | Features + *τ* | 1.022 | 0.897–1.160 | −0.478 | −0.355 | 0.006 | 58/64 |
| RBF-SVR | Features + *τ* | 1.026 | 0.915–1.146 | −0.474 | −0.351 | 0.006 | 57/64 |

TN: ^a^ Versus R0. R0 versus A2 belongs to the prespecified family of Table 5; the other comparisons belong to the 30-comparison extended family (Supplementary Note S4).

The waveform increment does not carry over to the GBDT backbone (Fig. 6). Whereas the neural backbone gains 0.123 pp in 58 of 64 cells, (G+R0) − (G+A2) is only −0.008 pp with 36 of 64 cells improved (Holm *p* = 0.35); none of the three residual learners improves GBDT significantly, and appending the raw waveform to GBDT gives no gain either (Supplementary Note S5). This pattern is consistent with the waveform branch recovering information that the neural backbone leaves unexploited in the handcrafted features, which a tree ensemble already extracts. The practical rule is therefore that the waveform complement should be added when a neural handcrafted backbone is used, for example where end-to-end differentiability is needed for derivative constraints, while a well-tuned tree ensemble can rely on the handcrafted features.

![**Fig. 6.** Dependence of the waveform increment on the backbone learner. Paired change in cell RMSE after adding the waveform (negative = lower error) for the 64 test cells. Top: neural backbone, R0 − A2. Bottom: GBDT backbone with the waveform added through residual networks, (G+R0) − (G+A2). Circles, cells; violins, distributions; boxes, interquartile ranges with medians; diamonds, means. Annotations give the mean difference, improved cells and Holm-adjusted *p*.](figures/fig6_learner.png){width=6.5in}

## 4.5. Contribution of design factors

### 4.5.1. Cumulative cycle information and monotonicity penalty

Among the ablated factors, removing the cycle count increases the error most (Table 5, Fig. 7a–c). Without the monotonicity penalty in either model, A3 (with $\tau$) is 0.578 pp below A4 (without $\tau$; reduction 0.421–0.731; Holm *p* < 0.001) in 52 of 64 cells, i.e. removing $\tau$ raises the error by about 41%. Together with the large error of $\tau$-only models (Section 4.4), the cycle record is an important but not self-sufficient input.

The soft monotonicity penalty yields a small accuracy gain (Fig. 7d–f): R0 is 0.028 pp below A3 (reduction 0.018–0.038; Holm *p* < 0.001) in 46 of 64 cells. The share of positive partial derivatives with respect to $\tau$ (5.39% for A3, 5.61% for R0) and of rising predictions between adjacent RPTs of the same cell and condition (26.88% and 27.03%) remain essentially unchanged, so the penalty acts as a regulariser rather than as a guarantee of monotone trajectories.

![**Fig. 7.** Effects of cumulative cycle information and the monotonicity penalty. Top row: A4 (no cycle input) versus A3 (cycle input), both without the penalty. Bottom row: A3 (no penalty) versus R0 (penalty). (a, d) RMSE; (b, e) MAE; (c, f) rate of rising predictions between adjacent RPTs. Error distributions contain the 64 test cells (seed-averaged); rise rates are computed within cell, nominal SOC and direction for the 15 runs. Violins are kernel densities limited to the observed range; points are observations and diamonds means. Rows use different y-axis ranges.](figures/fig7_cycle_mono.png){width=6.5in}

### 4.5.2. Design of the waveform branch

On the 2 × 2 patch grid, R0's (9, 4) setting gives the lowest error (Table 8; Supplementary Fig. S1). P1 (5, 2), P2 (5, 4) and P3 (9, 2) raise RMSE by 0.072, 0.094 and 0.055 pp (Holm *p* < 0.001), and length and stride interact: stride 4 is better with length 9 but stride 2 is better with length 5 (difference-in-differences −0.077 pp, −0.100 to −0.054). Of the three adaptations, the mean-pooling head is clearly supported: reverting to a flatten head (D1) raises the error by 0.062 pp (0.036–0.089; Holm *p* < 0.001) while more than doubling the parameter count (19,041 to 42,593). Adding RevIN (D2, −0.020 pp, −0.049 to 0.005) or using the raw voltage (D3, +0.007 pp, −0.005 to 0.017) makes no detectable difference (Holm *p* = 0.345 for both). With A2 as the common reference, all six PatchTST variants reduce the error by 0.028–0.143 pp (40–57 of 64 cells improved), so the increment is not tied to a single patch setting.

: PatchTST patch parameters and adaptation ablations (SOH pp).

| Config. | Difference from R0 | RMSE | 95% CI | Params. | ΔRMSE | 95% CI of ΔRMSE | Holm *p* | R0 better |
|:-------|:---------------------|------:|:----------|--------:|-------:|:-----------|------:|---------:|
| R0 | Patch (9, 4), mean-pooling head, no RevIN, ΔV input | 1.377 | 1.247–1.513 | 19,041 | — | — | — | — |
| P1 | Patch (5, 2) | 1.449 | 1.315–1.585 | 19,713 | +0.072 | 0.050–0.094 | <0.001 | 51/64 |
| P2 | Patch (5, 4) | 1.472 | 1.341–1.611 | 18,945 | +0.094 | 0.080–0.109 | <0.001 | 56/64 |
| P3 | Patch (9, 2) | 1.432 | 1.297–1.570 | 19,777 | +0.055 | 0.038–0.071 | <0.001 | 47/64 |
| D1 | Flatten head | 1.440 | 1.304–1.585 | 42,593 | +0.062 | 0.036–0.089 | <0.001 | 46/64 |
| D2 | With RevIN | 1.357 | 1.229–1.491 | 19,043 | −0.020 | −0.049 to 0.005 | 0.345 | 28/64 |
| D3 | Raw voltage input | 1.384 | 1.253–1.518 | 19,041 | +0.007 | −0.005 to 0.017 | 0.345 | 41/64 |

TN: ΔRMSE is the configuration minus R0; positive values mean the corresponding R0 design is beneficial. Patch (P1–P3) and adaptation (D1–D3) ablations form two families with Holm correction over three comparisons each.

### 4.5.3. Number of handcrafted features

Within the present candidate set, $K=50$ is supported (Supplementary Table S6, Fig. S2). $K$ = 10, 20 and 30 raise RMSE by 0.139, 0.122 and 0.091 pp (Holm *p* = 0.001), $K$ = 75 and 100 are slightly worse (+0.037 and +0.041 pp; Holm *p* = 0.002), and $K=144$ is indistinguishable from $K=50$ (−0.002 pp, −0.019 to 0.015; Holm *p* = 0.838) while requiring 31.6% more parameters. $K=50$ is therefore a balanced choice between accuracy and input size.

## 4.6. Generalisation, robustness and deployment

The experiments in this section use the CPU re-implementation of Section 3.5.5, which reproduces the cleaning exactly and the reported errors closely, including the sign and significance of the waveform increment (R0 − A2 = −0.046 pp, −0.070 to −0.023; *p* < 0.001; 42 of 64 cells) and the identity of the best- and worst-estimated cells (cells 12 and 3). All comparisons below are made within this implementation (Table 9, Fig. 8).

*Unseen cycling conditions.* When each cycling group is held out entirely, the error of the neural models rises only moderately (R0 from 1.427 to 1.560 pp, +9.3%; A2 +8.2%), whereas that of GBDT rises sharply (from 1.028 to 1.401 pp, +36.2%), so the advantage of GBDT over R0 shrinks from 0.40 to 0.16 pp (Fig. 8a). For group 3, the condition hardest to extrapolate, R0 is 0.59 pp more accurate than GBDT. The waveform increment persists for unseen conditions (R0 − A2 = −0.035 pp, −0.053 to −0.015; *p* = 0.008; 39 of 64 cells), with lower group-mean errors in seven of the 11 held-out groups.

*Cycle-count errors and sensor noise.* A systematic ±10% error in the cycle count raises R0's error by only 1.5–3.5% (1.449 and 1.477 pp), compared with 8.6–14.4% for GBDT (1.117 and 1.176 pp), and leaves the waveform increment intact (−0.048 and −0.041 pp; *p* ≤ 0.002) (Fig. 8b). With matched sensor noise in training and test data, accuracy degrades gracefully: at σ~n~ = 1 and 2 mV, R0 reaches 1.620 and 1.715 pp (+13.5% and +20.2%), against +33.5% and +38.5% for GBDT (1.372 and 1.424 pp), and the waveform increment is maintained (−0.058 pp in 50 of 64 cells and −0.045 pp in 46 of 64 cells; *p* ≤ 0.002).

*Number of pulses.* With a single pulse per RPT, R0 reaches 1.549 pp on average over the six conditions (1.490–1.582 pp); two and three pulses give 1.477 and 1.453 pp, against 1.427 pp for all six (Fig. 8c), and R0 stays below A2 for every pulse count. The three charge pulses alone reach 1.432 pp, within 0.005 pp of the full protocol, so the diagnostic time can be halved, and the pulses can be applied during charging as proposed for vehicles [16].

*Attribution.* Integrated gradients spread R0's waveform attribution almost evenly over the response: the mean absolute attribution per sample is 0.0132, 0.0138 and 0.0135 pp in the C/5, 1C and rest segments, with the same profile for all six conditions (Fig. 8d). The complement therefore draws on the shape of the whole transient rather than on one segment that a single handcrafted descriptor could isolate, in line with its intended role of capturing inter-segment information.

*Computational cost.* R0 has 19,041 parameters (74 kB in 32-bit floating point) and needs 0.42 MFLOPs per pulse, most of them in the waveform branch (A2: 13.7 kFLOPs). On a single CPU thread, one inference takes 0.52 ms in PyTorch and the reference Python implementation of all 143 waveform features takes 6.5 ms, both negligible against the 100-s pulse, whereas the validation-selected GBDT comprises 1,000 trees with 61,000 nodes. This footprint is of the order targeted by embedded SOH estimators [19].

: Generalisation and robustness in the re-implementation (cell-macro RMSE, SOH pp).

| Scenario | R0 | A2 | GBDT | R0 − A2 | 95% CI | *p* | Improved cells |
|:----------------------|------:|------:|------:|-------:|:-------------|------:|--------:|
| Five-fold cell-disjoint (reference) | 1.427 | 1.474 | 1.028 | −0.046 | −0.070 to −0.023 | <0.001 | 42/64 |
| Leave-one-group-out | 1.560 | 1.594 | 1.401 | −0.035 | −0.053 to −0.015 | 0.008 | 39/64 |
| Cycle count +10% (test) | 1.449 | 1.497 | 1.117 | −0.048 | −0.072 to −0.024 | <0.001 | 43/64 |
| Cycle count −10% (test) | 1.477 | 1.518 | 1.176 | −0.041 | −0.063 to −0.018 | 0.002 | 39/64 |
| Matched noise, σ~n~ = 1 mV | 1.620 | 1.678 | 1.372 | −0.058 | −0.077 to −0.040 | <0.001 | 50/64 |
| Matched noise, σ~n~ = 2 mV | 1.715 | 1.760 | 1.424 | −0.045 | −0.069 to −0.022 | 0.002 | 46/64 |
| One pulse per RPT^a^ | 1.549 | 1.574 | 1.203 | — | — | — | — |
| Three pulses per RPT^a^ | 1.453 | 1.494 | 1.066 | — | — | — | — |
| Three charge pulses | 1.432 | — | — | — | — | — | — |

TN: Paired comparisons use 5,000 fold- (or group-) stratified cell bootstraps and sign-flip tests; descriptive, without multiplicity correction. ^a^ Mean over all condition combinations (6 and 20, respectively).

![**Fig. 8.** Generalisation, robustness and attribution (re-implementation). (a) Group-mean RMSE when each cycling group is held out (leave-one-group-out); the legend gives the cell-macro RMSE under leave-one-group-out and, in parentheses, under five-fold cell-disjoint validation. (b) RMSE with a ±10% bias of the test cycle count (models fixed) and with matched Gaussian voltage noise in training and test data (models retrained). (c) RMSE versus the number of pulses averaged per RPT; lines show the mean over all condition combinations and bands their range. (d) Mean absolute integrated-gradient attribution of R0 per waveform sample for charge (solid) and discharge (dashed) pulses, averaged over test records, folds and seeds; shading marks the pulse segments.](figures/fig8_general.png){width=6.5in}

## 4.7. Error characteristics and scope

The overall mean bias is close to zero but conceals SOH-dependent errors (Supplementary Fig. S3). Averaging the three initialisations of each record, R0 reaches a record-level RMSE of 1.461 pp, an MAE of 1.089 pp and a mean bias of −0.093 pp over the 16,938 records. The 630 records below 80% SOH are overestimated by 2.003 pp on average and the 5,682 records above 95% are underestimated by 0.574 pp (GBDT: +1.068 and −0.130 pp). Individual cells differ markedly (Supplementary Fig. S4): cells 12, 6, 44 and 3, ranked 1st, 22nd, 43rd and 64th by R0 error, reach 0.526, 0.961, 1.519 and 3.568 pp (A2: 0.704, 1.066, 1.723 and 3.624 pp). Across the 11 cycling groups the cell-macro RMSE spans 0.878–2.333 pp (Supplementary Fig. S5), with positive biases in groups 3 and 6 and negative biases in group 8; the correlation between the per-cell fade rate and the mean bias is 0.59.

The analysis treats the 64 cells as the unit of inference on a fixed split; the low-SOH overestimation is reduced but not removed, and training data should come from the target sensor because difference- and spectrum-based features are sensitive to noise absent from training (Supplementary Note S6). Applied unchanged to the NMC/graphite dataset, the pipeline showed no waveform increment (R0 2.030 versus A2 2.026 pp, GBDT 0.935 pp; Supplementary Table S7), so the benefit is specific to the LFP data and neural backbones studied here, and validation on field data with varying temperature and on target BMS hardware remains future work.

# 5. Conclusions

Using 16,938 short-pulse records from 64 public LFP/graphite cells, 15 encoder pairings were compared under one five-fold cell-disjoint protocol and training budget. The main conclusions are as follows.

(1) Encoders should be evaluated as pairings. The MLP backbone with a PatchTST complement (R0) gives the lowest observed RMSE (1.377 pp), 8.7% below MLP + CNN, while PatchTST ranks last with the other two backbones; the pairing term explains 11.8% of between-pairing variation (47.6% without FT-Transformer).

(2) The waveform complement improves a neural handcrafted backbone. R0 is 0.123 pp (8.2%) below the backbone-only model in 58 of 64 cells, with the largest gain below 80% SOH (17.2%), consistent improvements in all six pulse conditions and a partial correction of the backbone's shrinkage bias. With a GBDT backbone, which already reaches 1.022 pp, the waveform adds no measurable gain.

(3) The waveform increment persists, in a re-implementation, for held-out cycling conditions (−0.035 pp) and under matched sensor noise of 1–2 mV (−0.045 to −0.058 pp), and the neural model degrades far less than GBDT when cycling conditions are unseen (+9% versus +36%) or the cycle count is biased by ±10% (at most +3.5% versus +14.4%).

(4) Among the ablated factors, the cumulative cycle count is the most influential input (+41% error without it), and the monotonicity penalty acts as a mild regulariser (−0.028 pp). The (9, 4) patch setting and the mean-pooling head (−0.062 pp and 55% fewer parameters than a flatten head) are the most effective design choices, and $K=50$ balances accuracy and input size.

(5) The model is light enough for embedded use, with 19,041 parameters (74 kB), 0.42 MFLOPs and sub-millisecond single-core inference per pulse; its attribution is spread over the whole transient, and the three charge pulses alone match the six-pulse protocol within 0.005 pp.

@@HEAD CRediT authorship contribution statement

[Author 1]: Conceptualization, Methodology, Software, Formal analysis, Writing – original draft. [Author 2]: Validation, Visualization, Writing – review & editing. [Corresponding Author]: Supervision, Funding acquisition, Writing – review & editing.

@@HEAD Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

@@HEAD Data availability

The battery data are publicly available from the UConn-ISU-ILCC LFP and UConn-ILCC NMC ageing datasets [28,31]. Cleaning rules, input definitions, splits and metrics are specified in Sections 2 and 3 and Supplementary Notes S1–S6. [Repository, version and access details of the code and derived data to be added by the authors.]

@@HEAD Acknowledgements

[Funding sources and grant numbers to be added.]

@@HEAD Appendix A. Supplementary data

Supplementary data to this article (Supplementary Notes S1–S6, Tables S1–S7, Figs. S1–S5 and Algorithm S1) are provided in a separate file.

@@HEAD References

@@REFS
