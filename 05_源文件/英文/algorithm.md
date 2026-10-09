**Algorithm S1.** Training and inference of the dual-branch SOH estimation model (handcrafted backbone + pulse-waveform complement).

**Input:** cleaned record set *D* = {(*V*, *x*, *y*, cell, rpt, soc, pulse)} with 16,938 records of 64 cells; *F* = 5 folds, split seed 42, initialisation seeds {0, 1, 2}; *K* = 50, redundancy threshold 0.98, *λ* = 0.1.

**Output:** record-level test predictions and cell-macro errors for every fold and seed.

**Offline training**

1: Remove cumulative cycle count, RPT index, coulomb-counted SOC and SOC deviation from the 150 candidates → 146 selectable features (Section 2.3.2)

2: Permute the 64 cells with seed 42 and divide them into five disjoint test folds (13/13/13/13/12 cells)

3: In each fold, draw validation cells from the non-test cells with the fold index as offset and stride 5 (train/validation/test = 40/11/13, 41/10/13, 41/10/13, 41/10/13, 42/10/12)

4: **for** outer fold *f* = 0, …, 4 **do**

5: ⠀⠀Split *D* by cell into *D*~tr~, *D*~va~, *D*~te~; all RPTs and condition records of a cell stay together

6: ⠀⠀Compute Spearman *ρ*~*j*~ of the 146 candidates with SOH on *D*~tr~ only ▷ Eq. (4)

7: ⠀⠀Select *K* features greedily by |*ρ*~*j*~|, skipping rank correlation ≥ 0.98 with selected ones; fill to *K* if needed ▷ Eq. (5)

8: ⠀⠀Fit medians, feature means/s.d., per-position waveform means/s.d., cycle-count and label statistics on *D*~tr~ only

9: ⠀⠀Transform *D*~tr~, *D*~va~, *D*~te~ with these statistics ▷ Eqs. (6)–(10)

10: ⠀⠀**for** seed *s* ∈ {0, 1, 2} **do**

11: ⠀⠀⠀⠀Initialise *E*~h~, *E*~v~ and *g* with seed *s*; AdamW (lr 10^−3^, weight decay 10^−4^), batch 256, ReduceLROnPlateau (patience 5, factor 0.5), gradient clipping 5, at most 100 epochs, early-stopping patience 15

12: ⠀⠀⠀⠀**for** each epoch and mini-batch **do**

13: ⠀⠀⠀⠀⠀⠀Enable gradients on *τ*; forward pass gives *ŷ*\* ▷ Eq. (11)

14: ⠀⠀⠀⠀⠀⠀Compute ∂*ŷ*\*/∂*τ* by automatic differentiation (create_graph = True)

15: ⠀⠀⠀⠀⠀⠀ℒ ← MSE(*ŷ*\*, *y*\*) + *λ*·⟨[∂*ŷ*\*/∂*τ*]~+~^2^⟩; back-propagate, clip, update ▷ Eq. (13)

16: ⠀⠀⠀⠀⠀⠀After each epoch, compute the cell-macro RMSE on *D*~va~; keep the best weights; stop after 15 epochs without improvement

17: ⠀⠀⠀⠀Load the best-epoch weights and predict *D*~te~

**Online inference (one pulse record)**

18: Reference the 101-point voltage to its first sample and standardise with the fold statistics → *ṽ* ▷ Eqs. (6), (7)

19: Compute the *K* selected features, impute and standardise, append standardised *τ* → **h** ▷ Eqs. (8), (9)

20: *ŷ*\* ← *g*([*E*~v~(*ṽ*); *E*~h~(**h**)]); SOH ← 100(*σ*~*y*~*ŷ*\* + *μ*~*y*~) ▷ Eqs. (11), (10)

**Metrics and statistics**

21: Average the six condition predictions of each cell–RPT, compute per-cell RMSE and the cell-macro mean ▷ Eq. (14)

22: Paired differences: 5,000 fold-stratified cell bootstrap intervals; 5,000 two-sided paired sign-flip tests; Holm correction within families
