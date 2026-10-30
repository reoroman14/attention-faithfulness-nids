# Task 37: decisions and outcome rule for A8x (exact-objective attacker on D+)

Recorded: 2026-10-30 00:54 Amman time (UTC+3), before any A8x code was written or run.
Source design: handoff\35_narrative.md section 2.2. handoff\33_decisions.md is not changed.

## Design

1. **Attacker A8x.** Same as A8 (src\stage6e_strong.py, terms_A8 and standin_features), with these changes:
   - IG in the loss: **20-step Gauss-Legendre** IG from the zero baseline (the nodes and weights of captum 0.9.0
     IntegratedGradients, n_steps 20, method gausslegendre), with double backprop through all 20 steps, instead of the
     8-step midpoint rule. Target class benign, as in A8 (equal to the predicted class on every row that can count).
   - Spearman and top-10 Jaccard: soft ranks and soft top-10 membership as in A8, with **tau annealed from 0.1 to 0.03**
     over the steps (geometric schedule, step t of T: tau_t = 0.1 x (0.03 / 0.1)^(t / (T - 1))).
   - Cosines, attention (rollout, layer mean), grad x input and the 4 D features: exact, as in A8.
   - **Best-iterate selection with the true D+ features and the true D+ decision.** At every step, for every row, the
     true 16 D+ features (hard Spearman, hard top-10 Jaccard with the evaluation tie-break, cosines, on rollout,
     layer-mean attention, grad x input and IG-20 of the predicted class) and the true D+ score are computed. Each row
     keeps the iterate that is predicted benign with the lowest true D+ score; if no iterate is predicted benign, the
     iterate with the highest benign margin is kept.
2. **Settings, rows, seeds, eps.** As Stage B: eps 1; 3 settings (lambda 1, lambda 100, normalised gradient w 0.5),
   with the PGD settings of A8 (100 steps, 1 restart, kappa, safety margin as in the saved configs); seeds 0, 1, 2;
   source rows: the 7 common classes, 1,044 rows (2017) and 985 rows (2018) per seed. Validity checker as in all stages.
3. **Detector.** The saved D+ of each seed (defenders_eps1.pkl), threshold at 5% FPR as saved. Detection is always
   evaluated with the true features.
4. **Chunk size 50**, fixed. No smaller chunk and no other settings without stopping and reporting.
5. **Not rerun:** the Stage B results for the controls (c0, c1, c2, a1, a2) are not rerun. Control b stays untested.

## Headline quantity

Equal-effort union, 2017 and 2018 separately, V1, eps 1: share of source rows that are successful (flipped to benign)
and undetected.
- D+ faces **A1 + A8x x 3**. D faces **A1 + A6 x 3** (unchanged).
- Reported: **D+ minus D**, pooled over seeds with the paired bootstrap 95% interval (1,000 draws, source rows
  resampled within seed), and the three per-seed values.
- Sensitivity row (unequal effort, marked as such): D+ faces A1 + A8 x 3 + A8x x 3.
- Secondary: V0, V2, per setting (paired TPR differences D+ minus D on A8x rows).

## Outcome rule (2017 V1, D+ minus D under A1 + A8x x 3)

- **R1:** the 95% interval contains 0 or lies above 0. The 2017 gain is an artefact of the stand-ins.
- **R2:** the interval lies below 0 (upper bound < 0) **and** all three per-seed values are negative (< 0). The gain
  persists against the strongest attacker we can build.
- **R3:** the interval lies below 0 and at least one per-seed value is not negative (>= 0). The claim stands as
  written, with the caveat reworded ("rank scores use surrogate gradients; all other pieces exact").

R1, R2, R3 are exhaustive and exclusive given the interval and the three seed values.

**2018:** an interval above 0 (lower bound > 0) is recorded as "D+ worse than D on 2018", one sentence, and cannot
change the conclusion (floor effect: D alone leaves 0.023 of the source rows successful and undetected). Any other
2018 result is reported without a change to the conclusion.

## Sanity checks (before any pilot; a failure stops the work)

- (a) With IG set to 8-step midpoint and tau fixed at 0.1, A8x reproduces the A8 loss and the A8 stand-in feature
  values on 200 rows to within 1e-5.
- (b) The IG-20 Gauss-Legendre values of A8x equal the captum IntegratedGradients values used for the true features
  to within 1e-5 on 200 rows.
- (c) The true D+ features computed inside A8x equal the saved D+ features of the same rows.

## Memory limits (full-size pilot, per dataset)

All 1,044 (2017) or 985 (2018) rows, 5 steps, then the scoring phase, at chunk 50. Logged: torch.cuda.max_memory_reserved
and the Windows counter "GPU Process Memory Shared Usage" of the process. **Go only if shared usage stays below 0.5 GB
and reserved memory below 6.5 GB.** torch.cuda.empty_cache() between chunks and before scoring. During all runs the
shared-usage counter is polled every minute.

## Stop conditions

- Any invalid row from the validity checker.
- A failed sanity check (a), (b) or (c).
- A full-size pilot that fails the memory limits above.
- A sustained shared GPU memory value above 1 GB during a run (sustained: 3 consecutive one-minute polls).

## Amendment 1 (sanity tolerance and selection margin)

Recorded: 2026-10-30 01:25 Amman time (UTC+3), before any A8x pilot or attack. No A8x attack row exists.

- **Reason.** Sanity check (c) failed on 2017 seed 0 at tolerance 1e-5 for mahalanobis_benign (max difference
  2.4e-3, about 3.5e-4 relative) and for the D+ decision (2.1e-4). Cause: float32 matrix products with different batch
  shapes (50 rows in A8x, score_batch rows in the saved evaluation) on an ill-conditioned inverse covariance. The same
  feature differed by up to 4.3e-3 in the Stage B stand-in quality check (c0). The amendment is made before any A8x
  attack or pilot.
- **New tolerance for check (c):** relative 1e-3 on mahalanobis_benign, absolute 1e-3 on the D+ decision. All other
  checks keep 1e-5.
- **Selection margin.** Inside A8x, an iterate counts as flipped and undetected for best-iterate selection only if the
  true D+ score inside A8x is below the saved threshold by more than 1e-3. This is conservative: it can only make A8x
  pick fewer borderline iterates, not more. Reported scores and the union are still computed by the evaluation
  pipeline, as for every other attack, with the saved threshold and no margin.
- Everything else in the rule (outcomes R1, R2, R3, headline, stop conditions, memory limits) is unchanged.
- Implementation note (added when writing this amendment): the selection key becomes three ordered classes,
  (1) flipped and true D+ score < thr - 1e-3, (2) flipped otherwise, (3) not flipped; within (1) and (2) the lowest
  true D+ score wins, within (3) the highest benign margin. Because the lowest true score already wins among flipped
  iterates, an iterate of class (1) always beats one of class (2), so the margin cannot change which iterate is kept;
  it only changes which kept iterates the in-loop log counts as undetected. The report gives both counts.
