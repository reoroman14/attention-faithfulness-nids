# Task 33: decisions and pre-registered rule for Stage B

Recorded: 2026-10-29 06:24 Amman time (UTC+3), before any Stage B code was written or run.
Source plan: handoff\32_restructure_plan.md (sections 4 and 6). Applies to src\stage9b_dplus_adaptive.py and to
results\stage9\adaptive\.

## Decisions

1. **Controls attacked:** a1, a2, b, c2 (A8-type adaptive attackers), plus c0 and c1 (A6-type attackers, exact
   gradients). With these six, every variant that passed the gate rule of handoff 18 is attacked except the second
   shuffle seeds a1s and a2s. Those are repeats of a1 and a2 with another random draw, and a2's draw is fixed per row
   in Stage B anyway (decision 3).

2. **a1 under attack (plan 4.5 risk 2, option ii):** the partner attention (rollout and layer mean) is computed on the
   **clean, unattacked source rows** of the same seed. The partner is taken from a fixed derangement within the true
   class (same `derangement` function as Stage A, seeded by year, seed and "a1fixed"). The attacker and the
   evaluation use the same fixed partner vector, so the attacker models the exact feature that the detector sees. The
   a1 detector itself is the Stage A detector (refit and checked by gate0b): it was trained with partner attention
   taken from the same set as the row. a1 uses the **true class of each test row (an oracle label)** to choose the
   partner, so no deployed system could compute it. It is a diagnostic control, not a deployable detector. The report
   states this. For the equal-effort union, the A1 and A6 rows are rescored with the same fixed-partner definition, so
   that all of a1's candidates are scored under one definition.

3. **a2 under attack:** each source row gets **one fixed draw from the clean benign bank** (bank_benign: 20,000 clean
   validation rows predicted benign, the same bank as Stage A). The index is seeded by year, seed and "a2fixed". The
   attacker and the evaluation use the same draw.
   Why: Stage A draws a2's attention from the bank of the row's *predicted* class. A row counts in the headline
   quantity, and in every TPR, only if the attack flipped it to benign, and for such a row Stage A's rule already
   picks the benign bank. A fixed benign-bank draw is therefore the Stage A rule on every row that can count. It also
   removes two problems. First, the attacker's loss does not jump when the predicted class changes during the attack.
   Second, the evaluation does not depend on the order of a random stream over classes. Rows that are not flipped get
   a benign-bank vector instead of their class bank. That changes their score but not any reported quantity, because
   unflipped rows are neither successful nor counted in a TPR. Like Stage A, the fixed draw uses the **predicted
   class (benign)**, not the true class, so a2 needs no oracle label.

4. **Control b:** the GradientSHAP part of the stand-in uses **m = 8 samples** (fixed draws per step, double
   backprop). Occlusion is computed exactly and is differentiable. The evaluation always uses the true b features (as
   in Stage A: occlusion with training-mean masking, GradientSHAP with 100 background rows and 50 samples). A timing
   pilot runs first. If the estimated full run for b (both datasets, 3 seeds, 3 settings) exceeds **12 h**, b is
   excluded from the full runs, and the exclusion is recorded in handoff\32_restructure_plan.md and the report.

5. **2017 rows:** the adaptive attacks on 2017 use only the **7 common classes (1,044 of 1,494 source rows per
   seed)**. 2018 already has only the 7 common classes (985 rows).

6. **Budget and settings:** eps 1 only. Each control faces 3 settings (lambda 1, lambda 100, normalised gradient with
   w 0.5), 3 seeds and 2 datasets, with the PGD settings of A6 and A8 (100 steps, 1 random restart, kappa 10, safety
   margin and stand-in temperature as configured in the saved run configs). The detection threshold is always that
   of the refit control at 5% FPR, and detection is always evaluated with the true features.

## Pre-registered rule (copied from plan 4.6)

**Headline quantity.** Equal-effort union: for each source row, it counts as successful and undetected against a
detector if any of that detector's candidates flipped the prediction to benign and stayed below that detector's
threshold.
- D faces A1 + A6 x 3.
- D+ faces A1 + A8 x 3.
- Each control faces A1 + its own adaptive attacker x 3.

Reported per control: the share of successful and undetected source rows as **control minus D** and **control minus
D+**, on **V1, eps 1, per dataset**, pooled over seeds with the paired bootstrap 95% interval (1,000 draws, source
rows resampled within seed), and **per seed**. All other quantities (paired TPR differences, V0, V2, per setting)
are secondary.

**Outcomes:**
- **Outcome A (the conclusion stands and becomes stronger):** every control under its own adaptive attacker has a
  union difference to D that is near zero or no better than D+'s. The paper can then say that no extra score family
  we tested, with or without attention, survives an attacker that models it. The Limitations paragraph "Controls for
  D+" is removed.
- **Outcome B (the conclusion changes):** a control without attention (b, or more importantly c2, which faces exact
  gradients) keeps a clearly larger union gain over D than D+ does on 2017. "Clearly" means the control minus D+
  interval excludes 0 in the control's favour, with the same sign in all three seeds. Attention is then dispensable
  even adaptively, and the recommendation becomes "use cheap logit and distance statistics", not consistency
  monitoring. Title and abstract change.
- **Outcome C (the conclusion changes in the other direction):** D+ keeps its 2017 gain while a1, a2, b and c2 fall
  to the level of D. Pairing a row's own attention with its attribution would then give some adaptive robustness,
  which reverses the current reading. Before that could be claimed, an exact-gradient attacker on D+ would be needed,
  because A8 uses stand-ins and c2's attacker does not.
- **Outcome D (adds a caution):** a control does worse than D under its adaptive attacker (more undetected rows).
  Extra scores then add attack surface. This gets one sentence in the Discussion.

Results on 2018 alone cannot change the conclusion: D alone leaves only 0.023 of the source rows successful and
undetected (the floor effect).

**Caveat stated in advance:** the attackers are not equally strong. c0, c1 and c2 face exact gradients. a1, a2 and
D+ face stand-ins with soft ranks and IG-8. b faces stand-ins with IG-8 and an 8-sample GradientSHAP. The stand-in
quality (R^2 per feature) of every control is reported before any full attack.

## Stop conditions

- gate0b fails: a refit threshold differs from the saved `thr_*` by more than 1e-5, or a validation AUROC differs
  from stageA_defenders.csv by more than 1e-9.
- The validity checker rejects any constrained adversarial row (smoke or pilot).
- The b pilot estimate exceeds 12 h: b is excluded, and the other controls go on.

## Notes added after the preparation runs (2026-10-29, about 07:45 Amman time (UTC+3); the rule above is unchanged)

- Decision 4 (b cap): **b stays in.** At chunk 100, b's peak was 11.0 GB on the 8 GB card. Windows spilled the excess
  into system memory without an OOM, so the first pilot estimated 36 to 39 h per dataset. At chunk 50, the peak is
  5.6 GB and the estimate is 4.6 h (2017) + 4.4 h (2018) = 9.0 h, under the 12 h cap. The pilot now treats a peak
  above 7.5 GB as an OOM. handoff\32_restructure_plan.md was not changed because b was not excluded.
- Decision 2 (a1): the consequence is measured, not assumed. Under the fixed clean-partner definition, a1's TPR on
  the non-adaptive A6 lambda 1 rows (2017, V1) is 0.669, against 0.782 under the Stage A definition (D: 0.589). For
  the a1 comparison, the matching non-adaptive baseline is therefore the fixed-definition one. The A1 and A6 rows are
  rescored with it in the union, as decided. Details: handoff\33_stageB_prep.md, section 5.

## Deviation: control b (2026-10-30, 00:13 Amman time (UTC+3); the rule above is unchanged)

Control b was not run in full. The full 2017 run of b (batch 2, started 2026-10-29 22:28:39) spilled into shared GPU
memory at chunk 50: the Python process used 7.76 GB of dedicated GPU memory and 2.98 GB of shared (system) memory,
although the pilot at chunk 50 had peaked at 5.6 GB. The run was stopped after 53 minutes (2026-10-29 23:21:40), with
no b rows written. Chunk 25 would need an estimated 12.6 h (6.0 h for 2017 + 6.5 h for 2018, from the pilot), over the
12 h cap of decision 4. b is missing for this reason only.

What had been seen when: the decision to stop b (23:21) and the first version of handoff\34_stageB.md were made before
any Stage B summary existed. During batch 1, the run log printed one line per run (attack success rate and the share
of rows successful and undetected by the attacked control alone), and these lines were seen; no comparison with D or
D+ and no interval had been computed. Task 35 ordered the summary run (step 1) before this note (step 2), so the
union and TPR summary tables (results\stage9\adaptive\summary_union.csv, written 00:12) had been computed and read
before this note was written at 00:13. This note does not use them: it states only the memory facts above, which
were recorded in handoff\34_stageB.md at 23:22 on 2026-10-29.
