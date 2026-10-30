# Task 40: decisions and outcome rule for A6x (selection-symmetric attacker on D)

Recorded: 2026-10-30 12:19 Amman time (UTC+3), before any A6x code was written and before any step 2 number was
computed. handoff\33_decisions.md and handoff\36_decisions_A8x.md are not changed.

## Purpose

Test whether the A8x reversal comes from the selection asymmetry: A8x selects its best iterate by the true D+ score,
while A6 keeps the Stage A rule. The reversal is D+ minus D = +0.075 [+0.061, +0.090] on 2017 V1, with D+ under
A1 + A8x x 3 and D under A1 + A6 x 3 (handoff\36_a8x.md).

**What the Stage A rule already does** (read from src\stage6d_final.py attack_chunk before this file was written):
- A6 already keeps, per row, an iterate that is flipped and below the threshold under the in-loop D score. That score
  is exact for D, because all four D features are differentiable and computed with feats_torch.
- Its key is 1e7 x (flipped and score < thr) + 1e6 x flipped + objective, where the objective is
  min(e, kappa) - lambda x relu(score - thr + safety).
- So A6 differs from the A8x rule only in two ways:
  - (i) within a class, A6 keeps the iterate with the highest objective, A8x the one with the lowest true score;
  - (ii) A8x adds the 1e-3 margin.

**Expected effect.** A6x should therefore change little. This expectation is recorded before any run; it does not
change the design or the rule.

## Design A6x

- **Attack.** A6 against the saved D of each seed (defenders_eps1.pkl), unchanged in every respect except selection:
  - settings lambda 1, lambda 100 and normalised gradient w 0.5;
  - 100 steps, 1 restart, kappa 10, safety margin 0.5, alpha0 0.25 with the cosine schedule;
  - eps 1, seeds 0, 1, 2;
  - the attack seeds of A6: rng_seed + 7300 + 100 + 17 s + sum of the character codes of the A6 name.
- **Selection.** At every step and for every row, the true D score is computed from the same tensors. The iterate is
  kept by the three-class key of A8x:
  - (1) flipped to benign and true D score < thr - 1e-3;
  - (2) flipped otherwise;
  - (3) not flipped.
  Within (1) and (2) the lowest true D score wins; within (3) the highest benign margin wins. The margin is 1e-3, as in
  Amendment 1.
- **Rows.** The common source rows of Stage B and A8x: 1,044 on 2017 and 985 on 2018 per seed. These are the rows of
  the union; the non-common test rows that Stage A also attacked are not needed.
- **Chunk 50**, as required by Task 40.
  - Stage A ran A6 at chunk 500 over all test source rows, with the attack seed + chunk start as the generator seed of
    the random restart.
  - So that the random restart of every row is the same as in A6, A6x draws the restart noise exactly as A6 did: per
    original 500-row chunk of all test source rows, same generator seed, shape, device and dtype. It then gives each
    common row its own slice. The zero-start run (run 0) needs no noise.
  - What remains different from A6 is the batch shape of the float32 forward passes (50 instead of 500). This can
    change the gradient sign of near-zero components. Sanity check (s1) measures whether that matters.
- **Union for D:** A1 + A6x x 3. The union for D+ is unchanged: A1 + A8x x 3 from results\stage9\a8x.
- **Code:** a new file src\stage9d_a6x.py, with outputs in results\stage9\a6x\. No existing script or result file is
  changed. The watchdog, validity checker, memory log and resumable structure are reused from src\stage9c_a8x.py by
  import.

## Headline and outcome rule

**Headline:** 2017 V1, eps 1: D+ (A1 + A8x x 3, unchanged) minus D (A1 + A6x x 3).
- Paired bootstrap 95% interval, 1,000 draws, source rows resampled within seed, as before.
- Three per-seed values.

**Rule:**
- **S1:** the interval lies above 0 (lower bound > 0). The reversal does not depend on the selection asymmetry.
- **S2:** the interval contains 0 or lies below 0 (lower bound <= 0). Part or all of the reversal came from the
  asymmetry. The paper claim changes, and is reported as such:
  - to "no gain" (S2, interval contains 0);
  - to "gain" (S2, interval below 0, upper bound < 0).

S1 and S2 are exhaustive and exclusive given the interval.

**2018:** report only, in one sentence. Floor effect as before: D alone leaves 0.023 of the source rows successful and
undetected. 2018 cannot change the conclusion.

Also reported, without a rule:
- V0 and V2;
- per setting;
- the sensitivity union of D with A1 + A6 x 3 + A6x x 3 (unequal effort, marked);
- D under A6x minus D under A6 (the size of the selection effect on D alone).

## Step 2 analyses (CPU only, saved rows, no new attack)

### (a) Do A8x rows evade D?

For each year, setting and seed:
- the share of flipped A8x rows (results\stage9\a8x) that D leaves undetected (score_D < thr_D, both saved in the A8x
  rows);
- next to it, the same share for the flipped A6 rows of the same setting against D (Stage A/B rows);
- each also as a share of source rows.

What it would show:
- If A8x rows, which were not aimed at D, evade D about as often as the A6 rows that were aimed at D, or more often,
  then A8x is at least as strong on the four shared scores as A6. In that case part of the reversal could be optimiser
  strength rather than the consistency scores, and the A6x run is the direct test of that.
- If A8x rows evade D clearly less often than A6 rows, A8x's strength against D+ comes from the consistency-score terms
  and not from a stronger attack on the shared scores.
- This is descriptive. It has no rule and decides nothing.

### (b) Controls against D+ under A8x

For every control c0, c1, c2, a1, a2 (Stage B rows): control minus D+ under A8x.
- Union share, V1, eps 1, per year.
- Paired bootstrap 95% interval, 1,000 draws within seed, and per-seed values.
- V0 and V2 in a secondary table.
- Each row states the attacker type: exact for c0, c1, c2; stand-in for a1, a2.

What it would show: whether a control without attention (c0, c1, c2: exact attackers, so roughly equal strength)
leaves fewer rows undetected than D+ under its exact-objective attacker. For a1 and a2 the comparison is not at equal
attacker strength, and the report says so. Descriptive; no rule.

Both go to results\stage9\a8x\summary_extra.csv, which is a new file.

## Sanity checks (before the pilot; a failure stops the work)

- **(s1) Selection off reproduces A6.** With the Stage A key in place of the A6x key, A6x reproduces the saved A6 rows
  of Stage B (att_eps1_A6_lam1_X.npz, seed 0) on 200 common rows of 2017 and of 2018 (the first 200 in order) to
  within 1e-5 (max absolute difference of the adversarial feature vectors), at chunk 50 with the A6 restart noise.
  - If it fails, the same 200 rows are run once more at the original partition (chunk 500 over all test rows, as in
    Stage A), for diagnosis only, and the work stops and is reported.
- **(s2) The true D score inside A6x equals the saved D score.** On the same 200 rows, the four D features and the D
  decision computed inside A6x for the saved A6 rows equal the saved values:
  - mahalanobis_benign to relative 1e-3;
  - D decision to absolute 1e-3;
  - msp, margin and knn5_benign to 1e-5.
  These are the tolerances of Amendment 1 in handoff\36_decisions_A8x.md.
- **Margin information:** how many selection decisions differ with and without the 1e-3 margin on the 200 rows.

## Memory limits and run limits

- **Full-size pilot per dataset:** all common rows, 5 steps, then the scoring phase, chunk 50. Logged:
  torch.cuda.max_memory_reserved and the Windows counter "GPU Process Memory Shared Usage".
  - Go only if shared usage stays below 0.5 GB and reserved below 6.5 GB.
  - If not, stop and report; the chunk size is not changed.
- **Driver:** 18 runs, resumable, --max-hours 3 per invocation.
- **Before the runs:** if the pilot projects any single run above 2 h, stop and report.

## Stop conditions

- Any invalid row from the validity checker.
- A failed sanity check (s1) or (s2).
- Shared GPU memory above 1 GB in any poll. This is stricter than for A8x: one poll is enough.
- A pilot that fails the memory limits.
- Any run projected or measured above 2 h.

## Amendment 1 (chunk size)

Recorded: 2026-10-30 12:39 Amman time (UTC+3), before any further run. No A6x attack row exists.

**Reason.**
- Sanity check (s1) failed at chunk 50 on 2017 seed 0: 3 of 200 rows changed path, with a max abs difference of
  1.4e+04 in raw feature units.
- The same code at the Stage A partition (chunk 500 over all test source rows) reproduces the saved A6 rows with
  difference 0.000 on all 200 rows.
- The cause is the batch shape of the float32 forward passes.

**Change.**
- A6x runs at chunk 500 over all test source rows, as A6 did in Stage A: 2017 has 1,494 rows, of which 1,044 are the
  common rows of the union; 2018 has 985 rows.
- The union uses the common rows only, as before.
- The restart noise is then A6's own draw per chunk; the slicing described in "Design A6x" is no longer needed.

**Unchanged.**
- Check (s1) keeps tolerance 1e-5 at chunk 500, on both years.
- All other checks, the selection key, the margin 1e-3, the outcome rule S1/S2 and the stop conditions.

**Memory.**
- A new full-size pilot at chunk 500: all test source rows, 5 steps, then the scoring phase.
- Limits: shared usage below 0.5 GB and torch max reserved below 6.5 GB.
- If the pilot fails the limits, stop and report; no other chunk size is chosen.
