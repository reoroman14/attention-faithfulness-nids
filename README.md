# Attention explanations and consistency monitoring in transformer intrusion detection under protocol-constrained adaptive attacks

Code, configuration files, summary result files and the generators of every table, figure and number of the paper
"Attention explanations and consistency monitoring in transformer intrusion detection under protocol-constrained
adaptive attacks" (submitted to Scientific Reports).

**What the project is.** A feature-tokenizer transformer (FT-Transformer) is trained to classify network flows of
two corrected intrusion datasets, LYCOS-IDS2017 and LycoS-Unicas-IDS2018, with three seeds. The project measures:
- whether attention is a faithful explanation of these flow classifiers, on clean flows and under protocol-constrained
  attacks;
- whether attention-attribution consistency scores improve a confidence-and-distance attack detector. The detector is
  evaluated against an attacker that ignores the scores (A6), one that models them through differentiable stand-ins
  (A8), and one that optimises the detector's exact score (A8x). A6x gives the attacker of the base detector the
  iterate selection of A8x. Control detectors separate the role of attention.

**Access.** The repository is publicly available, with a tagged release (tag `submitted-scientific-reports-2026-10`).

## Folder layout

| Entry | Content |
|---|---|
| `src/` | The pipeline, stages 1 to 9 (data, models, explanations, constraints and attacks, detectors, the 2018 dataset, near-duplicate analysis, controls for D+, Stage B adaptive attacks on the controls, A8x and A6x), the FT-Transformer model (`src/models/`) and the configuration (`src/config.py`) |
| `results/` | Summary result files only: CSV tables (means, intervals, counts and rates per seed, class, method, attack or budget), small metadata files, run logs and the two run configurations read by stage 9 (`results/runs/*/config.json`). `results/stage9/` holds the summaries of the controls for D+ (`ablation/`), of Stage B (`adaptive/`), of A8x (`a8x/`) and of A6x (`a6x/`) |
| `paper/` | Generators of the earlier version of the paper (numbers, tables, figures), imported by the newer generators, and their generated outputs (`paper/generated/`) |
| `paper_v3/` | Restructured manuscript: sources (`main.tex`, `sections/`, `references.bib`), generators (`gen/`) and generated tables, figures and number macros (`generated/`) |
| `paper_srep/` | Scientific Reports version: sources (`main.tex`, `supplementary.tex`, `decisions.tex`, `sections/`, `si/`, `references.bib`), generators (`gen/`) and generated outputs (`generated/`) |
| `decisions/` | The decision records with the outcome rules recorded before the runs (see below) |
| `requirements.txt`, `LICENSE`, `.gitignore` | Environment, licence (MIT) and ignore rules |

## Environment

- Python 3.14.
- The packages and versions are in `requirements.txt`.
- torch 2.11.0, CUDA 12.8 build:
  `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`
- The reported runs used one Windows PC with one GPU with 8 GB of memory and 32 GB of RAM.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Regenerating tables, figures and numbers

Every number in the paper is a LaTeX macro, written by a generator from the result files. Each macro line in
`numbers.tex` is preceded by a comment that names its source file. Run the commands from the repository root.

**Scientific Reports version** (`paper_srep/generated/`). The `paper_srep` generators import the `paper_v3` generators, which import those in `paper/`; each writes only into its own `generated/` folder.

```
.venv\Scripts\python.exe paper_srep\gen\make_numbers_srep.py    # numbers.tex, numbers_index.json, srep_numbers.tex
.venv\Scripts\python.exe paper_srep\gen\make_tables_srep.py     # every table (main_t_*.tex, si_t_*.tex)
.venv\Scripts\python.exe paper_srep\gen\make_figures_srep.py    # every figure (f_*.pdf)
.venv\Scripts\python.exe paper_srep\gen\make_bib_srep.py        # paper_srep\references.bib (see the note below)
.venv\Scripts\python.exe paper_srep\gen\build_srep.py           # flat main.tex and supplementary.tex
.venv\Scripts\python.exe paper_srep\gen\checks_srep.py          # journal limits and consistency checks
```

**Restructured version** (`paper_v3/generated/`):

```
.venv\Scripts\python.exe paper_v3\gen\make_numbers_v3.py
.venv\Scripts\python.exe paper_v3\gen\make_tables_v3.py
.venv\Scripts\python.exe paper_v3\gen\make_figures_v3.py
.venv\Scripts\python.exe paper_v3\gen\build.py
.venv\Scripts\python.exe paper_v3\gen\checks_v3.py
```

**Earlier version** (`paper/generated/`):

```
.venv\Scripts\python.exe paper\make_numbers_tex.py
.venv\Scripts\python.exe paper\make_tables_v2.py
.venv\Scripts\python.exe paper\make_figures_v2.py
```

**Notes on the generators:**
- **Build and check scripts write or read fixed local folders.** `build_srep.py` writes the flat submission files to
  `C:\attn_submission_srep` and `C:\attn_submission_srep_si`, and `build.py` writes to `C:\attn_submission_v3`; both
  refuse to write into folders that hold other files. The check scripts read those folders (`checks_v3.py` also reads
  the earlier submission `C:\attn_submission_cyse\main.tex`). Change the paths at the top of these scripts on
  another machine.
- **`make_bib_srep.py` reads a local file.** It takes the repository address from the named copy of the earlier
  submission, `C:\attn_submission\delete\main.tex`, which is not in this repository. The `paper_srep/references.bib`
  it wrote is included.
- **Font workaround.** On the machine used for the paper, an application-control policy blocked the compiled fontTools
  modules that matplotlib's PDF backend imports. `paper_v3/gen/common.py` therefore loads the pure-Python fontTools
  sources inside its own process. `paper/make_figures_v2.py` run on its own may fail on such a machine and works
  elsewhere.
- **Title page.** In this repository copy of `paper_srep/main.tex` the e-mail addresses of the title page are
  removed; the submitted manuscript has them.

### What can be regenerated from this repository alone

Each generator function was run on a copy of this folder, without the per-row files.

**Runs from the summary files** (42 of 55 functions):
- Numbers: the old macro groups params, neardup, clean, attacks, defenders, flip_split, appendix, audit,
  randomisation and ablation; the A8x union, TPR and extra macros (`union_a8x`, `tpr_a8x`, `extra_a8x`); and the A6x
  outcome phrase (`srep_extra`).
- Tables:
  - from `paper/make_tables_v2.py`: t1 to t6, t10, t12, t14 and t15;
  - from paper_v3: t_data, t_clean, t_attn_attack, t_flip, t_attacks, t_union_adaptive, t_neardup_compact,
    t_union_a8, t_a8x_setting, t_a8x_tpr, t_runs, t_standin and t_impl;
  - from paper_srep: the merged table t_attn_flip.
  - Each of these regenerated tables is identical to the included version, apart from the date line, except
    t_constraints (see below).
- Figures: the pipeline figure, the headline figure (f_headline), the mass-ratio figure and the evasion-by-budget
  figure.

**Needs per-row files that are not included** (13 functions). The included generated files are the versions used in
the paper.

| Generator function | Missing input |
|---|---|
| numbers `datasets` | `results/stage6/final/seed0/negatives.pkl` |
| numbers `review` | `results/stage3/selection.parquet` |
| numbers `stage_b`, `a8x_runs`, `a6x` | the per-row attack files `rows_*.parquet` of `results/stage9/adaptive`, `a8x` and `a6x` (row counts, validity, run times) |
| tables t7, t8, t9, t11, t13 of `paper/make_tables_v2.py` | per-row `rows_*.parquet` / `*.npz` of stages 5 to 8 |
| table t_per_class_v1 | `results/stage3/selection.parquet` |
| figures f4 (detection TPR) and f5 (per-class heatmap) of `paper/make_figures_v2.py` | per-row files of stages 6 and 7 |

- **Whole scripts stop.** Because of these functions, `make_numbers_tex.py`, `make_numbers_v3.py`,
  `make_numbers_srep.py` and the table and figure scripts stop with an error when run as a whole without the per-row
  files.
- **t_constraints.** Its validity count is computed from per-row files, so a regeneration without them writes
  "0 of 0". The included `t_constraints.tex` is correct.
- **Full regeneration** of every file needs the per-row files, which are produced by rerunning the pipeline below on
  the datasets.

## Rerunning the pipeline

The pipeline needs the datasets (see below) under `data/`, as described in `src/config.py` and
`src/stage7a_inspect_2018.py`:
- `data/pcap_lycos/<Day>-WorkingHours.pcap_lycos.csv`;
- `data/labelling.py` and `data/python_logs/labelling_info.log`, from the LYCOS-IDS2017 distribution;
- `data/lycos-unicas-ids2018/LycoS-Unicas-IDS2018.csv`.

Each stage writes to `results/`.

**LYCOS-IDS2017:**

```
python src/run_labelling.py
python src/stage1_data.py
python src/stage2_train.py
python src/stage3_explain.py
python src/stage3c_diagnostics.py
python src/stage4_constraints.py
python src/stage4c_static_control.py
python src/stage5_fullrun.py
python src/stage6d_final.py
python src/stage6e_strong.py
python src/stage6e_strong.py --set stage6e.seeds_e=[1] --set stage6e.chunk_candidates=[100,50] --set stage6e.out_dir_e=results/stage6/strong_s8c_staging --set stage6e.progress_log_e=false
```

The last command runs A8 for seed 1. Copy `results/stage6/strong_s8c_staging/seed1/` into `results/stage6/strong/`,
then check it with:
- `python src/stage8c_verify_a8_seed1.py`;
- `python src/stage8c_compare.py`.

**LycoS-Unicas-IDS2018:**

```
python src/stage7a_inspect_2018.py
python src/stage7b_prepare_train_2018.py
python src/stage7c_explain_2018.py
python src/stage7d_attack_2018.py
python src/stage7e_defense_2018.py
```

**Both datasets, near-duplicate analysis** (CPU, on the saved per-row results):

```
python src/stage8_neardup.py
python src/stage8_sensitivity.py
python src/stage8b_clean_v1.py
python src/stage8d_flip_split.py
```

**Stage 9, both datasets** (run as modules from the repository root; the attack phases use the GPU):

```
python -m src.stage9_dplus_ablation gate0
python -m src.stage9_dplus_ablation stageA
python -m src.stage9_dplus_ablation summary
python -m src.stage9b_dplus_adaptive gate0b
python -m src.stage9b_dplus_adaptive standin_quality
python -m src.stage9b_dplus_adaptive pilot
python -m src.stage9b_dplus_adaptive attack
python -m src.stage9b_dplus_adaptive summary
python -m src.stage9c_a8x sanity
python -m src.stage9c_a8x pilot
python -m src.stage9c_a8x attack
python -m src.stage9c_a8x summary
python -m src.stage9d_a6x extra
python -m src.stage9d_a6x sanity
python -m src.stage9d_a6x pilot
python -m src.stage9d_a6x attack
python -m src.stage9d_a6x summary
```

- **Stage 9 scripts:**
  - `src/stage9_dplus_ablation.py`: the controls for D+;
  - `src/stage9b_dplus_adaptive.py`: Stage B, the adaptive attackers of the controls;
  - `src/stage9c_a8x.py`: A8x;
  - `src/stage9d_a6x.py`: A6x.
- **Options and phases:** `--years` and `--max-hours` are optional. Each phase is described in the comment at the top
  of its script.
- **Gates:**
  - `gate0` reproduces the published detectors from `results/runs/20261005-170553_stage6d_final/config.json` (2017)
    and `results/runs/20261006-144518_stage7e_defense_2018/config.json` (2018).
  - The sanity phases of A8x and A6x stop on any failure.
- **Decision records:** code comments refer to them under `handoff/`. In this repository they are in `decisions/`.

**Pilot and superseded scripts**, kept for provenance and not part of the final pipeline:
- `src/stage4_attack.py` (stage 4 pilot);
- `src/stage4b_followup.py`;
- `src/stage6_defense.py`, `src/stage6b_defense.py` and `src/stage6c_defense.py`;
- `src/check_stage2.py`, `src/smoke_test.py` and `src/fix_xgb_checkpoint.py`;
- `paper/make_figures.py`, an earlier version that is still imported for its style settings.

`src/stage4_attack.py`, `src/stage4_constraints.py` and `src/stage4c_static_control.py` are also imported by later
stages.

## Decision records and outcome rules

The attackers A8x and A6x and the adaptive attacks on the control detectors (Stage B) were designed after the earlier
results were known. Their outcome rules and amendments were recorded before the runs, before any attack row of the
attacker concerned existed:

| Record | Content |
|---|---|
| `decisions/33_decisions.md` | Stage B: the controls attacked, their attackers, the headline quantity and the rule for the comparisons (control minus D, control minus D+), and the recorded deviation for control b, which was not attacked adaptively |
| `decisions/36_decisions_A8x.md` | A8x: design, the outcome rule (the 95% interval of D+ minus D on 2017, row set V1, as the deciding quantity; an interval containing 0 or lying above 0 means that the gain of D+ measured with A8 did not hold) and Amendment 1 (sanity-check tolerance and the 1e-3 selection margin) |
| `decisions/38_decisions_A6x.md` | A6x: design, the outcome rule (an interval above 0 means that the comparison does not depend on the selection rule) and the amendment on the chunk size |

The run records (`results/stage9/a8x/log_sanity*.txt`, `results/stage9/a6x/log_sanity*.txt` and the pilot and attack
logs) show the order of sanity checks, pilots and attacks.

## Not included, and why

- **Raw datasets and per-flow rows.** They are not redistributed. Obtain the datasets from their original sources and
  cite them as their owners request:
  - LYCOS-IDS2017 (Rosay et al., 2021; 2022): https://lycos-ids.univ-lemans.fr
  - LycoS-Unicas-IDS2018 (Cantone et al., 2024): https://github.com/MarcoCantone/LycoS-Unicas-IDS2018
  - CIC-IDS2017 and CSE-CIC-IDS2018 (Sharafaldin, Habibi Lashkari and Ghorbani, ICISSP 2018,
    doi:10.5220/0006639801080116). CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html. CSE-CIC-IDS2018 is
    distributed through AWS Open Data, https://registry.opendata.aws/cse-cic-ids2018, under terms that require citation
    and a link to that page.
- **Per-row intermediate files** (`*.parquet`, `*.npz`, `*.npy`, `*.pkl`): attacked rows, per-row explanations and
  scores, selections, negatives and detector inputs. They are not included because of their size: the stage 9 per-row
  files alone take about 0.9 GB, and the full `results/` folder about 3 GB.
- **Model checkpoints** (`*.pt`, the XGBoost model files). Too large, and reproducible from the pipeline.
- **The built submission folders** (flat `main.tex`, `supplementary.tex` and their figure copies). The build scripts
  produce them from the sources here.
- **Smoke-test and pilot outputs** that the paper does not use, for example `results/stage9/adaptive_smoke/`.
- **Development notes and handoff reports**, except the three decision records.

## AI use

During the preparation of this work the authors used Claude Opus 5.5 (Anthropic), through Claude Code, in order to
write and debug the analysis code and to draft and edit the manuscript text under the direction of the first author,
and Claude Sonnet 5.5 (Anthropic), used by the first author through the Claude app in a separate session, in order to
review intermediate results and to draft the instructions given to Claude Code.
After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the
content of the published article.
The authors also checked the numerical results with scripts, and the first author checked them manually against the
result files.
The analysis code added in this version (the adaptive attacks on the control detectors, the attackers A8x and A6x and
their summaries) was also written by Claude Opus 5.5 through Claude Code under the direction of the first author.

## License

MIT, see `LICENSE`.
