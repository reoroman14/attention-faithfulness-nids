# Shared setup for the Scientific Reports generators (Task 46). The paper_v3 generators (which in turn import the old
# generators in paper/) are imported as modules with their output directory redirected to paper_srep/generated, so
# nothing in paper/ or paper_v3/ is written. Read-only on results/. Every number is read from a result file.
#
# Import this module before any paper_v3 generator module: the v3 modules bind GEN at import time.
import sys
from pathlib import Path

sys.dont_write_bytecode = True                    # importing the v3 and old generators must not write __pycache__ there

SREP = Path(__file__).resolve().parents[1]
ROOT = SREP.parent
V3 = ROOT / "paper_v3"
GEN = SREP / "generated"

for p in (str(V3 / "gen"),):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402  (paper_v3/gen/common.py)

common.GEN = GEN

# ---------------------------------------------------------------- display items of the manuscript (at most 8)
# Order = order of first citation in the text = numbering. Figures and tables are numbered separately.
MAIN_FIGURES = [("f_pipeline", "mainFigPipeline"), ("f_headline", "mainFigHeadline")]
MAIN_TABLES = [("t_clean", "mainTabClean"), ("t_attn_flip", "mainTabAttn"), ("t_attacks", "mainTabAttacks"),
               ("t_dplus_ablation", "mainTabControls"), ("t_union_adaptive", "mainTabUnion"),
               ("t_neardup_compact", "mainTabNeardup")]

# ---------------------------------------------------------------- Supplementary Information
SUPP_NOTES = [("si_impl", "suppNoteImpl", "Implementation details"),
              ("si_rand", "suppNoteRand", "Parameter randomisation check"),
              ("si_neardup", "suppNoteNeardup", "Near-duplicate analysis"),
              ("si_results", "suppNoteResults", "Supplementary results"),
              ("si_limits", "suppNoteLimits", "Further limitations")]
# (table file, macro). The SI prints the tables in this order, so its automatic numbering S1, S2, ... equals the macros.
SUPP_TABLES = [("t_data", "suppTabData"), ("t_constraints", "suppTabConstraints"), ("t_impl", "suppTabImpl"),
               ("t_randomisation", "suppTabRandomisation"), ("t_limitations_data", "suppTabLimData"),
               ("t_limitations_stages", "suppTabLimStages"), ("t_neardup_split", "suppTabNeardupSplit"),
               ("t_sensitivity_2017_attacks", "suppTabSensAttSeventeen"),
               ("t_sensitivity_2017_defenders", "suppTabSensDefSeventeen"),
               ("t_sensitivity_2018_attacks", "suppTabSensAttEighteen"),
               ("t_sensitivity_2018_defenders", "suppTabSensDefEighteen"),
               ("t_evasion", "suppTabEvasion"), ("t_clean_agreement", "suppTabCleanAgreement"),
               ("t_clean_faith", "suppTabCleanFaith"), ("t_faith_attack", "suppTabFaithAttack"),
               ("t_defenders", "suppTabDefenders"), ("t_per_class", "suppTabPerClass"),
               ("t_per_class_v1", "suppTabPerClassVone"), ("t_union_a8", "suppTabUnionAeight"),
               ("t_a8x_setting", "suppTabAxSetting"), ("t_a8x_tpr", "suppTabAxTpr"), ("t_standin", "suppTabStandin"),
               ("t_runs", "suppTabRuns")]
SUPP_FIGURES = [("f_mass_ratio", "suppFigMassRatio"), ("f_evasion_eps", "suppFigEvasion"),
                ("f_detection_tpr", "suppFigDetection"), ("f_per_class_heatmap", "suppFigPerClass")]

# References removed for this version (author decision, Task 46), after checking each entry for a link to asu.edu.jo.
REF_CUTS = ["albashayreh2025", "zahid2025", "khaleel2024", "lundberg2020", "erion2021"]
