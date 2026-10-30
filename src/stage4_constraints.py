# Stage 4 constraint model for protocol-valid adversarial flows in RAW feature space.
#
# Contents
#   SPEC                feature-level spec (family, role, direction, integer, derived-from, rationale); written
#                       to results/stage4_constraints.csv, which is the spec of record for the validity checker.
#   TorchPreprocessor   differentiable re-implementation of src/preprocess.py (log1p, standardise, clip +-10).
#   AttackContext       per-row clean values, budgets, delta boxes and freeze masks for one eps.
#   derive()            recomputes the derived features from their bases (differentiable, delta form).
#   project()           repair + backtracking projection used after every attack step (torch, float64).
#   check_validity()    INDEPENDENT numpy validity checker: re-derives every relation from the CSV spec with
#                       its own formulas (it never calls derive() or project()).
#
# Roles
#   control    directly perturbed by the attacker (budgeted, direction-constrained)
#   derived    recomputed from its base features (not budgeted; its change is reported)
#   immutable  never changes (server-side, protocol-defining, or not modelled consistently)
# Budget per control feature j and row: B_j = eps * |r_j| if r_j != 0, else eps * s_j, where s_j is the train
# median of the non-zero |values| of feature j (the "small absolute floor" for zero features, scaled by eps).
#
# Usage:
#   .venv\Scripts\python.exe -m src.stage4_constraints   # writes the CSV, runs the preprocessing check and a
#                                                        # clean-row self-test of the checker
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.preprocess import Preprocessor
from src.utils import RESULTS, resolve

CONSTRAINTS_CSV = RESULTS / "stage4_constraints.csv"
WIN_MAX = 65535.0          # TCP window field is 16 bits (no window scaling in the initial SYN)
HDR_MAX = 60.0             # max TCP header length in bytes (20 + 40 bytes of options)
US = 1e6                   # durations are in microseconds; rates are per second

# Attack class -> class family (used for the semantic freezes).
CLASS_FAMILY = {"ddos": "volumetric_dos", "dos_hulk": "volumetric_dos", "dos_goldeneye": "volumetric_dos",
                "dos_slowloris": "slow_dos", "dos_slowhttptest": "slow_dos",
                "ftp_patator": "brute_force", "ssh_patator": "brute_force",
                "webattack": "web", "bot": "botnet", "portscan": "scan", "benign": "benign"}
TIMING_CONTROLS = ["flow_duration", "fwd_iat_tot", "fwd_iat_max", "fwd_iat_min", "fwd_iat_std",
                   "iat_max", "iat_min", "iat_std"]
# Controls frozen per class family because they ARE the attack mechanism.
FAMILY_FREEZE = {
    # Slow DoS: the number and pacing of the partial-request packets is the attack (slowloris keeps sockets
    # open by trickling header lines just under the server timeout). Packet count, ACK count and all timing
    # stay as observed; payload padding, TCP options and the window remain available.
    "slow_dos": ["fwd_pkt_cnt", "flag_ack"] + TIMING_CONTROLS,
    # Port scan: a probe is a fixed SYN / response exchange and its timing is the target's RTT, which the
    # scanner does not control. Probe payload (nmap --data-length), TCP options and the window remain.
    "scan": ["fwd_pkt_cnt", "flag_ack"] + TIMING_CONTROLS,
}

# (feature, family, role, direction, integer, derived_from, rationale)
_S = [
    ("flow_duration", "IAT/timing", "control", "add-only", True, "",
     "attacker can delay its own packets, which only lengthens the flow; it cannot make the server answer faster"),
    ("down_up_ratio", "rates/ratios", "derived", "either", False, "bwd_pkt_cnt / fwd_pkt_cnt",
     "verified: equals bwd_pkt_cnt / fwd_pkt_cnt (0 when fwd_pkt_cnt = 0) on 100% of train rows"),
    ("pkt_len_max", "packet length stats", "derived", "either", True, "max(fwd_pkt_len_max, bwd_pkt_len_max)",
     "verified on 100% of train rows"),
    ("pkt_len_min", "packet length stats", "derived", "either", True,
     "fwd_pkt_len_min if clean pkt_len_min == clean fwd_pkt_len_min, else min(clean pkt_len_min, fwd_pkt_len_min)",
     "bwd_pkt_len_min is 0 on 99.8% of train rows (extractor quirk), so the backward minimum is only known "
     "through pkt_len_min; this rule is exact given the observed values"),
    ("pkt_len_mean", "packet length stats", "derived", "either", False,
     "(fwd_pkt_len_tot + bwd_pkt_len_tot) / (fwd_pkt_cnt + bwd_pkt_cnt)", "verified on 100% of train rows"),
    ("pkt_len_var", "packet length stats", "derived", "either", False,
     "pooled sample variance of the fwd and bwd length families (ddof 1)", "verified on 100% of train rows"),
    ("pkt_len_std", "packet length stats", "derived", "either", False, "sqrt(pkt_len_var)",
     "verified on 100% of train rows"),
    ("bytes_per_s", "rates/ratios", "derived", "either", False,
     "(fwd_pkt_len_tot + bwd_pkt_len_tot) / flow_duration * 1e6 (0 if flow_duration = 0)",
     "verified on 100% of train rows"),
    ("pkt_per_s", "rates/ratios", "derived", "either", False, "(fwd_pkt_cnt + bwd_pkt_cnt) / flow_duration * 1e6",
     "verified on 100% of train rows"),
    ("fwd_pkt_per_s", "rates/ratios", "derived", "either", False, "fwd_pkt_cnt / flow_duration * 1e6",
     "verified on 100% of train rows"),
    ("bwd_pkt_per_s", "rates/ratios", "derived", "either", False, "bwd_pkt_cnt / flow_duration * 1e6",
     "verified on 100% of train rows"),
    ("fwd_pkt_cnt", "packet counts", "control", "add-only", True, "",
     "attacker can inject extra forward packets; it cannot remove packets the attack needs"),
    ("fwd_pkt_len_tot", "byte counts", "control", "add-only", True, "",
     "payload padding and injected packets only add bytes; the attack payload cannot be removed"),
    ("fwd_pkt_len_max", "packet length stats", "control", "add-only", True, "",
     "padding or a larger injected packet can only raise the maximum"),
    ("fwd_pkt_len_min", "packet length stats", "control", "either", True, "",
     "padding raises it, an injected small packet lowers it; it can only decrease when packets are added, and "
     "it cannot rise when it is also the flow minimum (the backward minimum is unknown)"),
    ("fwd_pkt_len_mean", "packet length stats", "derived", "either", False, "fwd_pkt_len_tot / fwd_pkt_cnt",
     "verified on 100% of train rows"),
    ("fwd_pkt_len_std", "packet length stats", "control", "either", False, "",
     "redistributing padding changes the spread; kept inside the exact feasibility bounds for (n, min, mean, max)"),
    ("fwd_pkt_hdr_len_tot", "window/header", "control", "add-only", True, "",
     "TCP options can be added and injected packets add headers; each added packet adds at least "
     "fwd_pkt_hdr_len_min bytes; at most 60 bytes per packet"),
    ("fwd_pkt_hdr_len_min", "window/header", "immutable", "none", True, "",
     "8 (UDP) / 20+ (TCP) is protocol-defining; injected packets are assumed to use at least this header length"),
    ("fwd_non_empty_pkt_cnt", "packet counts", "control", "add-only", True, "",
     "padding an empty packet or injecting a data packet adds non-empty packets; equals fwd_pkt_cnt iff "
     "fwd_pkt_len_min > 0 (verified)"),
    ("bwd_pkt_cnt", "packet counts", "immutable", "none", True, "", "backward packets are sent by the victim"),
    ("bwd_pkt_len_tot", "byte counts", "immutable", "none", True, "", "backward bytes are sent by the victim"),
    ("bwd_pkt_len_max", "packet length stats", "immutable", "none", True, "", "victim-side"),
    ("bwd_pkt_len_min", "packet length stats", "immutable", "none", True, "",
     "victim-side (and 0 on 99.8% of rows, an extractor quirk)"),
    ("bwd_pkt_len_mean", "packet length stats", "immutable", "none", False, "", "victim-side"),
    ("bwd_pkt_len_std", "packet length stats", "immutable", "none", False, "", "victim-side"),
    ("bwd_pkt_hdr_len_tot", "window/header", "immutable", "none", True, "", "victim-side"),
    ("bwd_pkt_hdr_len_min", "window/header", "immutable", "none", True, "", "victim-side"),
    ("bwd_non_empty_pkt_cnt", "packet counts", "immutable", "none", True, "", "victim-side"),
    ("iat_max", "IAT/timing", "control", "either", True, "",
     "delays lengthen gaps, injected packets split gaps; <= flow_duration and >= idle_max"),
    ("iat_min", "IAT/timing", "control", "either", True, "",
     "as iat_max; frozen when the clean value is negative (clock artefact)"),
    ("iat_mean", "IAT/timing", "derived", "either", False, "flow_duration / (fwd_pkt_cnt + bwd_pkt_cnt - 1)",
     "verified on 100% of train rows"),
    ("iat_std", "IAT/timing", "control", "either", False, "", "kept inside the feasibility bounds of the IAT family"),
    ("fwd_iat_tot", "IAT/timing", "control", "add-only", True, "",
     "time from first to last forward packet; delays only lengthen it; <= flow_duration (== when no bwd packets)"),
    ("fwd_iat_max", "IAT/timing", "control", "either", True, "", "delays and injected packets; <= fwd_iat_tot"),
    ("fwd_iat_min", "IAT/timing", "control", "either", True, "", "delays and injected packets"),
    ("fwd_iat_mean", "IAT/timing", "derived", "either", False, "fwd_iat_tot / (fwd_pkt_cnt - 1)",
     "verified on 100% of train rows"),
    ("fwd_iat_std", "IAT/timing", "control", "either", False, "", "kept inside the feasibility bounds"),
    ("bwd_iat_tot", "IAT/timing", "immutable", "none", True, "", "victim-side packet timing"),
    ("bwd_iat_max", "IAT/timing", "immutable", "none", True, "", "victim-side packet timing"),
    ("bwd_iat_min", "IAT/timing", "immutable", "none", True, "", "victim-side packet timing"),
    ("bwd_iat_mean", "IAT/timing", "immutable", "none", False, "", "victim-side packet timing"),
    ("bwd_iat_std", "IAT/timing", "immutable", "none", False, "", "victim-side packet timing"),
] + [(f"{k}_{s}", "active/idle", "immutable", "none", s != "std" and s != "mean", "",
      "depends on the 5 s idle threshold and per-packet timing that the flow statistics do not determine; "
      "held fixed (conservative: no direct attacker control)")
     for k in ("active", "idle") for s in ("max", "min", "mean", "std")] + [
    ("flag_syn", "flag counts", "immutable", "none", True, "", "handshake semantics, protocol-defining"),
    ("flag_fin", "flag counts", "immutable", "none", True, "", "teardown semantics, protocol-defining"),
    ("flag_rst", "flag counts", "immutable", "none", True, "", "teardown semantics, protocol-defining"),
    ("flag_ack", "flag counts", "control", "add-only", True, "",
     "injected packets may carry ACK; increase <= number of injected forward packets"),
    ("flag_psh", "flag counts", "derived", "either", True, "fwd_flag_psh + bwd_flag_psh",
     "verified on 100% of train rows"),
    ("fwd_flag_psh", "flag counts", "control", "add-only", True, "",
     "attacker sets PSH on its own data packets; <= fwd_non_empty_pkt_cnt (verified)"),
    ("bwd_flag_psh", "flag counts", "immutable", "none", True, "", "victim-side"),
    ("flag_cwr", "flag counts", "immutable", "none", True, "", "congestion signalling, protocol-defining"),
    ("flag_ece", "flag counts", "immutable", "none", True, "", "congestion signalling, protocol-defining"),
] + [(f"{d}_bulk_{s}_mean", "bulk", "immutable", "none", False, "",
      "bulk detection is a per-packet state machine (>= 4 consecutive data packets, 1 s gaps) that the flow "
      "statistics do not determine; held fixed (conservative)") for d in ("fwd", "bwd")
     for s in ("bytes", "pkt", "rate")] + [
    ("fwd_subflow_bytes_mean", "subflow", "derived", "either", False, "fwd_pkt_len_tot / n_sub",
     "n_sub = fwd_pkt_cnt / fwd_subflow_pkt_mean is an integer shared by all 4 subflow features (verified); "
     "held at its clean value (delays assumed not to create or merge subflows)"),
    ("fwd_subflow_pkt_mean", "subflow", "derived", "either", False, "fwd_pkt_cnt / n_sub", "see fwd_subflow_bytes_mean"),
    ("bwd_subflow_bytes_mean", "subflow", "immutable", "none", False, "bwd_pkt_len_tot / n_sub",
     "bwd bases and n_sub are fixed, so it never changes"),
    ("bwd_subflow_pkt_mean", "subflow", "immutable", "none", False, "bwd_pkt_cnt / n_sub",
     "bwd bases and n_sub are fixed, so it never changes"),
    ("fwd_tcp_init_win_bytes", "window/header", "control", "either", True, "",
     "attacker chooses its initial window (0..65535); frozen when -1 (no TCP handshake seen, e.g. UDP)"),
    ("bwd_tcp_init_win_bytes", "window/header", "immutable", "none", True, "", "victim-side"),
]
SPEC = {f: dict(family=fam, role=role, direction=d, integer=it, derived_from=df, rationale=why)
        for f, fam, role, d, it, df, why in _S}
CONTROLS = [f for f, s in SPEC.items() if s["role"] == "control"]
DERIVED = [f for f, s in SPEC.items() if s["role"] == "derived"]


# ---------------------------------------------------------------- spec CSV

def zero_floor_scales(train_df, feats):
    # Train median of the non-zero |values| per feature (budget floor for features that are 0 in the clean row).
    out = {}
    for f in feats:
        v = np.abs(train_df[f].to_numpy(dtype=np.float64))
        v = v[v > 0]
        out[f] = float(np.median(v)) if len(v) else 1.0
    return out


def write_spec_csv(feats, floors, path=CONSTRAINTS_CSV):
    assert set(SPEC) == set(feats), set(SPEC) ^ set(feats)
    rows = []
    for f in feats:
        s = SPEC[f]
        frozen = [fam for fam, lst in FAMILY_FREEZE.items() if f in lst]
        rows.append({"feature": f, "family": s["family"], "role": s["role"],
                     "mutable": "yes" if s["role"] != "immutable" else "no",
                     "direction": s["direction"] if s["role"] != "immutable" else "none",
                     "integer": "yes" if s["integer"] else "no",
                     "derived_from": s["derived_from"],
                     "budgeted": "yes" if s["role"] == "control" else "no",
                     "frozen_for_class_families": ";".join(frozen),
                     # Used as the budget floor for controls; for every feature it also normalises the
                     # reported relative perturbation of zero-valued features.
                     "zero_floor_scale": floors[f],
                     "rationale": s["rationale"]})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def load_spec_csv(path=CONSTRAINTS_CSV):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


# ---------------------------------------------------------------- differentiable preprocessing

class TorchPreprocessor(torch.nn.Module):
    # Same transform as Preprocessor.transform: log1p(max(x, 0)) on log features, standardise, clip.
    def __init__(self, pp, device):
        super().__init__()
        self.register_buffer("log_mask", torch.as_tensor(pp.log_mask, device=device))
        self.register_buffer("mean", torch.as_tensor(pp.mean, dtype=torch.float64, device=device))
        self.register_buffer("std", torch.as_tensor(pp.std, dtype=torch.float64, device=device))
        self.clip = pp.clip

    def forward(self, x):
        x = x.to(torch.float64)
        x = torch.where(self.log_mask, torch.log1p(x.clamp_min(0.0)), x)
        z = (x - self.mean) / self.std
        return z.clamp(-self.clip, self.clip) if self.clip is not None else z

    def inverse(self, z):
        # Raw values for standardised inputs (exact for unclipped values; used for the unconstrained reference).
        x = z.to(torch.float64) * self.std + self.mean
        return torch.where(self.log_mask, torch.expm1(x), x)


def preprocessing_equivalence(pp, tp, model, X_raw, device):
    # Max abs differences between src/preprocess.py and TorchPreprocessor on the same raw rows.
    Xr = X_raw.astype(np.float64)
    df = pd.DataFrame(Xr, columns=pp.features)
    z_np32 = pp.transform(df)
    Xl = np.where(pp.log_mask, np.log1p(np.maximum(Xr, 0.0)), Xr)
    z_np64 = np.clip((Xl - pp.mean) / pp.std, -pp.clip, pp.clip)
    with torch.no_grad():
        z_t64 = tp(torch.from_numpy(Xr).to(device)).cpu().numpy()
        lg_np = model(torch.from_numpy(z_np32).to(device)).cpu().numpy()
        lg_t = model(torch.from_numpy(z_t64.astype(np.float32)).to(device)).cpu().numpy()
    return {"n_rows": len(Xr),
            "max_abs_diff_float64": float(np.abs(z_t64 - z_np64).max()),
            "max_abs_diff_float32_output": float(np.abs(z_t64.astype(np.float32) - z_np32).max()),
            "max_abs_diff_logits": float(np.abs(lg_np - lg_t).max()),
            "argmax_agreement": float((lg_np.argmax(1) == lg_t.argmax(1)).mean())}


# ---------------------------------------------------------------- attack context

@dataclass
class AttackContext:
    r: torch.Tensor            # (B, F) clean raw rows, float64
    budget: torch.Tensor       # (B, F) per-feature raw budget (0 for non-controls)
    lo: torch.Tensor           # (B, F) delta box lower bound (in budget units)
    hi: torch.Tensor           # (B, F) delta box upper bound
    idx: dict
    attains: torch.Tensor      # (B,) clean pkt_len_min == clean fwd_pkt_len_min
    n_sub: torch.Tensor        # (B,) clean number of subflows
    old: dict                  # derived values at the clean row (delta formulation)

    def col(self, x, name):
        return x[:, self.idx[name]]


def n_subflows(fc, fsub_pkt):
    return np.where(fsub_pkt > 0, np.round(fc / np.where(fsub_pkt > 0, fsub_pkt, 1.0)), 1.0).clip(min=1.0)


def freeze_mask(R, feats, classes_of_rows, clean_fail):
    # (N, F) bool: controls frozen for each row. clean_fail: dict relation -> (N,) bool from check_validity on
    # the clean rows; every control involved in a relation the clean row already violates is frozen.
    i = {f: k for k, f in enumerate(feats)}
    fr = np.zeros(R.shape, dtype=bool)
    for n, cname in enumerate(classes_of_rows):
        for f in FAMILY_FREEZE.get(CLASS_FAMILY[cname], []):
            fr[n, i[f]] = True
    fr[R[:, i["fwd_tcp_init_win_bytes"]] < 0, i["fwd_tcp_init_win_bytes"]] = True
    fr[R[:, i["iat_min"]] < 0, i["iat_min"]] = True
    dur0 = R[:, i["flow_duration"]] <= 0
    for f in TIMING_CONTROLS:
        fr[dur0, i[f]] = True
    fc0 = R[:, i["fwd_pkt_cnt"]] <= 0
    for f in CONTROLS:
        fr[fc0, i[f]] = True
    for rel, bad in clean_fail.items():
        for f in RELATION_CONTROLS.get(rel, []):
            fr[bad, i[f]] = True
    return fr


def make_context(R, feats, floors, eps, frozen, device):
    # R: (N, F) clean raw (float64 numpy). Budgets and boxes for one eps.
    i = {f: k for k, f in enumerate(feats)}
    N, F = R.shape
    scale = np.zeros(F)
    lo, hi = np.zeros((N, F)), np.zeros((N, F))
    for f in CONTROLS:
        scale[i[f]] = floors[f]
        hi[:, i[f]] = 1.0
        lo[:, i[f]] = -1.0 if SPEC[f]["direction"] == "either" else 0.0
    budget = eps * np.where(R != 0, np.abs(R), scale[None, :])
    budget[:, [k for f, k in i.items() if SPEC[f]["role"] != "control"]] = 0.0
    lo[frozen], hi[frozen] = 0.0, 0.0
    t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    r = t(R)
    ctx = AttackContext(r=r, budget=t(budget), lo=t(lo), hi=t(hi), idx=i,
                        attains=torch.as_tensor(R[:, i["pkt_len_min"]] == R[:, i["fwd_pkt_len_min"]], device=device),
                        n_sub=t(n_subflows(R[:, i["fwd_pkt_cnt"]], R[:, i["fwd_subflow_pkt_mean"]])), old={})
    ctx.old = derived_values(r, ctx)
    return ctx


# ---------------------------------------------------------------- derived features (torch)

def _sdiv(a, b):
    # a / b where b > 0, else 0 (gradient-safe).
    ok = b > 0
    return torch.where(ok, a / torch.where(ok, b, torch.ones_like(b)), torch.zeros_like(a))


def derived_values(x, ctx):
    g = lambda n: ctx.col(x, n)
    fc, bc = g("fwd_pkt_cnt"), g("bwd_pkt_cnt")
    n = fc + bc
    ft, bt, dur = g("fwd_pkt_len_tot"), g("bwd_pkt_len_tot"), g("flow_duration")
    mf, mb = _sdiv(ft, fc), _sdiv(bt, bc)
    # Pooled sample variance (ddof 1) in the numerically stable two-group form.
    ss = ((fc - 1).clamp_min(0) * g("fwd_pkt_len_std") ** 2 + (bc - 1).clamp_min(0) * g("bwd_pkt_len_std") ** 2
          + _sdiv(fc * bc, n) * (mf - mb) ** 2)
    var = _sdiv(ss, n - 1)
    fmin = g("fwd_pkt_len_min")
    return {"fwd_pkt_len_mean": mf,
            "fwd_iat_mean": _sdiv(g("fwd_iat_tot"), fc - 1),
            "iat_mean": _sdiv(dur, n - 1),
            "pkt_len_max": torch.maximum(g("fwd_pkt_len_max"), g("bwd_pkt_len_max")),
            "pkt_len_min": torch.where(ctx.attains, fmin, torch.minimum(ctx.col(ctx.r, "pkt_len_min"), fmin)),
            "pkt_len_mean": _sdiv(ft + bt, n),
            "pkt_len_var": var,
            "pkt_len_std": torch.sqrt(var + 1e-12),
            "bytes_per_s": _sdiv((ft + bt) * US, dur),
            "pkt_per_s": _sdiv(n * US, dur),
            "fwd_pkt_per_s": _sdiv(fc * US, dur),
            "bwd_pkt_per_s": _sdiv(bc * US, dur),
            "down_up_ratio": _sdiv(bc, fc),
            "flag_psh": g("fwd_flag_psh") + g("bwd_flag_psh"),
            "fwd_subflow_bytes_mean": ft / ctx.n_sub,
            "fwd_subflow_pkt_mean": fc / ctx.n_sub}


def derive(x, ctx):
    # Derived columns = clean value + (f(new bases) - f(clean bases)): exact at the clean row, differentiable.
    new = derived_values(x, ctx)
    y = x.clone()
    for name, v in new.items():
        k = ctx.idx[name]
        y[:, k] = ctx.r[:, k] + (v - ctx.old[name])
    return y


# ---------------------------------------------------------------- projection (repair + backtracking)

def std_bounds(n, m, mu, M):
    # Feasible sample-std interval for n values in [m, M] with mean mu, min m and max M attained.
    # Upper: Bhatia-Davis, var_pop <= (M - mu)(mu - m). Lower: the other n - 2 values all equal.
    nn_ = n.clamp_min(2)
    hi2 = (nn_ / (nn_ - 1)) * ((M - mu) * (mu - m)).clamp_min(0)
    c = torch.where(n >= 3, (n * mu - m - M) / (n - 2).clamp_min(1), mu)
    lo2 = ((m - mu) ** 2 + (M - mu) ** 2 + (n - 2).clamp_min(0) * (c - mu) ** 2) / (nn_ - 1)
    lo, hi = torch.sqrt(lo2), torch.sqrt(hi2)
    small = n < 2
    return torch.where(small, torch.zeros_like(lo), torch.minimum(lo, hi)), torch.where(small, torch.zeros_like(hi), hi)


def _clamp_std(s, lo, hi, s_clean, rtol=1e-5):
    # Clamp into [lo, hi]. Only an UNCHANGED clean value that is outside by float32 rounding is kept as is
    # (so unchanged frozen rows are not moved); a perturbed value is always clamped (never negative).
    sc = torch.minimum(torch.maximum(s, lo), hi)
    keep = (s == s_clean) & ((sc - s).abs() <= rtol * (1 + s.abs()))
    return torch.where(keep, s, sc)


def _fix_family(n, tot, m, M, s, m_floor, M_floor, s_clean):
    # n values >= m_floor with total tot: enforce min <= mean <= max, the mean-feasibility interval and the
    # std bounds. Integer quantities stay integer (ceil / floor). Returns (m, M, s).
    one, none = n == 1, n <= 0
    M = torch.minimum(torch.maximum(M, M_floor), tot)
    m = torch.minimum(torch.maximum(m, m_floor), M)
    k = (n - 1).clamp_min(1)
    M = torch.minimum(torch.maximum(M, torch.ceil((tot - m) / k)), tot)
    m = torch.maximum(torch.minimum(m, torch.floor((tot - M) / k)), m_floor)
    lo, hi = std_bounds(n, m, tot / n.clamp_min(1), M)
    s = _clamp_std(s, lo, hi, s_clean)
    m = torch.where(one, tot, m)
    M = torch.where(one, tot, M)
    z = torch.zeros_like(s)
    s = torch.where(one | none, z, s)
    m = torch.where(none, torch.minimum(m, z), m)
    M = torch.where(none, z, M)
    return m, M, s


@torch.no_grad()
def repair(xp, ctx):
    # xp: (B, F) raw proposal (controls = clean + delta * budget, everything else clean). Returns a row whose
    # controls satisfy every structural relation; budget/box compliance is checked separately (backtracking).
    I, r = ctx.idx, ctx.r
    R = lambda n: r[:, I[n]]
    toward = lambda n: R(n) + torch.trunc(xp[:, I[n]] - R(n))      # integer change truncated toward clean
    x = r.clone()
    put = lambda n, v: x.__setitem__((slice(None), I[n]), v)

    fc0, bc = R("fwd_pkt_cnt"), R("bwd_pkt_cnt")
    fc = torch.maximum(toward("fwd_pkt_cnt"), fc0)
    dfc = fc - fc0
    put("fwd_pkt_cnt", fc)
    put("flag_ack", torch.minimum(torch.minimum(torch.maximum(toward("flag_ack"), R("flag_ack")), R("flag_ack") + dfc),
                                  fc + bc))
    # Forward payload-length family.
    M = torch.maximum(toward("fwd_pkt_len_max"), R("fwd_pkt_len_max"))
    m = toward("fwd_pkt_len_min")
    m = torch.where(dfc == 0, torch.maximum(m, R("fwd_pkt_len_min")), m)     # can only drop by adding packets
    m = torch.where(ctx.attains, torch.minimum(m, R("fwd_pkt_len_min")), m)  # flow min: backward min unknown
    m = torch.minimum(m.clamp_min(0), M)
    fne = torch.minimum(torch.maximum(toward("fwd_non_empty_pkt_cnt"), R("fwd_non_empty_pkt_cnt")), fc)
    fne = torch.where(m > 0, fc, torch.minimum(fne, (fc - 1).clamp_min(0)))
    # A non-empty packet carries >= 1 byte; with no non-empty packet every length is 0.
    M = torch.where(fne >= 1, M.clamp_min(1), torch.zeros_like(M))
    m = torch.minimum(m, M)
    tot = torch.maximum(toward("fwd_pkt_len_tot"), R("fwd_pkt_len_tot"))
    lo_t = torch.where(m > 0, (fc - 1) * m + M, M + (fne - 1).clamp_min(0))
    hi_t = torch.where(m > 0, m + (fc - 1) * M, fne * M)
    tot = torch.minimum(torch.maximum(tot, lo_t), hi_t)
    single = fc == 1
    v = torch.where(ctx.attains, R("fwd_pkt_len_tot"), torch.maximum(torch.maximum(tot, M), m))
    tot, M, m = (torch.where(single, v, a) for a in (tot, M, m))
    fne = torch.where(single, (v > 0).double(), fne)
    lo_s, hi_s = std_bounds(fc, m, tot / fc.clamp_min(1), M)
    s = _clamp_std(xp[:, I["fwd_pkt_len_std"]], lo_s, hi_s, R("fwd_pkt_len_std"))
    for n_, v_ in (("fwd_pkt_len_max", M), ("fwd_pkt_len_min", m), ("fwd_non_empty_pkt_cnt", fne),
                   ("fwd_pkt_len_tot", tot), ("fwd_pkt_len_std", torch.where(fc <= 1, torch.zeros_like(s), s))):
        put(n_, v_)
    # Headers: each added packet carries >= the clean header minimum, at most 60 bytes per packet.
    hmin = R("fwd_pkt_hdr_len_min")
    h = torch.maximum(toward("fwd_pkt_hdr_len_tot"), R("fwd_pkt_hdr_len_tot"))
    h = torch.minimum(torch.maximum(h, torch.maximum(R("fwd_pkt_hdr_len_tot") + dfc * hmin, hmin * fc)), HDR_MAX * fc)
    put("fwd_pkt_hdr_len_tot", h)
    put("fwd_flag_psh", torch.minimum(torch.maximum(toward("fwd_flag_psh"), R("fwd_flag_psh")), fne))
    w = R("fwd_tcp_init_win_bytes")
    put("fwd_tcp_init_win_bytes", torch.where(w < 0, w, toward("fwd_tcp_init_win_bytes").clamp(0, WIN_MAX)))
    # Timing. Without backward packets the flow IAT family IS the forward IAT family.
    nob = bc == 0
    dur = torch.maximum(toward("flow_duration"), R("flow_duration"))
    fit = torch.maximum(toward("fwd_iat_tot"), R("fwd_iat_tot"))
    tie = torch.maximum(dur, fit)
    dur = torch.where(nob, tie, dur)
    fit = torch.where(nob, tie, torch.minimum(fit, dur))
    gf = fc - 1
    fit = torch.where(gf <= 0, torch.zeros_like(fit), fit)
    idle_max = R("idle_max")
    zero = torch.zeros_like(fit)
    fm, fM, fs = _fix_family(gf, fit, toward("fwd_iat_min"), toward("fwd_iat_max"), xp[:, I["fwd_iat_std"]],
                             zero, torch.where(nob, idle_max, zero), R("fwd_iat_std"))
    im_floor = torch.minimum(R("iat_min"), zero)
    im, iM, is_ = _fix_family(fc + bc - 1, dur, toward("iat_min"), toward("iat_max"), xp[:, I["iat_std"]],
                              im_floor, idle_max, R("iat_std"))
    im = torch.where(R("iat_min") < 0, R("iat_min"), im)
    im, iM, is_ = (torch.where(nob, a, b) for a, b in ((fm, im), (fM, iM), (fs, is_)))
    for n_, v_ in (("flow_duration", dur), ("fwd_iat_tot", fit), ("fwd_iat_min", fm), ("fwd_iat_max", fM),
                   ("fwd_iat_std", fs), ("iat_min", im), ("iat_max", iM), ("iat_std", is_)):
        put(n_, v_)
    return derive(x, ctx)


@torch.no_grad()
def in_box(x, ctx, tol=1e-9):
    # Every control within its direction box and budget (rows, bool). Derived features are exempt (they follow
    # their bases); immutable features are never written by repair().
    ctrl = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    ctrl[[ctx.idx[f] for f in CONTROLS]] = True
    d = (x - ctx.r)[:, ctrl]
    t = tol * (1 + ctx.r[:, ctrl].abs())
    return ((d >= ctx.lo[:, ctrl] * ctx.budget[:, ctrl] - t) & (d <= ctx.hi[:, ctrl] * ctx.budget[:, ctrl] + t)).all(1)


@torch.no_grad()
def project(delta, ctx, max_halvings=12):
    # Repair the proposal; rows that leave the box after repair are pulled back toward the clean row by
    # halving delta, and fall back to the clean row (always valid) if that fails. Returns (x, scale).
    delta = torch.minimum(torch.maximum(delta, ctx.lo), ctx.hi)
    scale = torch.ones(delta.shape[0], dtype=delta.dtype, device=delta.device)
    x = repair(ctx.r + delta * ctx.budget, ctx)
    ok = in_box(x, ctx)
    for _ in range(max_halvings):
        if bool(ok.all()):
            break
        scale = torch.where(ok, scale, scale * 0.5)
        xn = repair(ctx.r + scale[:, None] * delta * ctx.budget, ctx)
        x = torch.where(ok[:, None], x, xn)
        ok = in_box(x, ctx)
    x = torch.where(ok[:, None], x, ctx.r)
    scale = torch.where(ok, scale, torch.zeros_like(scale))
    return x, scale


# ---------------------------------------------------------------- independent validity checker (numpy)

# Controls involved in each relation (a relation the clean row already violates freezes these controls).
RELATION_CONTROLS = {
    "fwd_len_family": ["fwd_pkt_cnt", "fwd_pkt_len_tot", "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_std",
                       "fwd_non_empty_pkt_cnt"],
    "fwd_len_total": ["fwd_pkt_cnt", "fwd_pkt_len_tot"],
    "fwd_non_empty": ["fwd_pkt_cnt", "fwd_non_empty_pkt_cnt", "fwd_pkt_len_min", "fwd_pkt_len_tot",
                      "fwd_pkt_len_max"],
    "fwd_psh": ["fwd_flag_psh", "fwd_non_empty_pkt_cnt"],
    "flag_psh_sum": ["fwd_flag_psh"],
    "ack": ["flag_ack", "fwd_pkt_cnt"],
    "header": ["fwd_pkt_hdr_len_tot", "fwd_pkt_cnt"],
    "pkt_len_family": ["fwd_pkt_cnt", "fwd_pkt_len_tot", "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_std"],
    "fwd_iat_family": ["fwd_pkt_cnt", "fwd_iat_tot", "fwd_iat_max", "fwd_iat_min", "fwd_iat_std", "flow_duration"],
    "iat_family": ["fwd_pkt_cnt", "flow_duration", "iat_max", "iat_min", "iat_std"],
    "no_bwd_timing": TIMING_CONTROLS,
    "rates": ["fwd_pkt_cnt", "fwd_pkt_len_tot", "flow_duration"],
    "down_up": ["fwd_pkt_cnt"],
    "subflow": ["fwd_pkt_cnt", "fwd_pkt_len_tot"],
    "window": ["fwd_tcp_init_win_bytes"],
    "fixed_timing_bounds": ["flow_duration"],
}


def _close(a, b, rtol=1e-4, atol=1e-3):
    return np.abs(a - b) <= rtol * np.maximum(np.abs(a), np.abs(b)) + atol


def _le(a, b, rtol=1e-4, atol=1e-3):
    return a <= b + rtol * np.maximum(np.abs(a), np.abs(b)) + atol


def _family_ok(n, tot, mn, mean, mx, sd):
    # Summary statistics of n non-negative values with total tot are jointly feasible (sample std).
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(n > 0, tot / np.maximum(n, 1), 0.0)
        ok = _close(mean, mu) & _le(mn, mean) & _le(mean, mx) & (sd >= -1e-9)
        ok &= np.where(n >= 2, _le((n - 1) * mn + mx, tot) & _le(tot, mn + (n - 1) * mx), True)
        ok &= np.where(n == 1, _close(mn, mx) & _close(mx, tot) & _close(sd, 0.0), True)
        ok &= np.where(n <= 0, _close(mx, 0.0) & _close(sd, 0.0) & _close(tot, 0.0), True)
        # Std bounds, computed independently from the population moments.
        var_hi = np.where(n >= 2, (mx - mu) * (mu - mn) * n / np.maximum(n - 1, 1), 0.0)
        rest = np.where(n >= 3, (tot - mn - mx) / np.maximum(n - 2, 1), mu)
        ssd = (mn - mu) ** 2 + (mx - mu) ** 2 + np.maximum(n - 2, 0) * (rest - mu) ** 2
        var_lo = np.where(n >= 2, ssd / np.maximum(n - 1, 1), 0.0)
        ok &= np.where(n >= 2, _le(np.sqrt(np.maximum(var_lo, 0)), sd, rtol=1e-3, atol=1e-2)
                       & _le(sd, np.sqrt(np.maximum(var_hi, 0)), rtol=1e-3, atol=1e-2), True)
    return ok


def relations(X, R, feats):
    # Per-row booleans (True = relation holds) for every structural relation. Written independently of the
    # torch projection: raw-moment pooled variance, explicit rate formulas, n_sub from the clean row.
    i = {f: k for k, f in enumerate(feats)}
    g = lambda n: X[:, i[n]]
    c = lambda n: R[:, i[n]]
    fc, bc = g("fwd_pkt_cnt"), g("bwd_pkt_cnt")
    n = fc + bc
    ft, bt, dur = g("fwd_pkt_len_tot"), g("bwd_pkt_len_tot"), g("flow_duration")
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["fwd_len_family"] = _family_ok(fc, ft, g("fwd_pkt_len_min"), g("fwd_pkt_len_mean"), g("fwd_pkt_len_max"),
                                           g("fwd_pkt_len_std"))
        out["fwd_len_total"] = _close(ft, fc * g("fwd_pkt_len_mean"), atol=1e-2)
        fne, fmin = g("fwd_non_empty_pkt_cnt"), g("fwd_pkt_len_min")
        out["fwd_non_empty"] = (_le(fne, fc) & np.where(fmin > 0, fne == fc, True)
                                & np.where((fmin == 0) & (fc > 0), fne < fc, True) & ((ft > 0) == (fne > 0))
                                & _le(ft, fne * g("fwd_pkt_len_max")) & _le(fne, ft))
        out["fwd_psh"] = _le(g("fwd_flag_psh"), fne)
        out["flag_psh_sum"] = _close(g("flag_psh"), g("fwd_flag_psh") + g("bwd_flag_psh"))
        out["ack"] = _le(g("flag_ack"), n) & _le(g("flag_ack") - c("flag_ack"), fc - c("fwd_pkt_cnt"))
        h, hmin = g("fwd_pkt_hdr_len_tot"), g("fwd_pkt_hdr_len_min")
        out["header"] = (_le(hmin * fc, h) & _le(h, HDR_MAX * fc)
                         & _le(hmin * (fc - c("fwd_pkt_cnt")), h - c("fwd_pkt_hdr_len_tot")))
        # Flow-level packet length family from the directional families (raw moments, ddof 1).
        s2 = lambda cnt, tot_, sd: np.where(cnt > 0, (cnt - 1) * sd ** 2 + tot_ ** 2 / np.maximum(cnt, 1), 0.0)
        sum_sq = s2(fc, ft, g("fwd_pkt_len_std")) + s2(bc, bt, g("bwd_pkt_len_std"))
        var = np.where(n > 1, (sum_sq - (ft + bt) ** 2 / np.maximum(n, 1)) / np.maximum(n - 1, 1), 0.0)
        pmin_ref = np.where(c("pkt_len_min") == c("fwd_pkt_len_min"), fmin, np.minimum(c("pkt_len_min"), fmin))
        out["pkt_len_family"] = (_close(g("pkt_len_max"), np.maximum(g("fwd_pkt_len_max"), g("bwd_pkt_len_max")))
                                 & _close(g("pkt_len_min"), pmin_ref)
                                 & _close(g("pkt_len_mean"), np.where(n > 0, (ft + bt) / np.maximum(n, 1), 0.0))
                                 & _close(g("pkt_len_var"), var, rtol=1e-3, atol=1.0)
                                 & _close(g("pkt_len_var"), g("pkt_len_std") ** 2, rtol=1e-3, atol=1e-2)
                                 & _le(g("pkt_len_min"), g("pkt_len_mean")) & _le(g("pkt_len_mean"), g("pkt_len_max")))
        fit = g("fwd_iat_tot")
        out["fwd_iat_family"] = (_family_ok(fc - 1, fit, g("fwd_iat_min"), g("fwd_iat_mean"), g("fwd_iat_max"),
                                            g("fwd_iat_std")) & _le(fit, dur) & _le(g("fwd_iat_max"), fit))
        out["iat_family"] = (_family_ok(n - 1, dur, g("iat_min"), g("iat_mean"), g("iat_max"), g("iat_std"))
                             & _le(g("iat_max"), dur) & _le(g("idle_max"), g("iat_max")))
        nob = bc == 0
        out["no_bwd_timing"] = np.where(nob, _close(dur, fit) & _close(g("iat_max"), g("fwd_iat_max"))
                                        & _close(g("iat_min"), g("fwd_iat_min")) & _close(g("iat_std"), g("fwd_iat_std")),
                                        True)
        rate = lambda num: np.where(dur > 0, num / np.where(dur > 0, dur, 1.0) * US, 0.0)
        out["rates"] = (_close(g("bytes_per_s"), rate(ft + bt)) & _close(g("pkt_per_s"), rate(n))
                        & _close(g("fwd_pkt_per_s"), rate(fc)) & _close(g("bwd_pkt_per_s"), rate(bc)))
        out["down_up"] = _close(g("down_up_ratio"), np.where(fc > 0, bc / np.maximum(fc, 1), 0.0))
        ns = n_subflows(c("fwd_pkt_cnt"), c("fwd_subflow_pkt_mean"))
        out["subflow"] = (_close(g("fwd_subflow_pkt_mean"), fc / ns) & _close(g("fwd_subflow_bytes_mean"), ft / ns)
                          & _close(g("bwd_subflow_pkt_mean"), bc / ns) & _close(g("bwd_subflow_bytes_mean"), bt / ns))
        w = g("fwd_tcp_init_win_bytes")
        out["window"] = np.where(c("fwd_tcp_init_win_bytes") < 0, w == c("fwd_tcp_init_win_bytes"),
                                 (w >= 0) & (w <= WIN_MAX))
        out["fixed_timing_bounds"] = (_le(g("active_max"), dur) & _le(g("idle_max"), dur) & _le(g("bwd_iat_tot"), dur))
    return out


def check_validity(X, R, feats, spec_df, eps, row_classes):
    # Returns (valid (N,) bool, per-check violation dict, per-relation clean-fail dict). Relations are only
    # enforced on rows whose clean values satisfy them (extractor quirks are reported, not held against x).
    sp = spec_df.set_index("feature").loc[feats]
    role, direction = sp["role"].to_numpy(), sp["direction"].to_numpy()
    integer = (sp["integer"] == "yes").to_numpy()
    floor = pd.to_numeric(sp["zero_floor_scale"], errors="coerce").fillna(0).to_numpy()
    frozen_fams = [set(s.split(";")) if s else set() for s in sp["frozen_for_class_families"].fillna("").astype(str)]
    N = X.shape[0]
    viol = {}
    D = X - R
    # Immutable features and per-row frozen controls must be bit-identical to the clean row.
    imm = role == "immutable"
    viol["immutable_changed"] = (D[:, imm] != 0).any(1)
    fam_rows = np.array([CLASS_FAMILY[cn] for cn in row_classes])
    frz = np.zeros(X.shape, dtype=bool)
    for j, fams in enumerate(frozen_fams):
        if fams:
            frz[:, j] = np.isin(fam_rows, list(fams))
    viol["class_family_frozen_changed"] = (frz & (D != 0)).any(1)
    addonly = (direction == "add-only") & (role == "control")
    viol["add_only_decreased"] = (D[:, addonly] < -1e-9 * (1 + np.abs(R[:, addonly]))).any(1)
    isint_clean = np.isclose(R, np.round(R), atol=1e-6)
    viol["non_integer"] = (integer[None, :] & isint_clean & ~np.isclose(X, np.round(X), atol=1e-6)).any(1)
    viol["negative"] = (X < np.minimum(R, 0.0) - 1e-9).any(1)
    ctrl = role == "control"
    bud = eps * np.where(R != 0, np.abs(R), floor[None, :])
    viol["over_budget"] = (ctrl[None, :] & (np.abs(D) > bud * (1 + 1e-6) + 1e-9)).any(1)
    viol["not_finite"] = ~np.isfinite(X).all(1)
    rel_x, rel_r = relations(X, R, feats), relations(R, R, feats)
    clean_fail = {}
    for k in rel_x:
        clean_fail[k] = ~rel_r[k]
        viol[f"rel:{k}"] = ~rel_x[k] & rel_r[k]
    bad = np.zeros(N, dtype=bool)
    for v in viol.values():
        bad |= v
    return ~bad, viol, clean_fail


# ---------------------------------------------------------------- main: spec CSV + checks

def main():
    import time
    from src.stage3_explain import load_ftt
    s1, s2 = resolve("results/stage1"), resolve("results/stage2")
    feats = json.loads((s1 / "features.json").read_text(encoding="utf-8"))
    t0 = time.time()
    train = pd.read_parquet(s1 / "train.parquet", columns=feats)
    floors = zero_floor_scales(train, feats)
    del train
    spec_df = write_spec_csv(feats, floors)
    print(spec_df.groupby(["role", "direction"]).size().to_string())
    dev = torch.device("cuda")
    pp = Preprocessor.load(s2 / "preprocess.json")
    tp = TorchPreprocessor(pp, dev)
    test = pd.read_parquet(s1 / "test.parquet", columns=feats + ["label_id"])
    rng = np.random.default_rng(2026)
    Xr = test[feats].to_numpy(np.float64)[rng.choice(len(test), 10000, replace=False)]
    model, _ = load_ftt(s2 / "ftt_seed0.pt", dev)
    print(preprocessing_equivalence(pp, tp, model, Xr, dev))
    valid, viol, cf = check_validity(Xr, Xr, feats, spec_df, 0.0, ["benign"] * len(Xr))
    print("clean self-check valid:", valid.mean(), {k: int(v.sum()) for k, v in cf.items() if v.sum()})
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
