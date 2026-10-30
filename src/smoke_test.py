# Smoke test for the Stage 3 toolchain on this Python version:
#   - imports torch, captum, shap, xgboost and prints versions
#   - FT-Transformer attention extraction in fp32 (rows sum to 1, shapes per layer/head)
#   - double backprop through the explicit attention (grad of an input-gradient w.r.t. parameters)
#   - captum IntegratedGradients and GradientShap on the FT-Transformer
#   - a tiny shap explainer run, and xgboost GPU training
# Usage: .venv\Scripts\python.exe -m src.smoke_test
import importlib
import sys
import time
import traceback

import numpy as np

results = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        results.append((name, "OK", detail, time.time() - t0))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}", time.time() - t0))
        traceback.print_exc()


def imports():
    out = []
    for m in ("torch", "captum", "shap", "xgboost", "numpy", "pandas", "sklearn", "scipy"):
        mod = importlib.import_module(m)
        out.append(f"{m}={getattr(mod, '__version__', '?')}")
    return ", ".join(out)


def cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    free, total = torch.cuda.mem_get_info()
    return f"{torch.cuda.get_device_name(0)}, free {free / 2**30:.1f}/{total / 2**30:.1f} GB, bf16={torch.cuda.is_bf16_supported()}"


def make_model():
    import torch
    from src.models.ft_transformer import FTTransformer
    torch.manual_seed(0)
    return FTTransformer(n_features=72, n_classes=11, d_model=96, n_blocks=3, n_heads=4).cuda().eval()


def attention():
    import torch
    m = make_model()
    x = torch.randn(8, 72, device="cuda")
    with torch.no_grad():
        logits, attns = m(x, return_attn=True)
    assert len(attns) == 3 and attns[0].shape == (8, 4, 73, 73), [a.shape for a in attns]
    assert all(a.dtype == torch.float32 for a in attns)
    err = max(float((a.sum(-1) - 1).abs().max()) for a in attns)
    assert err < 1e-5, err
    return f"{len(attns)} layers x {tuple(attns[0].shape)} fp32, max |row sum - 1| = {err:.1e}"


def double_backprop():
    import torch
    import torch.nn.functional as F
    m = make_model()
    x = torch.randn(16, 72, device="cuda", requires_grad=True)
    y = torch.randint(0, 11, (16,), device="cuda")
    loss = F.cross_entropy(m(x), y)
    (gx,) = torch.autograd.grad(loss, x, create_graph=True)
    penalty = gx.pow(2).sum()
    grads = torch.autograd.grad(penalty, [p for p in m.parameters()], allow_unused=True)
    n_nonzero = sum(1 for g in grads if g is not None and g.abs().sum() > 0)
    assert n_nonzero > 0
    return f"second-order grads non-zero for {n_nonzero}/{len(grads)} parameter tensors"


def captum_methods():
    import torch
    from captum.attr import GradientShap, IntegratedGradients
    m = make_model()
    x = torch.randn(4, 72, device="cuda")
    target = m(x).argmax(1)
    ig = IntegratedGradients(m)
    a_ig, delta = ig.attribute(x, baselines=torch.zeros_like(x), target=target, n_steps=32,
                               return_convergence_delta=True)
    gs = GradientShap(m)
    a_gs = gs.attribute(x, baselines=torch.randn(20, 72, device="cuda"), target=target, n_samples=16)
    assert a_ig.shape == x.shape and a_gs.shape == x.shape
    return f"IG shape {tuple(a_ig.shape)}, max |convergence delta| {float(delta.abs().max()):.2e}; GradientShap OK"


def shap_check():
    import shap
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    ex = shap.Explainer(clf.predict_proba, X[:50])
    sv = ex(X[:5])
    return f"shap.Explainer values shape {sv.values.shape}"


def xgb_gpu():
    # XGBoost reports device fallbacks as Python warnings, and a booster that ends up on CPU
    # trains silently, so fail on any XGBoost warning AND check the booster's effective device.
    import json
    import warnings
    import xgboost as xgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bst = xgb.train({"tree_method": "hist", "device": "cuda", "objective": "binary:logistic"},
                        xgb.DMatrix(X, label=y), 10)
        contrib = bst.predict(xgb.DMatrix(X[:5]), pred_contribs=True)
    xgb_warnings = [str(w.message) for w in caught if "xgboost" in w.filename.lower() or "WARNING" in str(w.message)]
    assert not xgb_warnings, f"XGBoost warnings (possible CPU fallback): {xgb_warnings}"
    device = json.loads(bst.save_config())["learner"]["generic_param"]["device"]
    assert device.startswith("cuda"), f"booster device is {device!r}, expected cuda"
    return f"GPU hist OK (booster device {device}, no warnings), TreeSHAP contribs shape {contrib.shape}"


def main():
    print(f"Python {sys.version}")
    for name, fn in [("imports", imports), ("cuda", cuda), ("attention_fp32", attention),
                     ("double_backprop", double_backprop), ("captum_ig_gradshap", captum_methods),
                     ("shap", shap_check), ("xgboost_gpu", xgb_gpu)]:
        check(name, fn)
    print("\nSMOKE TEST RESULTS")
    for name, status, detail, dt in results:
        print(f"  [{status}] {name:20s} ({dt:.1f}s) {detail}")
    sys.exit(0 if all(r[1] == "OK" for r in results) else 1)


if __name__ == "__main__":
    main()
