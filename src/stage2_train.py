# Stage 2: preprocessing, FT-Transformer (several seeds), XGBoost baseline and a day-confound diagnostic.
# Model selection uses validation macro-F1 only; the test split is only evaluated, never tuned on.
#
# Usage: .venv\Scripts\python.exe -m src.stage2_train [--set stage2.models='["xgb"]'] [--set stage2.ftt.batch_size=2048]
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb

from src.config import load_config
from src.metrics import by_day_report, confusion_long, macro_f1, per_class_report
from src.models.ft_transformer import FTTransformer
from src.preprocess import Preprocessor
from src.utils import RESULTS, finish_run, resolve, set_seed, start_run


def load_data(cfg, log):
    s1_dir = resolve(cfg["paths"]["stage1_dir"])
    feats = json.loads((s1_dir / "features.json").read_text(encoding="utf-8"))
    label_map = json.loads((s1_dir / "label_map.json").read_text(encoding="utf-8"))
    classes = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    data = {}
    for split in ("train", "val", "test"):
        df = pd.read_parquet(s1_dir / f"{split}.parquet", columns=feats + ["label_id", "day"])
        data[split] = df
        log.info(f"Loaded {split}: {len(df)} rows")
    return feats, classes, data


def make_class_weights(y, n_classes, mode):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    if mode == "none":
        w = np.ones(n_classes)
    elif mode == "sqrt_inv":
        w = np.sqrt(counts.sum() / counts)
    elif mode == "inv":
        w = counts.sum() / counts
    else:
        raise ValueError(mode)
    # Normalise so the mean weight over training samples is 1.
    w = w / ((w * counts).sum() / counts.sum())
    return w, counts


def evaluate_predictions(model_name, seed, split, y, pred, days, classes):
    rep = per_class_report(y, pred, classes)
    rep.insert(0, "split", split)
    rep.insert(0, "seed", seed)
    rep.insert(0, "model", model_name)
    cm, _ = confusion_long(y, pred, classes)
    cm.insert(0, "split", split)
    cm.insert(0, "seed", seed)
    cm.insert(0, "model", model_name)
    bd = None
    if split == "test":
        bd = by_day_report(y, pred, days, classes)
        bd.insert(0, "seed", seed)
        bd.insert(0, "model", model_name)
    return rep, cm, bd


# ---------------------------------------------------------------- FT-Transformer

@torch.no_grad()
def predict_torch(model, X, batch_size):
    # fp32 inference (no autocast).
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        out.append(model(X[i:i + batch_size]).float())
    return torch.cat(out)


def train_ftt(seed, Xtr, ytr, Xva, yva, n_classes, class_w, sample_w, cfg, log, ckpt_path, extra_meta):
    c = cfg["stage2"]["ftt"]
    dev = torch.device("cuda")
    set_seed(seed)
    model = FTTransformer(Xtr.shape[1], n_classes, d_model=c["d_model"], n_blocks=c["n_blocks"],
                          n_heads=c["n_heads"], ffn_factor=c["ffn_factor"], attn_dropout=c["attn_dropout"],
                          ffn_dropout=c["ffn_dropout"], residual_dropout=c["residual_dropout"]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    bs = c["batch_size"]
    n = Xtr.shape[0]
    steps_per_epoch = math.ceil(n / bs)
    total_steps = steps_per_epoch * c["max_epochs"]

    def lr_lambda(step):
        # Linear warm-up then cosine decay over the maximum number of epochs.
        if step < c["warmup_steps"]:
            return (step + 1) / c["warmup_steps"]
        prog = (step - c["warmup_steps"]) / max(1, total_steps - c["warmup_steps"])
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    w_t = torch.tensor(class_w, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    best = {"f1": -1.0, "epoch": -1}
    curves = []
    bad_epochs = 0
    nonfinite = 0
    yva_np = yva.cpu().numpy()
    log.info(f"[ftt seed={seed}] params={n_params:,} steps/epoch={steps_per_epoch} amp={c['amp']}")
    for epoch in range(1, c["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        if sample_w is None:
            order = torch.randperm(n, device=dev, generator=gen)
        else:
            order = torch.multinomial(sample_w, n, replacement=True, generator=gen)
        loss_sum = torch.zeros((), device=dev)
        for i in range(steps_per_epoch):
            idx = order[i * bs:(i + 1) * bs]
            xb, yb = Xtr[idx], ytr[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=c["amp"]):
                logits = model(xb)
            loss = F.cross_entropy(logits.float(), yb, weight=w_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
            opt.step()
            sched.step()
            loss_sum += loss.detach()
            # Periodic NaN/inf check on the running loss sum (avoids a GPU sync every step).
            if (i + 1) % c["finite_check_every"] == 0 and not torch.isfinite(loss_sum):
                nonfinite += 1
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {i + 1}; "
                                   "try stage2.ftt.amp=false")
        train_loss = float(loss_sum) / steps_per_epoch
        t_train = time.time() - t0
        pred_va = predict_torch(model, Xva, c["eval_batch_size"]).argmax(1).cpu().numpy()
        f1_va = macro_f1(yva_np, pred_va, n_classes)
        curves.append({"seed": seed, "epoch": epoch, "train_loss": train_loss, "val_macro_f1": f1_va,
                       "lr": sched.get_last_lr()[0], "epoch_time_s": round(t_train, 1)})
        improved = f1_va > best["f1"]
        log.info(f"[ftt seed={seed}] epoch {epoch:2d} loss={train_loss:.4f} val_macroF1={f1_va:.4f} "
                 f"({t_train:.1f}s){' *' if improved else ''}")
        if improved:
            best = {"f1": f1_va, "epoch": epoch}
            bad_epochs = 0
            torch.save({"state_dict": model.state_dict(), "model_kwargs": model.kwargs, "seed": seed,
                        "best_epoch": epoch, "val_macro_f1": f1_va, **extra_meta}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= c["patience"]:
                log.info(f"[ftt seed={seed}] early stop at epoch {epoch} (best epoch {best['epoch']})")
                break
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    info = {"seed": seed, "best_epoch": best["epoch"], "val_macro_f1": best["f1"], "epochs_run": epoch,
            "n_params": n_params, "peak_gpu_mem_gb": round(peak_gb, 3), "nonfinite_steps": nonfinite}
    log.info(f"[ftt seed={seed}] best epoch {best['epoch']} val macro-F1 {best['f1']:.4f}, peak GPU mem {peak_gb:.2f} GB")
    return model, curves, info


# ---------------------------------------------------------------- XGBoost

def train_xgb(Xtr, ytr, wtr, Xva, yva, n_classes, cfg, log):
    c = cfg["stage2"]["xgb"]
    params = {"objective": "multi:softprob", "num_class": n_classes, "tree_method": "hist",
              "device": c["device"], "max_depth": c["max_depth"], "eta": c["eta"],
              "subsample": c["subsample"], "colsample_bytree": c["colsample_bytree"],
              "min_child_weight": c["min_child_weight"], "seed": c["seed"],
              "disable_default_eval_metric": 1}
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dva = xgb.DMatrix(Xva, label=yva)

    def f1_metric(predt, dmat):
        y = dmat.get_label().astype(int)
        p = predt.reshape(len(y), -1).argmax(1)
        return "macro_f1", macro_f1(y, p, n_classes)

    hist = {}
    t0 = time.time()
    bst = xgb.train(params, dtr, num_boost_round=c["num_boost_round"], evals=[(dva, "val")],
                    custom_metric=f1_metric, maximize=True, early_stopping_rounds=c["early_stopping_rounds"],
                    evals_result=hist, verbose_eval=50)
    info = {"best_iteration": int(bst.best_iteration), "val_macro_f1": float(bst.best_score),
            "rounds_run": len(hist["val"]["macro_f1"]), "train_time_s": round(time.time() - t0, 1)}
    log.info(f"[xgb] best iteration {info['best_iteration']} val macro-F1 {info['val_macro_f1']:.4f} "
             f"({info['train_time_s']} s)")
    curve = pd.DataFrame({"round": np.arange(1, len(hist["val"]["macro_f1"]) + 1),
                          "val_macro_f1": hist["val"]["macro_f1"]})
    return bst, info, curve


def predict_xgb(bst, X):
    p = bst.predict(xgb.DMatrix(X), iteration_range=(0, bst.best_iteration + 1))
    return p.argmax(1)


def day_confound(data, Xs, cfg, log):
    # Can the DAY be predicted from benign flows alone? Train on benign train rows, evaluate on benign test rows.
    c = cfg["stage2"]["day_confound"]
    days = cfg["stage1"]["days"]
    d2i = {d: i for i, d in enumerate(days)}
    mtr = data["train"]["label_id"].to_numpy() == 0
    mte = data["test"]["label_id"].to_numpy() == 0
    ytr = data["train"]["day"].map(d2i).to_numpy()[mtr]
    yte = data["test"]["day"].map(d2i).to_numpy()[mte]
    params = {"objective": "multi:softprob", "num_class": len(days), "tree_method": "hist",
              "device": c["device"], "max_depth": c["max_depth"], "eta": c["eta"], "seed": c["seed"]}
    t0 = time.time()
    bst = xgb.train(params, xgb.DMatrix(Xs["train"][mtr], label=ytr), num_boost_round=c["num_boost_round"])
    pred = bst.predict(xgb.DMatrix(Xs["test"][mte])).argmax(1)
    acc = float((pred == yte).mean())
    recalls = {days[i]: float((pred[yte == i] == i).mean()) for i in range(len(days))}
    bal_acc = float(np.mean(list(recalls.values())))
    prior = np.bincount(yte, minlength=len(days)) / len(yte)
    majority = float(prior.max())
    rows = [{"metric": "accuracy", "value": acc}, {"metric": "balanced_accuracy", "value": bal_acc},
            {"metric": "majority_class_accuracy", "value": majority},
            {"metric": "chance_balanced_accuracy", "value": 1 / len(days)},
            {"metric": "n_train_benign", "value": int(mtr.sum())}, {"metric": "n_test_benign", "value": int(mte.sum())}]
    rows += [{"metric": f"recall_{d}", "value": v} for d, v in recalls.items()]
    cm = pd.crosstab(pd.Series([days[i] for i in yte], name="true_day"),
                     pd.Series([days[i] for i in pred], name="pred_day")).reindex(index=days, columns=days, fill_value=0)
    imp = bst.get_score(importance_type="gain")
    log.info(f"[day_confound] benign test accuracy {acc:.4f}, balanced {bal_acc:.4f} "
             f"(majority {majority:.4f}, chance 0.2) in {time.time() - t0:.1f}s")
    return pd.DataFrame(rows), cm, imp


# ---------------------------------------------------------------- main

def main():
    cfg, overrides = load_config("Stage 2: train FT-Transformer and XGBoost; report per-class F1.")
    t_start = time.time()
    run_dir, log = start_run("stage2_train", cfg, overrides)
    s2 = cfg["stage2"]
    out_dir = resolve(cfg["paths"]["stage2_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    feats, classes, data = load_data(cfg, log)
    K = len(classes)

    # Preprocessing (fit on train only).
    pp = Preprocessor.fit(data["train"], feats, s2["preprocess"]["log1p_skew_threshold"], s2["preprocess"]["clip"])
    pp.save(out_dir / "preprocess.json")
    Xs, pp_stats = {}, {}
    for split in ("train", "val", "test"):
        st = {}
        Xs[split] = pp.transform(data[split], st)
        pp_stats[split] = st
        log.info(f"Preprocess {split}: clipped {st['n_clipped']} of {st['n_values']} values, "
                 f"negatives into log1p {st['n_negative_into_log1p']}")
    pd.DataFrame({"feature": feats, "log1p": pp.log_mask, "skew_train": pp.skew_values,
                  "mean_after_log": pp.mean, "std_after_log": pp.std,
                  **{f"clipped_{s}": [pp_stats[s]["clipped_per_feature"].get(f, 0) for f in feats]
                     for s in ("train", "val", "test")}}).to_csv(RESULTS / "stage2_preprocessing.csv", index=False)
    summary["preprocess"] = {"n_log1p": int(pp.log_mask.sum()), "clip": pp.clip,
                             **{s: {k: v for k, v in pp_stats[s].items() if k != "clipped_per_feature"}
                                for s in pp_stats}}
    y = {s: data[s]["label_id"].to_numpy().astype(np.int64) for s in data}
    days = {s: data[s]["day"].to_numpy() for s in data}

    class_w, counts = make_class_weights(y["train"], K, s2["class_weight"])
    cw_df = pd.DataFrame({"class": classes, "train_count": counts.astype(int), "weight": class_w})
    cw_df.to_csv(RESULTS / "stage2_class_weights.csv", index=False)
    (out_dir / "class_weights.json").write_text(json.dumps(dict(zip(classes, class_w.tolist())), indent=2))
    log.info(f"Class weights ({s2['class_weight']}), sampler={s2['sampler']}:\n{cw_df.to_string(index=False)}")

    reports, confs, bydays = [], [], []
    meta = {"features": feats, "classes": classes, "preprocess_path": "results/stage2/preprocess.json",
            "config": cfg}

    if "ftt" in s2["models"]:
        dev = torch.device("cuda")
        Xt = {s: torch.from_numpy(Xs[s]).to(dev) for s in Xs}
        yt = {s: torch.from_numpy(y[s]).to(dev) for s in y}
        sample_w = None
        if s2["sampler"] == "balanced":
            sample_w = torch.tensor(class_w[y["train"]], dtype=torch.float32, device=dev)
        all_curves, ftt_infos = [], []
        for seed in s2["ftt"]["seeds"]:
            ckpt = out_dir / f"ftt_seed{seed}.pt"
            model, curves, info = train_ftt(seed, Xt["train"], yt["train"], Xt["val"], yt["val"], K, class_w,
                                            sample_w, cfg, log, ckpt, meta)
            all_curves += curves
            for split in ("val", "test"):
                pred = predict_torch(model, Xt[split], s2["ftt"]["eval_batch_size"]).argmax(1).cpu().numpy()
                rep, cm, bd = evaluate_predictions("ftt", seed, split, y[split], pred, days[split], classes)
                reports.append(rep)
                confs.append(cm)
                if bd is not None:
                    bydays.append(bd)
                info[f"{split}_macro_f1"] = float(rep.loc[rep["class"] == "macro_avg", "f1"].iloc[0])
            log.info(f"[ftt seed={seed}] val macro-F1 {info['val_macro_f1']:.4f}  test macro-F1 {info['test_macro_f1']:.4f}")
            ftt_infos.append(info)
            del model
            torch.cuda.empty_cache()
        pd.DataFrame(all_curves).to_csv(RESULTS / "stage2_ftt_training_curves.csv", index=False)
        summary["ftt"] = ftt_infos

    if "xgb" in s2["models"]:
        wtr = class_w[y["train"]]
        bst, info, curve = train_xgb(Xs["train"], y["train"], wtr, Xs["val"], y["val"], K, cfg, log)
        # A loaded booster predicts with ALL trees by default, so the saved xgb.json is truncated to the
        # best (val macro-F1) iteration; the untruncated model is kept for reference only.
        bst.save_model(out_dir / "xgb_all_rounds.json")
        bst[: bst.best_iteration + 1].save_model(out_dir / "xgb.json")
        curve.to_csv(RESULTS / "stage2_xgb_training_curve.csv", index=False)
        for split in ("val", "test"):
            pred = predict_xgb(bst, Xs[split])
            rep, cm, bd = evaluate_predictions("xgb", s2["xgb"]["seed"], split, y[split], pred, days[split], classes)
            reports.append(rep)
            confs.append(cm)
            if bd is not None:
                bydays.append(bd)
            info[f"{split}_macro_f1_eval"] = float(rep.loc[rep["class"] == "macro_avg", "f1"].iloc[0])
        log.info(f"[xgb] val macro-F1 {info['val_macro_f1_eval']:.4f}  test macro-F1 {info['test_macro_f1_eval']:.4f}")
        summary["xgb"] = info

    if reports:
        rep = pd.concat(reports, ignore_index=True)
        # Mean and std across FT-Transformer seeds.
        ftt = rep[rep["model"] == "ftt"]
        if len(ftt):
            g = ftt.groupby(["split", "class"], sort=False)[["precision", "recall", "f1", "support"]]
            mean = g.mean().reset_index().assign(model="ftt", seed="mean")
            std = g.std(ddof=1).reset_index().assign(model="ftt", seed="std")
            rep = pd.concat([rep, mean, std], ignore_index=True)
        rep.to_csv(RESULTS / "stage2_per_class_f1.csv", index=False)
        pd.concat(confs, ignore_index=True).to_csv(RESULTS / "stage2_confusion.csv", index=False)
        bd = pd.concat(bydays, ignore_index=True)
        bftt = bd[bd["model"] == "ftt"]
        if len(bftt):
            g = bftt.groupby(["day", "class"], sort=False)[["precision", "recall", "f1", "support"]]
            bd = pd.concat([bd, g.mean().reset_index().assign(model="ftt", seed="mean"),
                            g.std(ddof=1).reset_index().assign(model="ftt", seed="std")], ignore_index=True)
        bd.to_csv(RESULTS / "stage2_test_f1_by_day.csv", index=False)

        # Compact summary: macro-F1 / weighted-F1 / accuracy per model, split (FT-T as mean +- std).
        rows = []
        for (model, split), g in rep[rep["seed"].astype(str).str.fullmatch(r"\d+")].groupby(["model", "split"]):
            for metric, cls in (("macro_f1", "macro_avg"), ("weighted_f1", "weighted_avg"), ("accuracy", "accuracy")):
                v = g.loc[g["class"] == cls, "f1"].to_numpy()
                rows.append({"model": model, "split": split, "metric": metric, "n_seeds": len(v),
                             "mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else np.nan,
                             "per_seed": ";".join(f"{x:.4f}" for x in v)})
        summ = pd.DataFrame(rows)
        summ.to_csv(RESULTS / "stage2_summary.csv", index=False)
        log.info("Summary:\n" + summ.to_string(index=False))

    if "day_confound" in s2["models"]:
        dc, dc_cm, imp = day_confound(data, Xs, cfg, log)
        dc.to_csv(RESULTS / "stage2_day_confound.csv", index=False)
        dc_cm.to_csv(RESULTS / "stage2_day_confound_confusion.csv")
        imp_df = (pd.DataFrame({"xgb_feature": list(imp), "gain": list(imp.values())})
                  .assign(feature=lambda d: d["xgb_feature"].str[1:].astype(int).map(dict(enumerate(feats))))
                  .sort_values("gain", ascending=False))
        imp_df.to_csv(RESULTS / "stage2_day_confound_importance.csv", index=False)
        summary["day_confound"] = dict(zip(dc["metric"], dc["value"]))
        log.info("Day confusion (benign test):\n" + dc_cm.to_string())
        log.info("Top day-predictive features:\n" + imp_df.head(10).to_string(index=False))

    for p in RESULTS.glob("stage2_*.csv"):
        (run_dir / p.name).write_bytes(p.read_bytes())
    finish_run(run_dir, log, summary, t_start)


if __name__ == "__main__":
    main()
