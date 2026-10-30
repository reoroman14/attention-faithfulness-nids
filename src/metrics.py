# Classification metrics shared by Stage 2 (and later stages).
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def macro_f1(y_true, y_pred, n_classes):
    return float(f1_score(y_true, y_pred, labels=np.arange(n_classes), average="macro", zero_division=0))


def per_class_report(y_true, y_pred, classes):
    # One row per class plus macro_avg, weighted_avg and accuracy rows.
    k = len(classes)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(k), zero_division=0)
    rows = [{"class": c, "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
            for i, c in enumerate(classes)]
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(k),
                                                        average=avg, zero_division=0)
        rows.append({"class": f"{avg}_avg", "precision": pa, "recall": ra, "f1": fa, "support": int(s.sum())})
    acc = float((y_true == y_pred).mean())
    rows.append({"class": "accuracy", "precision": np.nan, "recall": np.nan, "f1": acc, "support": int(s.sum())})
    return pd.DataFrame(rows)


def confusion_long(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    rows = [{"true": classes[i], "pred": classes[j], "count": int(cm[i, j])}
            for i in range(len(classes)) for j in range(len(classes))]
    return pd.DataFrame(rows), cm


def by_day_report(y_true, y_pred, days, classes):
    # Per day: per-class metrics for classes present that day, macro-F1 over those classes,
    # and benign false-positive rate (benign rows predicted as any attack).
    rows = []
    for day in pd.unique(days):
        m = days == day
        yt, yp = y_true[m], y_pred[m]
        present = np.unique(yt)
        p, r, f, s = precision_recall_fscore_support(yt, yp, labels=present, zero_division=0)
        for i, c in enumerate(present):
            rows.append({"day": day, "class": classes[c], "precision": p[i], "recall": r[i],
                         "f1": f[i], "support": int(s[i])})
        rows.append({"day": day, "class": "macro_avg_present", "precision": p.mean(), "recall": r.mean(),
                     "f1": f.mean(), "support": int(m.sum())})
        ben = yt == 0
        fpr = float((yp[ben] != 0).mean()) if ben.any() else np.nan
        rows.append({"day": day, "class": "benign_fpr", "precision": np.nan, "recall": np.nan,
                     "f1": fpr, "support": int(ben.sum())})
        rows.append({"day": day, "class": "accuracy", "precision": np.nan, "recall": np.nan,
                     "f1": float((yt == yp).mean()), "support": int(m.sum())})
    return pd.DataFrame(rows)
