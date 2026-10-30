# Preprocessing fitted on the training split only:
#   1. log1p on features that are non-negative on train and heavy-tailed (skewness > threshold);
#      inputs to log1p are clipped at 0 so val/test negatives cannot produce NaN.
#   2. Standardise with train mean / std (std == 0 -> 1).
#   3. Optionally clip standardised values to [-clip, clip].
# The fitted parameters are saved as plain JSON so every later stage applies identical transforms.
import json

import numpy as np
from scipy.stats import skew


class Preprocessor:
    def __init__(self, features, log_mask, mean, std, clip, skew_values=None, skew_threshold=None):
        self.features = list(features)
        self.log_mask = np.asarray(log_mask, dtype=bool)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.clip = clip
        self.skew_values = None if skew_values is None else np.asarray(skew_values, dtype=np.float64)
        self.skew_threshold = skew_threshold

    @classmethod
    def fit(cls, df_train, features, skew_threshold, clip):
        X = df_train[features].to_numpy(dtype=np.float64)
        sk = skew(X, axis=0, bias=True)
        sk = np.nan_to_num(sk, nan=0.0)
        log_mask = (X.min(axis=0) >= 0) & (sk > skew_threshold)
        Xt = np.where(log_mask, np.log1p(np.maximum(X, 0.0)), X)
        mean = Xt.mean(axis=0)
        std = Xt.std(axis=0)
        std[std < 1e-12] = 1.0
        return cls(features, log_mask, mean, std, clip, sk, skew_threshold)

    def transform(self, df, stats=None):
        # Returns float32 array; if a dict is given, clip/negative counts are added to it.
        X = df[self.features].to_numpy(dtype=np.float64)
        neg = (X < 0) & self.log_mask
        X = np.where(self.log_mask, np.log1p(np.maximum(X, 0.0)), X)
        Z = (X - self.mean) / self.std
        n_clip = 0
        if self.clip is not None:
            over = np.abs(Z) > self.clip
            n_clip = int(over.sum())
            Z = np.clip(Z, -self.clip, self.clip)
        if stats is not None:
            stats["n_values"] = int(Z.size)
            stats["n_clipped"] = n_clip
            stats["n_negative_into_log1p"] = int(neg.sum())
            stats["clipped_per_feature"] = {f: int(c) for f, c in
                                            zip(self.features, (np.abs((X - self.mean) / self.std) >
                                                                (self.clip or np.inf)).sum(axis=0)) if c}
        return Z.astype(np.float32)

    def to_dict(self):
        return {"features": self.features, "log_mask": self.log_mask.tolist(),
                "mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip,
                "skew_train": None if self.skew_values is None else self.skew_values.tolist(),
                "skew_threshold": self.skew_threshold}

    def save(self, path):
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["features"], d["log_mask"], d["mean"], d["std"], d["clip"],
                   d.get("skew_train"), d.get("skew_threshold"))
