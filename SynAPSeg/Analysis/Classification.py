from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
import pandas as pd
import numpy as np

# Optional imports (guarded) for concrete examples
try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    StandardScaler = OneHotEncoder = RobustScaler = PCA = KMeans = None

# from .ClassificationInterp import CompositePreprocessor

# -----------------------------
# 1) Data Schema & Resolver
# -----------------------------

@dataclass
class DataSchema:
    id_cols: List[str] = field(default_factory=list)
    group_cols: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    num_cols: List[str] = field(default_factory=list)

class SchemaResolver:
    @staticmethod
    def resolve(df: pd.DataFrame, schema: DataSchema) -> DataSchema:
        """Validate schema columns and infer missing numeric cols if needed."""
        missing = []
        for col in schema.id_cols + schema.group_cols + schema.cat_cols + schema.num_cols:
            if col not in df.columns:
                missing.append(col)
        if missing:
            raise ValueError(f"Schema columns missing from DataFrame: {missing}")

        # If num_cols not specified, infer as numeric and not in other roles
        if not schema.num_cols:
            reserved = set(schema.id_cols + schema.group_cols + schema.cat_cols)
            schema.num_cols = [
                c for c in df.columns
                if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
            ]
        return schema

# -----------------------------
# 2) Preprocessing (ABC)
# -----------------------------

class BasePreprocessor(ABC):
    name: str = "base_preprocessor"

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BasePreprocessor":
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return ONLY constructor kwargs (no learned state)."""
        pass
    
    def clone(self) -> "BasePreprocessor":
        return type(self)(**self.get_params())
    
# Example concrete preprocessors

class StandardScalePreprocessor(BasePreprocessor):
    name = "standard_scale"

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        if StandardScaler is None:
            raise ImportError("sklearn is required for StandardScalePreprocessor")
        self._init_params = {"with_mean": with_mean, "with_std": with_std}
        self._scaler = StandardScaler(**self._init_params)

    def fit(self, X, y=None): self._scaler.fit(X); return self
    def transform(self, X): return self._scaler.transform(X)
    def get_params(self): return dict(self._init_params)


class RobustScalePreprocessor(BasePreprocessor):
    name = "robust_scale"

    def __init__(self, with_centering: bool = True, with_scaling: bool = True):
        if RobustScaler is None:
            raise ImportError("sklearn is required for RobustScalePreprocessor")
        self._init_params = {"with_centering": with_centering, "with_scaling": with_scaling}
        self._scaler = RobustScaler(**self._init_params)

    def fit(self, X, y=None): self._scaler.fit(X); return self
    def transform(self, X): return self._scaler.transform(X)
    def get_params(self): return dict(self._init_params)


class PCAPreprocessor(BasePreprocessor):
    name = "pca"

    def __init__(self, n_components: Union[int, float] = 0.95, random_state: Optional[int] = 0):
        if PCA is None:
            raise ImportError("sklearn is required for PCAPreprocessor")
        self._init_params = {"n_components": n_components, "random_state": random_state}
        self._pca = PCA(**self._init_params)

    def fit(self, X, y=None): self._pca.fit(X); return self
    def transform(self, X): return self._pca.transform(X)
    def get_params(self): return dict(self._init_params)



# -----------------------------
# 3) Categorical Encoding (ABC)
# -----------------------------

class BaseEncoder(ABC):
    name: str = "base_encoder"

    @abstractmethod
    def fit(self, df: pd.DataFrame, schema: DataSchema) -> "BaseEncoder":
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame, schema: DataSchema) -> Tuple[pd.DataFrame, List[str]]:
        """Return transformed df and the list of new numeric feature columns created."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        pass

    def clone(self) -> "BaseEncoder":
        return type(self)(**self.get_params())

# concrete encoders
import inspect

class OneHotCategoricalEncoder(BaseEncoder):
    name = "one_hot"

    def __init__(self, drop: Optional[str] = None, handle_unknown: str = "ignore", sparse_output: bool = False):
        if OneHotEncoder is None:
            raise ImportError("sklearn is required for OneHotCategoricalEncoder")

        # version-safe kwargs for sklearn
        kwargs = dict(drop=drop, handle_unknown=handle_unknown)
        if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
            kwargs["sparse_output"] = sparse_output  # sklearn >=1.2
        else:
            kwargs["sparse"] = not sparse_output      # sklearn <1.2 (dense => sparse=False)

        self._encoder = OneHotEncoder(**kwargs)
        self._feature_names: List[str] = []
        self._init_params = {"drop": drop, "handle_unknown": handle_unknown, "sparse_output": sparse_output}

    def fit(self, df: pd.DataFrame, schema: DataSchema):
        if not schema.cat_cols:
            return self
        X_cat = df[schema.cat_cols].astype("category")
        self._encoder.fit(X_cat)
        try:
            self._feature_names = list(self._encoder.get_feature_names_out(schema.cat_cols))
        except Exception:
            # fallback name construction
            cats = self._encoder.categories_
            self._feature_names = [f"{col}__{lvl}" for col, levels in zip(schema.cat_cols, cats) for lvl in levels]
        return self

    def transform(self, df: pd.DataFrame, schema: DataSchema) -> Tuple[pd.DataFrame, List[str]]:
        if not schema.cat_cols:
            return df, []
        X_cat = df[schema.cat_cols].astype("category")
        X_enc = self._encoder.transform(X_cat)
        enc_df = pd.DataFrame(X_enc, columns=self._feature_names, index=df.index)
        out = pd.concat([df.drop(columns=schema.cat_cols), enc_df], axis=1)
        return out, self._feature_names

    def get_params(self) -> Dict[str, Any]:
        # ONLY constructor kwargs — NO learned state like feature names
        return dict(self._init_params)


class BinaryToIntEncoder(BaseEncoder):
    name = "binary_to_int"

    def __init__(self, binary_cols: Optional[List[str]] = None):
        self._init_params = {"binary_cols": binary_cols}
        self._binary_cols = binary_cols or []
        self._effective_cols: List[str] = []

    def fit(self, df: pd.DataFrame, schema: DataSchema):
        self._effective_cols = self._init_params["binary_cols"] or [
            c for c in df.columns if pd.api.types.is_bool_dtype(df[c])
        ]
        return self

    def transform(self, df: pd.DataFrame, schema: DataSchema) -> Tuple[pd.DataFrame, List[str]]:
        if not self._effective_cols:
            return df, []
        out = df.copy()
        for c in self._effective_cols:
            out[c] = out[c].astype(int)
        return out, list(self._effective_cols)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._init_params)




# -----------------------------
# 4) Clustering (ABC)
# -----------------------------

class BaseClusterer(ABC):
    name: str = "base_clusterer"

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseClusterer":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard labels. If the algorithm doesn't support predict, you can implement a nearest-centroid or raise NotImplementedError."""
        pass

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)

    def soft_scores(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Optional: return per-cluster scores/probabilities if available. Default None."""
        return None

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        pass

    def clone(self) -> "BaseClusterer":
        return type(self)(**self.get_params())

# Example concrete clusterers

class KMeansClusterer(BaseClusterer):
    name = "kmeans"

    def __init__(self, n_clusters: int = 5, random_state: Optional[int] = 0, **kwargs):
        if KMeans is None:
            raise ImportError("sklearn is required for KMeansClusterer")
        self._init_params = {"n_clusters": n_clusters, "random_state": random_state}
        self._init_params.update(kwargs)
        self._model = KMeans(**self._init_params)

    def fit(self, X, y=None):
        self._model.fit(X)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def soft_scores(self, X):
        # k-means doesn't have probabilities; return distances to cluster centers (negated) as a proxy
        try:
            # Using transform returns distances to cluster centers
            dists = self._model.transform(X)
            # convert to similarity-like scores
            sim = 1 / (1 + dists)
            return sim
        except Exception:
            return None

    # def get_params(self):
    #     return {"n_clusters": int(self._model.n_clusters), "random_state": self._model.random_state}
    def get_params(self): 
        return dict(self._init_params)

try:
    from sklearn_extra.cluster import KMedoids
except Exception:
    KMedoids = None

class PAMClusterer(BaseClusterer):
    """
    Partition Around Medoids (PAM) via sklearn-extra's KMedoids.
    Works with metrics: 'euclidean', 'manhattan', or any supported by KMedoids.
    """
    name = "pam"

    def __init__(
        self,
        n_clusters: int = 5,
        metric: str = "euclidean",          # 'euclidean' or 'manhattan' are common; others per sklearn-extra
        init: str = "heuristic",            # 'heuristic' or 'random'
        method: str = "pam",                # keep 'pam' (the standard algorithm); 'alternate' also supported
        max_iter: int = 300,
        random_state: Optional[int] = 0,
    ):
        if KMedoids is None:
            raise ImportError(
                "sklearn-extra is required for PAMClusterer. "
                "Install with: pip install scikit-learn-extra"
            )
        self._init_params = {
            "n_clusters": n_clusters,
            "metric": metric,
            "init": init,
            "method": method,
            "max_iter": max_iter,
            "random_state": random_state,
        }
        self._model = KMedoids(**self._init_params)

        # cached after fit
        self._centers_: Optional[np.ndarray] = None  # medoid coordinates (cluster_centers_)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PAMClusterer":
        self._model.fit(X)
        # sklearn-extra exposes medoid coordinates as cluster_centers_
        self._centers_ = np.asarray(self._model.cluster_centers_, dtype=float)
        return self

    def _pairwise_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Minimal, dependency-free distance computation for euclidean/manhattan.
        Falls back to euclidean if unknown metric (but you should match model's metric).
        Returns shape (n_samples, n_clusters).
        """
        if self._init_params["metric"] == "manhattan":
            # L1 distance
            # broadcast: (n_samples, 1, n_features) - (1, n_clusters, n_features)
            diffs = np.abs(X[:, None, :] - centers[None, :, :])
            return diffs.sum(axis=2)
        else:
            # default to euclidean
            diffs = X[:, None, :] - centers[None, :, :]
            # we can skip sqrt for argmin, but keep it for soft scores to be interpretable
            return np.sqrt((diffs ** 2).sum(axis=2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._centers_ is None:
            # Some sklearn-extra versions provide predict; use if available
            if hasattr(self._model, "predict"):
                return self._model.predict(X)
            raise ValueError("PAMClusterer must be fit before calling predict()")

        dists = self._pairwise_distances(X, self._centers_)
        return dists.argmin(axis=1)

    def soft_scores(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self._centers_ is None:
            return None
        dists = self._pairwise_distances(X, self._centers_)
        # similarity-like scores in [0,1): larger is closer
        return 1.0 / (1.0 + dists)

    def get_params(self) -> Dict[str, Any]:
        # constructor kwargs only (no learned state)
        return dict(self._init_params)

class ClaraPAMClusterer(BaseClusterer):
    """
    CLARA-style K-Medoids:
      - Fit medoids on subsamples (repeats)
      - Assign full X to nearest medoid (chunked)
      - Choose the best medoids by full-data cost (sum of distances)
    Pros: scales to millions (needs only O(n·k) memory/time per pass)
    Cons: approximate; quality depends on subsample size & repeats
    """
    name = "clara_pam"

    def __init__(
        self,
        n_clusters: int = 5,
        metric: str = "euclidean",      # 'euclidean' or 'manhattan' supported here
        subsample_size: int = 100_000,
        repeats: int = 5,
        chunk_size: int = 200_000,      # for assign-only passes
        init: str = "heuristic",
        method: str = "pam",
        max_iter: int = 300,
        random_state: Optional[int] = 0,
    ):
        if KMedoids is None:
            raise ImportError("scikit-learn-extra is required for ClaraPAMClusterer (pip install scikit-learn-extra)")

        self._init_params = dict(
            n_clusters=n_clusters, metric=metric, subsample_size=subsample_size, repeats=repeats,
            chunk_size=chunk_size, init=init, method=method, max_iter=max_iter, random_state=random_state
        )

        self._rng = np.random.default_rng(random_state)
        self._centers_: Optional[np.ndarray] = None  # medoid coordinates
        self._labels_: Optional[np.ndarray] = None

    def _pairwise(self, A, B):
        if self._init_params["metric"] == "manhattan":
            return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
        diffs = A[:, None, :] - B[None, :, :]
        return np.sqrt((diffs * diffs).sum(axis=2))

    def _assign_full_chunked(self, X, medoids, return_scores=False):
        n = X.shape[0]
        k = medoids.shape[0]
        chunk = self._init_params["chunk_size"]
        labels = np.empty(n, dtype=int)
        if return_scores:
            # similarity-like score 1/(1+dist) per cluster; streamed -> collect then stack
            scores_list = []
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            D = self._pairwise(X[start:stop], medoids)      # (m, k)
            labels[start:stop] = D.argmin(axis=1)
            if return_scores:
                scores_list.append(1.0 / (1.0 + D))
        if return_scores:
            return labels, np.vstack(scores_list)
        return labels

    def _total_within_distance(self, X, labels, medoids):
        # sum of distances of each point to its assigned medoid (chunked)
        n = X.shape[0]
        chunk = self._init_params["chunk_size"]
        total = 0.0
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            D = self._pairwise(X[start:stop], medoids)      # (m, k)
            idx = labels[start:stop]
            total += D[np.arange(stop - start), idx].sum()
        return float(total)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ClaraPAMClusterer":
        n = X.shape[0]
        s = min(self._init_params["subsample_size"], n)
        best = {"cost": np.inf, "medoids": None, "labels": None}

        for r in range(self._init_params["repeats"]):
            idx = self._rng.choice(n, size=s, replace=False)
            Xs = X[idx]
            km = KMedoids(
                n_clusters=self._init_params["n_clusters"],
                metric=self._init_params["metric"],
                init=self._init_params["init"],
                method=self._init_params["method"],
                max_iter=self._init_params["max_iter"],
                random_state=(None if self._init_params["random_state"] is None else self._init_params["random_state"] + r),
            )
            km.fit(Xs)  # fits on subsample only (builds s×s matrix — feasible)
            medoids = np.asarray(km.cluster_centers_, dtype=float)

            labels_full = self._assign_full_chunked(X, medoids)
            cost = self._total_within_distance(X, labels_full, medoids)
            if cost < best["cost"]:
                best = {"cost": cost, "medoids": medoids, "labels": labels_full}

        self._centers_ = best["medoids"]
        self._labels_ = best["labels"]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._centers_ is None:
            raise ValueError("ClaraPAMClusterer must be fit before predict().")
        return self._assign_full_chunked(X, self._centers_)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self._labels_.astype(int)

    def soft_scores(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self._centers_ is None:
            return None
        # compute in chunks
        n = X.shape[0]
        chunk = self._init_params["chunk_size"]
        scores = []
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            D = self._pairwise(X[start:stop], self._centers_)
            scores.append(1.0 / (1.0 + D))
        return np.vstack(scores)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._init_params)
    
# -----------------------------
# 5) Grouping strategies
# -----------------------------

class BaseGroupingStrategy(ABC):
    name: str = "base_grouping"

    @abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        encoder: BaseEncoder,
        preprocessor: BasePreprocessor,
        clusterer: BaseClusterer
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute encoding->preprocessing->clustering with the chosen grouping logic.
        Returns:
            result_df: original df with appended columns (cluster_label, optional scores).
            meta: dict with pipeline metadata (params, names, grouping info).
        """
        pass
    
class ClusterWithinGroups(BaseGroupingStrategy):
    name = "within_groups"

    def run(self, df, schema, encoder, preprocessor, clusterer):
        if not schema.group_cols:
            # raise ValueError("No group_cols specified for 'within_groups' strategy.")
            schema.group_cols = ['dummy_group']
            df['dummy_group'] = True

        results = []
        meta = {"groups": [], "strategy": self.name}

        for group_key, gdf in df.groupby(schema.group_cols, dropna=False, sort=False):
            group_key = group_key if isinstance(group_key, tuple) else (group_key,)
            meta["groups"].append(tuple(zip(schema.group_cols, group_key)))

            # ---- clone fresh instances (no learned state carried across groups)
            enc = encoder.clone()
            prep = preprocessor.clone()
            clus = clusterer.clone()

            enc.fit(gdf, schema)
            g_enc, _ = enc.transform(gdf, schema)

            feat_cols = [c for c in g_enc.columns if c not in (schema.id_cols + schema.group_cols)]
            X = g_enc[feat_cols].to_numpy(dtype=float)

            Xp = prep.fit_transform(X)
            labels = clus.fit_predict(Xp)
            scores = clus.soft_scores(Xp)

            out = gdf.copy()
            out["cluster_label"] = labels.astype(int)
            if scores is not None:
                for j in range(scores.shape[1]):
                    out[f"cluster_score_{j}"] = scores[:, j]
            results.append(out)

        result_df = pd.concat(results).sort_index()
        meta.update({
            "encoder": getattr(encoder, "name", type(encoder).__name__),
            "preprocessor": getattr(preprocessor, "name", type(preprocessor).__name__),
            "clusterer": getattr(clusterer, "name", type(clusterer).__name__),
            "schema": schema.__dict__,
        })
        return result_df, meta


class UseGroupsAsFeatures(BaseGroupingStrategy):
    name = "use_groups_as_features"

    def run(self, df, schema, encoder, preprocessor, clusterer):
        enc = encoder.clone()
        prep = preprocessor.clone()
        clus = clusterer.clone()

        work_schema = DataSchema(
            id_cols=schema.id_cols,
            group_cols=schema.group_cols,
            cat_cols=list(set(schema.cat_cols + schema.group_cols)),
            num_cols=schema.num_cols,
        )
        enc.fit(df, work_schema)
        df_enc, _ = enc.transform(df, work_schema)

        reserved = set(schema.id_cols + schema.group_cols)
        feat_cols = [c for c in df_enc.columns if c not in reserved]
        X = df_enc[feat_cols].to_numpy(dtype=float)

        Xp = prep.fit_transform(X)
        labels = clus.fit_predict(Xp)
        scores = clus.soft_scores(Xp)

        out = df.copy()
        out["cluster_label"] = labels.astype(int)
        if scores is not None:
            for j in range(scores.shape[1]):
                out[f"cluster_score_{j}"] = scores[:, j]

        meta = {
            "strategy": self.name,
            "encoder": getattr(encoder, "name", type(encoder).__name__),
            "preprocessor": getattr(preprocessor, "name", type(preprocessor).__name__),
            "clusterer": getattr(clusterer, "name", type(clusterer).__name__),
            "schema": schema.__dict__,
            "feature_cols": feat_cols,
        }
        return out, meta

# -----------------------------
# 6) Registries / Factory
# -----------------------------

class _Registry:
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def register(self, name: str):
        def deco(cls):
            self._store[name] = cls
            return cls
        return deco

    def get(self, name: str):
        if name not in self._store:
            raise KeyError(f"Unknown component: {name}. Available: {list(self._store.keys())}")
        return self._store[name]

    def available(self) -> List[str]:
        return list(self._store.keys())


PreprocessorRegistry = _Registry()
EncoderRegistry = _Registry()
ClustererRegistry = _Registry()
GroupingRegistry = _Registry()

# Register the example implementations
PreprocessorRegistry.register(StandardScalePreprocessor.name)(StandardScalePreprocessor)
PreprocessorRegistry.register(RobustScalePreprocessor.name)(RobustScalePreprocessor)
PreprocessorRegistry.register(PCAPreprocessor.name)(PCAPreprocessor)


EncoderRegistry.register(OneHotCategoricalEncoder.name)(OneHotCategoricalEncoder)
EncoderRegistry.register(BinaryToIntEncoder.name)(BinaryToIntEncoder)

ClustererRegistry.register(KMeansClusterer.name)(KMeansClusterer)
ClustererRegistry.register(PAMClusterer.name)(PAMClusterer)
ClustererRegistry.register(ClaraPAMClusterer.name)(ClaraPAMClusterer)

GroupingRegistry.register(ClusterWithinGroups.name)(ClusterWithinGroups)
GroupingRegistry.register(UseGroupsAsFeatures.name)(UseGroupsAsFeatures)


########################
# interp
########################
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as skPipeline

from scipy import stats
from statsmodels.stats.multitest import multipletests

ProfilerRegistry = _Registry()
SelectorRegistry = _Registry()
UnivariateTestRegistry = _Registry()
ImportanceRegistry = _Registry()
BackprojectorRegistry = _Registry()
InterpGroupingRegistry = _Registry()


# =========== Feature Matrix Builder ===========
class FeatureBuilder:
    """
    Builds the feature matrix X for interpretation.
    By default: numeric columns from schema + optional extra feature cols user supplies.
    If you want to include encoded categoricals, pass an already-encoded df or provide `extra_feature_cols`
    that include one-hot columns.
    """
    @staticmethod
    def build_features(
        df: pd.DataFrame,
        schema: DataSchema,
        extra_feature_cols: Optional[List[str]] = None,
        drop_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        extra_feature_cols = extra_feature_cols or []
        drop_cols = set(drop_cols or [])
        reserved = set(schema.id_cols + schema.group_cols)
        # default numeric features (raw)
        base_num = [c for c in schema.num_cols if c in df.columns]
        # include additional columns explicitly
        feats = [c for c in (base_num + extra_feature_cols) if c in df.columns and c not in reserved and c not in drop_cols]
        X = df[feats].copy()
        return X, feats


# =========== ABCs ===========

class BaseProfiler(ABC):
    name: str = "base_profiler"

    @abstractmethod
    def compute(self, X: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """
        Return dict with keys like:
          - counts: pd.Series (# per cluster)
          - means: pd.DataFrame (per-cluster means)
          - z_profiles: pd.DataFrame (standardized per-cluster feature effects)
          - overall_mean, overall_std
        """
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]: ...


class BaseTopDriverSelector(ABC):
    name: str = "base_selector"

    @abstractmethod
    def select(self, z_profiles: pd.DataFrame) -> Dict[Any, pd.DataFrame]:
        """Return {cluster_id: DataFrame[feature, score]} ranked by |score|."""
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]: ...


class BaseUnivariateTester(ABC):
    name: str = "base_univariate"

    @abstractmethod
    def run(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """
        Return DataFrame: [feature, statistic, p_value, effect_size, (optional) q_value]
        """
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]: ...


class BaseGlobalImportance(ABC):
    name: str = "base_importance"

    @abstractmethod
    def run(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """
        Return DataFrame: [feature, importance] sorted desc
        """
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]: ...


class BaseBackprojector(ABC):
    name: str = "base_backprojector"

    @abstractmethod
    def run(self, artifacts: Dict[str, Any], feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Optional: returns DataFrame of centroids in original feature space.
        `artifacts` can contain fitted objects (e.g., 'kmeans', 'scaler', 'pca').
        Return None if not applicable.
        """
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]: ...


class BaseInterpGroupingStrategy(ABC):
    name: str = "base_interpretation_grouping"

    @abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        labels_col: str,
        components: Dict[str, Any],
        extra_feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes the interpretation across the dataset (globally or per group).
        Returns:
          - results_df: input df with attached per-feature/cluster summaries exploded as needed (or df unchanged)
          - meta: dictionary with rich outputs (profiles, top drivers, tests, importance, backprojection)
        """
        ...


# =========== Concrete Implementations ===========

# -- Profiler: Z-score effect vs global
@ProfilerRegistry.register("z_profile")
class ZProfileProfiler(BaseProfiler):
    name = "z_profile"

    def __init__(self, ddof: int = 0, replace_zero_std_with_nan: bool = True):
        self._init = dict(ddof=ddof, replace_zero_std_with_nan=replace_zero_std_with_nan)

    def compute(self, X: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        df = X.copy()
        df["_cluster"] = labels
        overall_mean = X.mean(numeric_only=True)
        overall_std = X.std(ddof=self._init["ddof"], numeric_only=True)
        if self._init["replace_zero_std_with_nan"]:
            overall_std = overall_std.replace(0, np.nan)

        means = df.groupby("_cluster").mean(numeric_only=True)
        zprof = (means - overall_mean) / overall_std
        counts = df["_cluster"].value_counts().sort_index()
        return {
            "counts": counts,
            "means": means,
            "z_profiles": zprof,
            "overall_mean": overall_mean,
            "overall_std": overall_std
        }

    def get_params(self) -> Dict[str, Any]:
        return dict(self._init)


# -- Top drivers: highest |z| features
@SelectorRegistry.register("topn_by_absz")
class TopNByAbsZSelector(BaseTopDriverSelector):
    name = "topn_by_absz"

    def __init__(self, top_n: int = 10):
        self._init = dict(top_n=top_n)

    def select(self, z_profiles: pd.DataFrame) -> Dict[Any, pd.DataFrame]:
        out: Dict[Any, pd.DataFrame] = {}
        for c in z_profiles.index:
            z = z_profiles.loc[c]
            top_idx = (z.abs().sort_values(ascending=False).head(self._init["top_n"]).index.tolist())
            out[c] = pd.DataFrame({
                "feature": top_idx,
                "score": z[top_idx].values
            }).sort_values("score", key=np.abs, ascending=False).reset_index(drop=True)
        return out

    def get_params(self) -> Dict[str, Any]:
        return dict(self._init)


# -- Univariate tests: ANOVA / Kruskal
@UnivariateTestRegistry.register("anova")
class ANOVAUnivariateTester(BaseUnivariateTester):
    name = "anova"

    def __init__(self, fdr_method: Optional[str] = "fdr_bh"):
        self._init = dict(fdr_method=fdr_method)

    def run(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        if stats is None:
            raise ImportError("scipy is required for ANOVAUnivariateTester")
        df = X.copy()
        df["_cluster"] = labels
        groups = [g.drop(columns="_cluster") for _, g in df.groupby("_cluster")]

        rows = []
        for col in X.columns:
            cols = [g[col].dropna().values for g in groups]
            # ANOVA
            stat, p = stats.f_oneway(*cols)
            # eta^2 (biased)
            overall_mean = df[col].mean()
            ss_between = sum(len(x) * (x.mean() - overall_mean) ** 2 for x in cols)
            ss_total = float(((df[col] - overall_mean) ** 2).sum())
            eta2 = ss_between / ss_total if ss_total != 0 else np.nan
            rows.append((col, stat, p, eta2))

        out = pd.DataFrame(rows, columns=["feature", "statistic", "p_value", "effect_size"]).sort_values(
            ["p_value", "effect_size"], ascending=[True, False]
        )
        if self._init["fdr_method"] and multipletests is not None:
            out["q_value"] = multipletests(out["p_value"].values, method=self._init["fdr_method"])[1]
        return out.reset_index(drop=True)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._init)


@UnivariateTestRegistry.register("kruskal")
class KruskalUnivariateTester(BaseUnivariateTester):
    name = "kruskal"

    def __init__(self, fdr_method: Optional[str] = "fdr_bh"):
        self._init = dict(fdr_method=fdr_method)

    def run(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        if stats is None:
            raise ImportError("scipy is required for KruskalUnivariateTester")
        df = X.copy()
        df["_cluster"] = labels
        groups = [g.drop(columns="_cluster") for _, g in df.groupby("_cluster")]

        rows = []
        for col in X.columns:
            cols = [g[col].dropna().values for g in groups]
            stat, p = stats.kruskal(*cols)
            n = sum(len(x) for x in cols)
            k = len(cols)
            eps2 = (stat - k + 1) / (n - k) if n != k else np.nan
            rows.append((col, stat, p, eps2))

        out = pd.DataFrame(rows, columns=["feature", "statistic", "p_value", "effect_size"]).sort_values(
            ["p_value", "effect_size"], ascending=[True, False]
        )
        if self._init["fdr_method"] and multipletests is not None:
            out["q_value"] = multipletests(out["p_value"].values, method=self._init["fdr_method"])[1]
        return out.reset_index(drop=True)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._init)


# -- Global importance: RF + permutation importance
@ImportanceRegistry.register("rf_permutation")
class RFPermutationImportance(BaseGlobalImportance):
    name = "rf_permutation"

    def __init__(self, n_estimators: int = 400, min_samples_leaf: int = 2, n_repeats: int = 15,
                 random_state: int = 42, n_jobs: int = -1, scale_before_rf: bool = True):
        if RandomForestClassifier is None or permutation_importance is None:
            raise ImportError("sklearn is required for RFPermutationImportance")
        self._init = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_repeats=n_repeats,
                          random_state=random_state, n_jobs=n_jobs, scale_before_rf=scale_before_rf)

    def run(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        Z = X.values
        if self._init["scale_before_rf"]:
            if StandardScaler is None:
                raise ImportError("sklearn is required for StandardScaler")
            Z = StandardScaler().fit_transform(Z)

        rf = RandomForestClassifier(
            n_estimators=self._init["n_estimators"],
            max_depth=None,
            min_samples_leaf=self._init["min_samples_leaf"],
            random_state=self._init["random_state"],
            n_jobs=self._init["n_jobs"]
        )
        rf.fit(Z, labels)
        perm = permutation_importance(
            rf, Z, labels, n_repeats=self._init["n_repeats"],
            random_state=self._init["random_state"], n_jobs=self._init["n_jobs"]
        )
        imp = pd.DataFrame({"feature": X.columns, "importance": perm.importances_mean})
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._init)


# -- Backprojector: PCA centroids -> original space (requires artifacts)
@BackprojectorRegistry.register("pca_backproject")
class PCACentroidBackprojector(BaseBackprojector):
    name = "pca_backproject"

    def __init__(self): self._init = {}

    def run(self, artifacts: Dict[str, Any], feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Expects artifacts to contain:
          - 'kmeans': fitted sklearn KMeans (has .cluster_centers_)
          - 'scaler': fitted StandardScaler (with inverse_transform)
          - 'pca': fitted PCA (with inverse_transform)
        Returns centroids in original feature space as DataFrame, or None if missing.
        """
        # TODO what are these supposed to be? How do I access them from the pipeline?
        km = artifacts.get("kmeans")
        scaler = artifacts.get("scaler")
        pca = artifacts.get("pca")
        if km is None or scaler is None or pca is None:
            return None
        centroids_pc = km.cluster_centers_
        centroids_orig = scaler.inverse_transform(pca.inverse_transform(centroids_pc))
        return pd.DataFrame(centroids_orig, columns=feature_names)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._init)


# =========== Grouping Strategies for Interpretation ===========

@InterpGroupingRegistry.register("global")
class InterpretGlobally(BaseInterpGroupingStrategy):
    name = "global"

    def run(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        labels_col: str,
        components: Dict[str, Any],
        extra_feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if labels_col not in df.columns:
            raise ValueError(f"labels_col '{labels_col}' not found in df")
        labels = df[labels_col].to_numpy()

        X, feat_cols = FeatureBuilder.build_features(df, schema, extra_feature_cols=extra_feature_cols)

        profiles = components["profiler"].compute(X, labels)
        z_profiles = profiles.get("z_profiles")
        top_drivers = components["selector"].select(z_profiles) if z_profiles is not None else {}

        tests = components["univariate"].run(X, labels)
        
        print("running importance...")
        importance = components["importance"].run(X, labels)

        backproj_df = None
        if components.get("backprojector") is not None:
            print("running backprojection...")
            backproj_df = components["backprojector"].run(components.get("artifacts", {}), feat_cols)

        meta = {
            "feature_cols": feat_cols,
            "profiles": profiles,
            "top_drivers": top_drivers,
            "univariate": tests,
            "importance": importance,
            "backprojection": backproj_df
        }
        return df, meta


@InterpGroupingRegistry.register("within_groups")
class InterpretWithinGroups(BaseInterpGroupingStrategy):
    """
    Mirrors your clustering 'within_groups': interpret separately per group in schema.group_cols.
    """
    name = "within_groups"

    def run(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        labels_col: str,
        components: Dict[str, Any],
        extra_feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not schema.group_cols:
            raise ValueError("No group_cols specified for 'within_groups' interpretation.")
        if labels_col not in df.columns:
            raise ValueError(f"labels_col '{labels_col}' not found in df")

        meta_per_group: Dict[Tuple, Dict[str, Any]] = {}
        for key, gdf in df.groupby(schema.group_cols, dropna=False, sort=False):
            
            labels = gdf[labels_col].to_numpy()

            print("building features...")
            X, feat_cols = FeatureBuilder.build_features(gdf, schema, extra_feature_cols=extra_feature_cols)

            print("building profiles...")
            profiles = components["profiler"].compute(X, labels)
            z_profiles = profiles.get("z_profiles")
            top_drivers = components["selector"].select(z_profiles) if z_profiles is not None else {}

            tests = components["univariate"].run(X, labels)
            
            print("running importance...")
            importance = components["importance"].run(X, labels)

            backproj_df = None
            if components.get("backprojector") is not None:
                print("running backprojection...")
                backproj_df = components["backprojector"].run(components.get("artifacts", {}), feat_cols)
            
            key_tuple = key if isinstance(key, tuple) else (key,)

            meta_per_group[key_tuple] = {
                "group": list(zip(schema.group_cols, key_tuple)),
                "feature_cols": feat_cols,
                "profiles": profiles,
                "top_drivers": top_drivers,
                "univariate": tests,
                "importance": importance,
                "backprojection": backproj_df
            }

        meta = {"groups": meta_per_group, "strategy": self.name}
        return df, meta


# =========== Orchestrator & Config ===========

@dataclass
class InterpretationConfig:
    profiler: Dict[str, Any] = field(default_factory=lambda: {"name": "z_profile", "params": {"ddof": 0}})
    selector: Dict[str, Any] = field(default_factory=lambda: {"name": "topn_by_absz", "params": {"top_n": 8}})
    univariate: Dict[str, Any] = field(default_factory=lambda: {"name": "anova", "params": {"fdr_method": "fdr_bh"}})
    importance: Dict[str, Any] = field(default_factory=lambda: {"name": "rf_permutation", "params": {"n_estimators": 400}})
    grouping: Dict[str, Any] = field(default_factory=lambda: {"name": "global", "params": {}})
    backprojector: Optional[Dict[str, Any]] = field(default_factory=lambda: None)
    # Optional: if you have engineered/encoded columns to include in interpretation:
    extra_feature_cols: List[str] = field(default_factory=list)

class ClusterInterpretation:
    """
    High-level runner that mirrors the classification pipeline ergonomics.
    """
    def __init__(self, schema: DataSchema, config: InterpretationConfig, artifacts: Optional[Dict[str, Any]] = None):
        self.schema = schema
        self.config = config
        self._profiler: BaseProfiler = self._build(ProfilerRegistry, config.profiler)
        self._selector: BaseTopDriverSelector = self._build(SelectorRegistry, config.selector)
        self._univariate: BaseUnivariateTester = self._build(UnivariateTestRegistry, config.univariate)
        self._importance: BaseGlobalImportance = self._build(ImportanceRegistry, config.importance)
        self._grouping: BaseInterpGroupingStrategy = self._build(InterpGroupingRegistry, config.grouping)
        self._backprojector: Optional[BaseBackprojector] = None
        if config.backprojector:
            self._backprojector = self._build(BackprojectorRegistry, config.backprojector)
        # Artifacts can hold fitted objects from your classifier run (kmeans/scaler/pca/etc.)
        self._artifacts = artifacts or {}

    @staticmethod
    def _build(registry: _Registry, spec: Dict[str, Any]):
        cls = registry.get(spec["name"])
        return cls(**spec.get("params", {}))

    @classmethod
    def from_config(cls, schema: DataSchema, config_dict: Dict[str, Any], artifacts: Optional[Dict[str, Any]] = None):
        cfg = InterpretationConfig(
            profiler=config_dict.get("profiler", {"name": "z_profile", "params": {"ddof": 0}}),
            selector=config_dict.get("selector", {"name": "topn_by_absz", "params": {"top_n": 8}}),
            univariate=config_dict.get("univariate", {"name": "anova", "params": {"fdr_method": "fdr_bh"}}),
            importance=config_dict.get("importance", {"name": "rf_permutation", "params": {"n_estimators": 400}}),
            grouping=config_dict.get("grouping", {"name": "global", "params": {}}),
            backprojector=config_dict.get("backprojector", None),
            extra_feature_cols=config_dict.get("extra_feature_cols", [])
        )
        return cls(schema=schema, config=cfg, artifacts=artifacts)

    def run(self, df: pd.DataFrame, labels_col: str = "cluster_label") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        components = {
            "profiler": self._profiler,
            "selector": self._selector,
            "univariate": self._univariate,
            "importance": self._importance,
            "backprojector": self._backprojector,
            "artifacts": self._artifacts
        }
        result_df, meta = self._grouping.run(
            df=df,
            schema=self.schema,
            labels_col=labels_col,
            components=components,
            extra_feature_cols=self.config.extra_feature_cols
        )
        meta.update({
            "interpreter_meta": {
                "profiler": self._profiler.name,
                "selector": self._selector.name,
                "univariate": self._univariate.name,
                "importance": self._importance.name,
                "grouping": self._grouping.name,
                "backprojector": None if self._backprojector is None else self._backprojector.name,
                "params": {
                    "profiler": self._profiler.get_params(),
                    "selector": self._selector.get_params(),
                    "univariate": self._univariate.get_params(),
                    "importance": self._importance.get_params(),
                    "backprojector": {} if self._backprojector is None else self._backprojector.get_params(),
                }
            }
        })
        return result_df, meta



class CompositePreprocessor(BasePreprocessor):
    """
    Wrap multiple sklearn-like transformers (each with fit/transform/optional inverse_transform)
    into a single BasePreprocessor with inverse_transform support.
    """
    name = "composite"

    def __init__(self, steps: Sequence[Tuple[str, Any]]):
        """
        steps: list of (name, transformer) where each transformer supports fit/transform.
        Example:
            CompositePreprocessor([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=0))
            ])
        """
        self._steps = skPipeline(steps)
        self._init_params = {"steps": [(n, type(t).__name__) for n, t in steps]}

    def fit(self, X, y=None):
        self._steps.fit(X, y)
        return self

    def transform(self, X):
        return self._steps.transform(X)

    def inverse_transform(self, X):
        """Works if *all* steps implement inverse_transform."""
        return self._steps.inverse_transform(X)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._init_params)
    
PreprocessorRegistry.register(CompositePreprocessor.name)(CompositePreprocessor)



# -----------------------------
# 7) Pipeline Orchestrator
# -----------------------------

@dataclass
class PipelineConfig:
    preprocessor: Dict[str, Any] = field(default_factory=lambda: {"name": "standard_scale", "params": {}})
    encoder: Dict[str, Any] = field(default_factory=lambda: {"name": "one_hot", "params": {"drop": None}})
    clusterer: Dict[str, Any] = field(default_factory=lambda: {"name": "kmeans", "params": {"n_clusters": 5, "random_state": 0}})
    grouping: Dict[str, Any] = field(default_factory=lambda: {"name": "within_groups", "params": {}})

class ClassificationPipeline:
    def __init__(self, schema: DataSchema, config: PipelineConfig):
        self.schema = schema
        self.config = config
        self._encoder: BaseEncoder = self._build(EncoderRegistry, config.encoder)
        self._preprocessor: BasePreprocessor = self._build(PreprocessorRegistry, config.preprocessor)
        self._clusterer: BaseClusterer = self._build(ClustererRegistry, config.clusterer)
        self._grouping: BaseGroupingStrategy = self._build(GroupingRegistry, config.grouping)

    @staticmethod
    def _build(registry: _Registry, spec: Dict[str, Any]):
        cls = registry.get(spec["name"])
        return cls(**spec.get("params", {}))

    @classmethod
    def from_config(cls, schema: DataSchema, config_dict: Dict[str, Any]) -> "ClassificationPipeline":
        cfg = PipelineConfig(
            preprocessor=config_dict.get("preprocessor", {"name": "standard_scale", "params": {}}),
            encoder=config_dict.get("encoder", {"name": "one_hot", "params": {"drop": None}}),
            clusterer=config_dict.get("clusterer", {"name": "kmeans", "params": {"n_clusters": 5, "random_state": 0}}),
            grouping=config_dict.get("grouping", {"name": "within_groups", "params": {}})
        )
        return cls(schema=schema, config=cfg)

    def pre_run(self, df: pd.DataFrame):
        """ optionally calculate info such as optimal number of clusters """
        pass 
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # Validate schema
        schema = SchemaResolver.resolve(df, self.schema)
        # Execute grouping strategy
        result_df, meta = self._grouping.run(
            df=df,
            schema=schema,
            encoder=self._encoder,
            preprocessor=self._preprocessor,
            clusterer=self._clusterer,
        )
        meta["pipeline_meta"] = {
            "encoder_params": self._encoder.get_params(),
            "preprocessor_params": self._preprocessor.get_params(),
            "clusterer_params": self._clusterer.get_params(),
            "grouping": getattr(self._grouping, "name", type(self._grouping).__name__),
        }
        return result_df, meta


# -----------------------------
# 8) Example usage
# -----------------------------
if __name__ == "__main__":
    # Example DataFrame with IDs, groups, categorical and numeric features
    df = pd.DataFrame({
        "synapse_id": np.arange(10),
        "genotype": ["WT"]*5 + ["KO"]*5,
        "treatment": ["drugA"]*3 + ["vehicle"]*2 + ["drugA"]*2 + ["vehicle"]*3,
        "sex": ["male", "female", "male", "female", "male", "female", "female", "male", "female", "male"],
        "intensity": np.random.rand(10)*100,
        "area": np.random.rand(10)*2.0,
        "is_puncta": [True, False, True, False, True, True, False, False, True, False],
    })

    schema = DataSchema(
        id_cols=["synapse_id"],
        group_cols=["genotype"],            # <- cluster per genotype (WT vs KO)
        cat_cols=["treatment", "sex"],      # <- encode these
        num_cols=["intensity", "area"]      # <- numeric features
    )

    config = {
        "encoder": {"name": "one_hot", "params": {"drop": None}},
        "preprocessor": {"name": "standard_scale", "params": {}},
        "clusterer": {"name": "kmeans", "params": {"n_clusters": 3, "random_state": 0}},
        "grouping": {"name": "within_groups", "params": {}},
    }

    pipeline = ClassificationPipeline.from_config(schema, config)
    result, meta = pipeline.run(df)
    print(result.head())
    print(meta)


    # 2) Set up interpretation
    ############################
    interp_cfg = {
        "profiler": {"name": "z_profile", "params": {"ddof": 0}},
        "selector": {"name": "topn_by_absz", "params": {"top_n": 8}},
        "univariate": {"name": "anova", "params": {"fdr_method": "fdr_bh"}},
        "importance": {"name": "rf_permutation", "params": {"n_estimators": 400, "n_repeats": 5, "n_jobs": 1}},
        "grouping": {"name": "global"},  # or {"name": "within_groups"} to match your clustering scheme
        # Optional: only if you have fitted objects and want centroid back-projection:
        # "backprojector": {"name": "pca_backproject"},
        # "extra_feature_cols": ["treatment__drugA", "sex__female"]  # if you've kept encoded cols
    }

    # Optional artifacts if you want back-projection (supply if you have them)
    artifacts = {
        # "kmeans": fitted_kmeans,
        # "scaler": fitted_scaler,
        # "pca": fitted_pca
        "encoder": pipeline._encoder,
        "preprocessor": pipeline._preprocessor,
        "clusterer": pipeline._clusterer
    }

    # schema = DataSchema(
    #     id_cols=["synapse_id"],
    #     group_cols=["genotype"],         # should match your clustering grouping strategy if using within_groups
    #     cat_cols=["treatment", "sex"],   # only used if you pass encoded cols in extra_feature_cols
    #     num_cols=["intensity", "area"]
    # )

    interpreter = ClusterInterpretation.from_config(schema, interp_cfg, artifacts=artifacts)
    _, meta_interp = interpreter.run(result, labels_col="cluster_label")

    # Outputs:
    # meta_interp["profiles"]["z_profiles"] (global) or meta_interp["groups"][key]["profiles"]["z_profiles"] (per-group)
    # meta_interp["top_drivers"], meta_interp["univariate"], meta_interp["importance"]
    # meta_interp["backprojection"] (if artifacts provided)