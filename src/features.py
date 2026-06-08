import ast
import numpy as np
import pandas as pd

JSON_COLS = ["genres", "cast", "crew", "production_companies",
             "production_countries", "spoken_languages", "Keywords"]


def safe_json(s):
    """Parse a stringified list-of-dicts; return [] on NaN / malformed input."""
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def parse_year(y):
    """Map a possibly 2-digit release year to a 4-digit year."""
    y = int(y)
    if y >= 100:
        return y
    return y + 2000 if y <= 19 else y + 1900


def _count(series):
    return series.map(lambda s: len(safe_json(s)))


def _gender_count(series, gender):
    return series.map(lambda s: sum(1 for d in safe_json(s)
                                     if d.get("gender") == gender))


def _add_features(df):
    """Return a new DataFrame of engineered numeric features for df."""
    out = pd.DataFrame(index=df.index)

    # --- dates ---
    parts = df["release_date"].astype(str).str.split("/", expand=True)
    month = pd.to_numeric(parts[0], errors="coerce")
    day = pd.to_numeric(parts[1], errors="coerce")
    year = pd.to_numeric(parts[2], errors="coerce")
    out["release_month"] = month.fillna(0).astype(int)
    out["release_day"] = day.fillna(0).astype(int)
    out["release_year"] = year.fillna(0).astype(int).map(
        lambda y: parse_year(y) if y > 0 else 0)
    dt = pd.to_datetime(df["release_date"], errors="coerce")
    out["release_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
    out["release_quarter"] = dt.dt.quarter.fillna(0).astype(int)
    ref_year = out.loc[out["release_year"] > 0, "release_year"].max()
    out["years_since_release"] = np.where(
        out["release_year"] > 0, ref_year - out["release_year"], 0)

    # --- budget / popularity / runtime ---
    budget = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    runtime = pd.to_numeric(df["runtime"], errors="coerce")
    out["budget"] = budget
    out["log_budget"] = np.log1p(budget)
    out["budget_is_zero"] = (budget == 0).astype(int)
    out["popularity"] = pop
    out["runtime"] = runtime.fillna(runtime.median())
    out["budget_per_popularity"] = budget / (pop + 1)
    out["budget_per_runtime"] = budget / (out["runtime"] + 1)

    # --- JSON counts ---
    for col in JSON_COLS:
        out[f"num_{col}"] = _count(df[col]) if col in df else 0
    for g in (0, 1, 2):
        out[f"cast_gender_{g}"] = _gender_count(df["cast"], g) if "cast" in df else 0
        out[f"crew_gender_{g}"] = _gender_count(df["crew"], g) if "crew" in df else 0

    # --- flags / text ---
    out["has_homepage"] = df.get("homepage", pd.Series(index=df.index)).notna().astype(int)
    out["has_collection"] = df.get(
        "belongs_to_collection", pd.Series(index=df.index)).notna().astype(int)
    out["has_tagline"] = df.get("tagline", pd.Series(index=df.index)).notna().astype(int)
    out["is_english"] = (df.get("original_language", "") == "en").astype(int)
    out["title_len"] = df.get("title", pd.Series(index=df.index)).fillna("").str.len()
    out["overview_word_count"] = df.get(
        "overview", pd.Series(index=df.index)).fillna("").str.split().map(len)
    out["tagline_word_count"] = df.get(
        "tagline", pd.Series(index=df.index)).fillna("").str.split().map(len)

    return out


def _frequency_encode(train_lists, test_lists, min_count=10):
    """Top categories (by combined count >= min_count)."""
    from collections import Counter
    counter = Counter()
    for lst in list(train_lists) + list(test_lists):
        for d in lst:
            name = d.get("name")
            if name:
                counter[name] += 1
    keep = {name for name, c in counter.items() if c >= min_count}
    return keep


def build_features(train_df, test_df):
    """Build aligned numeric feature matrices.

    Returns (X_train, y_train, X_test, feature_names). y_train is log1p(revenue).
    """
    y_train = np.log1p(pd.to_numeric(train_df["revenue"], errors="coerce").fillna(0))
    y_train.name = "logRevenue"

    X_train = _add_features(train_df)
    X_test = _add_features(test_df)

    keep_genres = _frequency_encode(
        train_df["genres"].map(safe_json), test_df["genres"].map(safe_json), min_count=10)
    for name in sorted(keep_genres):
        col = f"genre_{name}"
        X_train[col] = train_df["genres"].map(
            lambda s: int(any(d.get("name") == name for d in safe_json(s))))
        X_test[col] = test_df["genres"].map(
            lambda s: int(any(d.get("name") == name for d in safe_json(s))))

    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    medians = pd.concat([X_train, X_test]).median(numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    feature_names = list(X_train.columns)
    return X_train, y_train, X_test, feature_names
