import numpy as np
import pandas as pd
from src.features import parse_year, safe_json, build_features


def _synthetic_frame(with_revenue=True):
    rows = {
        "id": [1, 2, 3],
        "budget": [1000000, 0, 5000000],
        "popularity": [10.0, 2.5, 50.0],
        "runtime": [120.0, 90.0, np.nan],
        "release_date": ["3/15/05", "12/1/99", "7/4/18"],
        "homepage": ["http://x.com", np.nan, np.nan],
        "belongs_to_collection": [
            "[{'id': 1, 'name': 'C'}]", np.nan, np.nan],
        "genres": [
            "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Drama'}]",
            "[{'id': 2, 'name': 'Drama'}]",
            np.nan],
        "cast": ["[{'id': 1, 'name': 'A', 'gender': 2}]", "[]", np.nan],
        "crew": ["[{'id': 1, 'name': 'B', 'gender': 1}]", np.nan, np.nan],
        "production_companies": ["[{'id': 1, 'name': 'Co'}]", np.nan, np.nan],
        "production_countries": ["[{'iso_3166_1': 'US', 'name': 'USA'}]", np.nan, np.nan],
        "spoken_languages": ["[{'iso_639_1': 'en', 'name': 'English'}]", np.nan, np.nan],
        "Keywords": ["[{'id': 1, 'name': 'k'}]", np.nan, np.nan],
        "original_language": ["en", "fr", "en"],
        "tagline": ["A tagline", np.nan, "Hi"],
        "title": ["Movie One", "Deux", "Three"],
        "overview": ["An overview here", np.nan, "Short"],
    }
    if with_revenue:
        rows["revenue"] = [10000000, 500000, 80000000]
    return pd.DataFrame(rows)


def test_parse_year_two_digit_fix():
    assert parse_year(5) == 2005
    assert parse_year(99) == 1999
    assert parse_year(18) == 2018
    assert parse_year(2015) == 2015


def test_safe_json_handles_nan_and_bad():
    assert safe_json(np.nan) == []
    assert safe_json("not json") == []
    assert safe_json("[{'name': 'x'}]") == [{"name": "x"}]


def test_build_features_shapes_and_no_nan():
    train = _synthetic_frame(with_revenue=True)
    test = _synthetic_frame(with_revenue=False)
    X_train, y_train, X_test, feature_names = build_features(train, test)
    assert np.allclose(y_train.values, np.log1p(train["revenue"].values))
    assert list(X_train.columns) == list(X_test.columns) == list(feature_names)
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert np.isfinite(X_train.to_numpy()).all()
    for col in ["num_genres", "num_cast", "num_crew", "budget_is_zero",
                "has_homepage", "has_collection", "release_year"]:
        assert col in X_train.columns
    assert X_train["budget_is_zero"].tolist() == [0, 1, 0]
    assert X_train["num_genres"].tolist() == [2, 1, 0]
