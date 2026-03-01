import numpy as np
from pandas import DataFrame
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from utils.dataframes import make_preprocessor
from utils.extractors import extract_correlation_pairs
from checks.statistics import check_independence, check_linearity
from sklearn.metrics import r2_score, accuracy_score


# Treinar modelos de LogisticRegression, NaiveBayes e DecisionTree
def test_classification_algorithms(
    target: str, df: DataFrame, numericals: list, categoricals: list, ordinals: list
):
    num_cols = len(df.columns)
    num_rows = len(df)
    is_linear = check_linearity(df, target)
    preprocessor = make_preprocessor(numericals, categoricals, ordinals)
    numericals_correlation = [*numericals, target]
    preprocessor_correlations = make_preprocessor(
        numericals_correlation, categoricals, ordinals
    )
    X = df.copy()
    X_corr_transformed = preprocessor_correlations.fit_transform(X)
    X.drop(columns=target, inplace=True)
    X_transformed = preprocessor.fit_transform(X)
    y = df[target]

    df_transformed = DataFrame(
        X_corr_transformed, columns=preprocessor_correlations.get_feature_names_out()
    )
    corr_pairs = extract_correlation_pairs(df_transformed)
    df_all_correlations = DataFrame(corr_pairs)
    df_high_correlations = df_all_correlations[
        abs(df_all_correlations["correlation"]) >= 0.6
    ]

    len_target = len(df[target].nunique())
    # Few classes and is linear -> LogisticRegression
    if is_linear:
        if len_target < 10:
            model, accuracy = train_logistic_model(X_transformed, y)

    else:
        # Features are approximately independents (or complexe relations)...
        if check_independence(df, target):
            # and multiple classes -> NaiveBayes
            if len_target > 1:
                model, accuracy = train_naive_model(X_transformed, y)

        # not independents and a short dataset -> DecisionTree
        elif num_rows < 1000 and num_cols < 10:
            model = train_decision_tree_model(X_transformed, y)

    if not model:
        # If no model until here and multiple classes -> RandomForestClassifier
        if len_target > 1:
            model = train_random_forest_classifier_model(X_transformed, y)

    # Fallback -> GradientBoostingClassifier
    else:
        model = train_gradient_boosting_classifier_model(X_transformed, y)
    return (
        model,
        preprocessor,
        accuracy,
        df_high_correlations.to_csv(index=False),
        df_all_correlations.to_csv(index=False),
    )


def train_logistic_model(X_transformed, y):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51, shuffle=True
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test)
    return model, accuracy


def train_naive_model(X_transformed, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51, shuffle=True
    )
    polynomial_degrees = [n for n in range(1, 11)]
    best_r2 = -float("inf")
    best_degree = 1
    for degree in polynomial_degrees:
        poly_feat = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly, X_test_poly = poly_feat.fit_transform(
            X_train
        ), poly_feat.fit_transform(X_test)

        model = Pipeline(
            steps=[("poly_feat", poly_feat), ("regressor", LinearRegression())]
        )

        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)

        actual_r2 = r2_score(y_test, y_pred)

        if not best_r2:
            best_r2 = actual_r2
            best_degree = degree

        elif actual_r2 > best_r2:
            best_r2 = actual_r2
            best_degree = degree

    poly_feat = PolynomialFeatures(degree=best_degree)
    model = Pipeline(
        steps=[("poly_feat", poly_feat), ("regressor", LinearRegression())]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test)
    return model, accuracy


def train_decision_tree_model(X_transformed, y):
    # Generic GradientBoostingModel
    model = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    return model, accuracy


def train_gradient_boosting_classifier_model(X_transformed, y):
    # Generic GradientBoostingClassifier Model
    model = HistGradientBoostingClassifier(
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=51,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    return model, accuracy


def train_random_forest_classifier_model(X_transformed, y):
    # Generic RandomForestClassifier Model
    model = RandomForestClassifier(
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=51,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)

    return model, accuracy
