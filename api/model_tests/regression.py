from pandas import DataFrame
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from utils.dataframes import make_preprocessor, return_accuracy_regression
from utils.extractors import extract_correlation_pairs
from checks.statistics import check_linearity
from sklearn.metrics import r2_score


def test_regression_algorithms(
    target: str,
    df: DataFrame,
    numericals: list = [],
    categoricals: list = [],
    ordinals: list = [],
):
    num_cols = len(df.columns)
    num_rows = len(df)
    is_linear = check_linearity(df, target)
    X = df.copy()

    preprocessor = make_preprocessor(
        numericals=numericals, categoricals=categoricals, ordinals=ordinals
    )
    numericals_correlation = [*numericals, target]
    preprocessor_correlations = make_preprocessor(
        numericals_correlation, categoricals, ordinals
    )
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

    # Clearly linear and few columns -> LinearRegression
    if is_linear and num_cols <= 10:
        model, stats_df = train_linear_model(X_transformed, y)

    # Not linear and few columns -> PolynomialRegression
    elif num_cols * 3 <= 30 and not is_linear:
        model, stats_df = train_polynomial_model(X_transformed, y)

    # Much rows and much columns -> RandomForestRegressor
    elif num_rows > 1000 and num_cols > 10:
        model, stats_df = train_random_forest_regression_model(X_transformed, y)

    # Fallback -> GradientBoostingRegressor
    else:
        model, stats_df = train_gradient_boosting_regression_model(X_transformed, y)

    return (
        model,
        preprocessor,
        stats_df,
        df_high_correlations.to_csv(index=False),
        df_all_correlations.to_csv(index=False),
    )


def train_linear_model(X_transformed, y):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51, shuffle=True
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = return_accuracy_regression(y_pred, y_test)

    stats_df = DataFrame(
        [["linear_model", f"{accuracy:.2f}"]], columns=["model_type", "accuracy"]
    )

    return model, stats_df


def train_polynomial_model(X_transformed, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51, shuffle=True
    )
    polynomial_degrees = [1, 2, 3]
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

    accuracy = return_accuracy_regression(y_pred, y_test)
    stats_df = DataFrame(
        [["polynomial_model", f"{accuracy:.2f}", best_degree]],
        columns=["model_type", "accuracy", "degree"],
    )

    return model, stats_df


def train_random_forest_regression_model(X_transformed, y):
    # Generic RandomForestRegression
    model = RandomForestRegressor(
        n_estimators=200, max_features="sqrt", random_state=51
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = return_accuracy_regression(y_pred, y_test)

    stats_df = DataFrame(
        [["random_forest_regression", f"{accuracy:.2f}"]],
        columns=["model_type", "accuracy"],
    )

    return model, stats_df


def train_gradient_boosting_regression_model(X_transformed, y):
    # Generic GradientBoostingRegression Model
    model = HistGradientBoostingRegressor(
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
    accuracy = return_accuracy_regression(y_pred, y_test)

    stats_df = DataFrame(
        [["gradient_boosting_regression", f"{accuracy:.2f}"]],
        columns=["model_type", "accuracy"],
    )

    return model, stats_df
