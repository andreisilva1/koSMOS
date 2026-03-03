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
    compressed_df: DataFrame = None,
):
    num_cols = len(df.columns)
    num_rows = len(df)
    is_linear = check_linearity(df, target)
    X = (
        compressed_df.copy()
        if len(compressed_df) is not None and len(compressed_df) > 0
        else df.copy()
    )
    categoricals = X.select_dtypes(include=["object"]).columns
    preprocessor = make_preprocessor(numericals=numericals, ordinals=ordinals)
    numericals_correlation = [*numericals, target]
    preprocessor_correlations = make_preprocessor(
        numericals_correlation, categoricals, ordinals
    )
    X_corr_transformed = preprocessor_correlations.fit_transform(X)
    X.drop(columns=target, inplace=True)
    X_transformed = preprocessor.fit_transform(X)
    y = (
        compressed_df[target]
        if len(compressed_df) is not None and len(compressed_df) > 0
        else df[target]
    )

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
        model, accuracy = train_linear_model(X_transformed, y)

    # Not linear and few columns -> PolynomialRegression
    elif num_cols * 3 <= 30 and not is_linear:
        model, accuracy = train_polynomial_model(X_transformed, y)

    # Much rows and much columns -> RandomForestRegressor
    elif num_rows > 1000 and num_cols > 10:
        model, accuracy = train_random_forest_regression_model(X_transformed, y)

    # Fallback -> GradientBoostingRegressor
    else:
        model, accuracy = train_gradient_boosting_regression_model(X_transformed, y)

    hiperparameter_df = DataFrame([[f"{accuracy:.2f}"]], columns=["accuracy"])
    return (
        model,
        preprocessor,
        hiperparameter_df,
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
    return model, accuracy


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

    print(best_degree)
    print(best_r2)
    poly_feat = PolynomialFeatures(degree=best_degree)
    model = Pipeline(
        steps=[("poly_feat", poly_feat), ("regressor", LinearRegression())]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = return_accuracy_regression(y_pred, y_test)
    return model, accuracy


def train_random_forest_regression_model(X_transformed, y):
    # Generic RandomForestRegression
    model = RandomForestRegressor(
        n_estimators=200, max_features="sqrt", random_state=51, n_jobs=-1
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = return_accuracy_regression(y_pred, y_test)
    return model, accuracy


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
    return model, accuracy
