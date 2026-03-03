from pandas import DataFrame
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from model_tests.optuna import optuna_test
from utils.dataframes import make_preprocessor, return_accuracy_classification
from utils.extractors import extract_correlation_pairs
from checks.statistics import check_independence, check_linearity
from sklearn.metrics import accuracy_score


# Treinar modelos de LogisticRegression, NaiveBayes e DecisionTree
def test_classification_algorithms(
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
    preprocessor = make_preprocessor(numericals, ordinals)
    preprocessor_correlations = make_preprocessor(numericals, ordinals)
    X = (
        compressed_df.copy()
        if len(compressed_df) > 0 and compressed_df is not None
        else df.copy()
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

    len_target = df[target].nunique()
    # Few classes and is linear -> LogisticRegression
    if is_linear:
        if len_target < 10:
            model, hiperparameter_df = train_logistic_model(X_transformed, y)

    else:
        # Features are approximately independents (or complexe relations)...
        if check_independence(df, target):
            # and multiple classes -> NaiveBayes
            if len_target > 1:
                model, hiperparameter_df = train_naive_model(num_cols, X_transformed, y)

        # not independents and a short dataset -> DecisionTree
        elif num_rows < 1000 and num_cols < 10:
            model, hiperparameter_df = train_decision_tree_model(X_transformed, y)

    if not model:
        # If no model until here and multiple classes -> RandomForestClassifier
        if len_target > 1:
            model, hiperparameter_df = train_random_forest_classifier_model(
                X_transformed, y
            )

            # Fallback -> GradientBoostingClassifier
        else:
            model, hiperparameter_df = train_gradient_boosting_classifier_model(
                X_transformed, y
            )
    return (
        model,
        preprocessor,
        hiperparameter_df,
        df_high_correlations.to_csv(index=False),
        df_all_correlations.to_csv(index=False),
    )


def train_logistic_model(X_transformed, y):
    penalty, c_values = optuna_test("logistic", X_transformed, y)
    model = LogisticRegression(penalty=penalty, C=c_values)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51, shuffle=True
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = return_accuracy_classification(y_pred, y_test)

    logistic_hiperparameter_info_df = DataFrame(
        [["logistic_regression", accuracy, penalty, c_values]],
        columns=["model_type", "accuracy", "penalty", "c_values"],
    )

    return model, logistic_hiperparameter_info_df


def train_naive_model(num_cols, X_transformed, y):
    k = optuna_test("naive", X_transformed, y, num_cols=num_cols)

    model = GaussianNB()

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51, shuffle=True
    )

    kbest = SelectKBest(score_func=f_classif, k=k)

    X_train_best_features = kbest.fit_transform(X_train, y_train)
    model.fit(X_train_best_features, y_train)
    X_test_best_features = kbest.transform(X_test)
    y_pred = model.predict(X_test_best_features)

    accuracy = return_accuracy_classification(y_pred, y_test)

    naive_hiperparameter_info_df = DataFrame(
        [
            [
                "naive_bayes",
                accuracy,
                k,
            ]
        ],
        columns=["model_type", "accuracy", "k"],
    )
    return model, naive_hiperparameter_info_df


def train_decision_tree_model(X_transformed, y):
    # Hiperparameter tuning
    min_samples_leaf, max_depth = optuna_test("decision_tree", X_transformed, y)

    model = DecisionTreeClassifier(
        min_samples_leaf=min_samples_leaf, max_depth=max_depth
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = return_accuracy_classification(y_pred, y_test)

    decision_tree_hiperparameter_info_df = DataFrame(
        [["decision_tree", accuracy, min_samples_leaf, max_depth]],
        columns=["model_type", "accuracy", "min_samples_leaf", "max_depth"],
    )
    return model, decision_tree_hiperparameter_info_df


def train_gradient_boosting_classifier_model(X_transformed, y):
    # Hiperparameter tuning
    (
        learning_rate,
        max_iter,
        max_depth,
        min_samples_leaf,
        max_leaf_nodes,
        l2_regularization,
        max_bins,
    ) = optuna_test("gradient", X_transformed, y, classifier=True)
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        max_bins=max_bins,
        random_state=51,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = return_accuracy_classification(y_pred, y_test)

    gradient_boosting_hiperparameter_info_df = DataFrame(
        [
            [
                "gradient_boosting",
                accuracy,
                learning_rate,
                max_iter,
                max_depth,
                min_samples_leaf,
                max_leaf_nodes,
                l2_regularization,
                max_bins,
            ]
        ],
        columns=[
            "model_type",
            "accuracy",
            "learning_rate",
            "max_iter",
            "max_depth",
            "min_samples_leaf",
            "max_leaf_nodes",
            "l2_regularization",
            "max_bins",
        ],
    )
    return model, gradient_boosting_hiperparameter_info_df


def train_random_forest_classifier_model(X_transformed, y):
    # Hiperparameter tuning
    (
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
    ) = optuna_test("random_forest", X_transformed, y, classifier=True)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=51,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=51
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = return_accuracy_classification(y_pred, y_test)

    random_forest_hiperparameter_info_df = DataFrame(
        [
            [
                "random_forest",
                accuracy,
                n_estimators,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features,
                bootstrap,
            ]
        ],
        columns=[
            "model_type",
            "accuracy",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
        ],
    )
    return model, random_forest_hiperparameter_info_df
