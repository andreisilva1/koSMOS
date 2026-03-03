from math import ceil

import optuna
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans, KMeans
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    f1_score,
    log_loss,
    recall_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score


def optuna_test(
    algorithm,
    X_transformed,
    y=None,
    n_groups: int = None,
    num_cols: int = None,
    num_rows: int = None,
    classifier: bool = False,
    n_trials: int = 20,
):
    if y:
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.3, random_state=51, shuffle=True
        )

    if algorithm == "logistic":

        def logistic_optuna(trial):
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
            c_values = trial.suggest_categorical("c_values", [100, 10, 1, 0.1, 0.01])

            logistic_model = LogisticRegression(penalty=penalty, C=c_values)
            logistic_model.fit(X_train, y_train)

            y_decision_optuna = logistic_model.decision_function(X_test)

            y_pred = logistic_model.predict(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_decision_optuna)

            roc_auc_optuna = auc(fpr, tpr)
            log_loss_optuna = log_loss(y_test, y_pred)
            f1_score_optuna = f1_score(y_test, y_pred)

            return roc_auc_optuna, log_loss_optuna, f1_score_optuna

        search_space = {"penalty": ["l1", "l2"], "c_values": [100, 10, 1, 0.1, 0.01]}
        logistic_study = optuna.create_study(
            sampler=optuna.samplers.GridSampler(search_space=search_space),
            directions=["maximize", "minimize", "maximize"],
        )

        logistic_study.optimize(logistic_optuna, n_trials=n_trials)
        best_params = max(logistic_study.best_trials, key=lambda t: t.values[1]).params
        penalty, c_values = best_params["penalty"], best_params["c_values"]
        return penalty, c_values

    if algorithm == "naive":

        def naive_bayes_optuna(trial):
            k = trial.suggest_int("k", 1, num_cols + 1)

            kbest = SelectKBest(score_func=f_classif, k=k)
            naive_model = GaussianNB()

            X_train_best_features = kbest.fit_transform(X_train, y_train)
            naive_model.fit(X_train_best_features, y_train)
            X_test_best_features = kbest.transform(X_test)
            y_pred = naive_model.predict(X_test_best_features)

            recall = recall_score(y_test, y_pred, average="macro")

            return recall

        naive_bayes_seach_space = {"k": (1, num_cols)}
        naive_bayes_study = optuna.create_study(
            sampler=optuna.samplers.GridSampler(search_space=naive_bayes_seach_space),
            directions=["maximize"],
        )
        naive_bayes_study.optimize(naive_bayes_optuna)
        k = naive_bayes_study.best_params["k"]
        return k

    if algorithm == "decision_tree":

        def decisiontree_optuna(trial):
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20)
            max_depth = trial.suggest_int("max_depth", 2, 8)

            decision_tree_model = DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf, max_depth=max_depth
            )

            cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=51)

            scores = cross_val_score(
                decision_tree_model, X_transformed, y, cv=cv_folds, scoring="accuracy"
            )

            return scores.mean()

        decision_tree_study = optuna.create_study(direction="maximize")
        decision_tree_study.optimize(decisiontree_optuna, n_trials=n_trials)
        min_samples_leaf, max_depth = (
            decision_tree_study.best_params["min_samples_leaf"],
            decision_tree_study["max_depth"],
        )
        return min_samples_leaf, max_depth

    if algorithm == "random_forest":

        def random_forest_optuna(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 3, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            )
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])

            if classifier:
                random_forest_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=bootstrap,
                )
            else:
                random_forest_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=bootstrap,
                )

            random_forest_model.fit(X_train, y_train)
            y_pred = random_forest_model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

            return r2

        random_forest_search_space = {
            "n_estimators": (50, 500),
            "max_depth": (3, 30),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 20),
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }

        random_forest_study = optuna.create_study(
            sampler=optuna.samplers.GridSampler(
                search_space=random_forest_search_space
            ),
            direction="maximize",
        )
        random_forest_study.optimize(random_forest_optuna, n_trials=n_trials)

        (
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
        ) = (
            random_forest_study.best_params["n_estimators"],
            random_forest_study.best_params["max_depth"],
            random_forest_study.best_params["min_samples_split"],
            random_forest_study.best_params["min_samples_leaf"],
            random_forest_study.best_params["max_features"],
            random_forest_study.best_params["bootstrap"],
        )

        return (
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
        )

    if algorithm == "gradient":

        def gradient_boosting_optuna(trial):
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            max_iter = trial.suggest_int("max_iter", 50, 500)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 100)
            l2_regularization = trial.suggest_float("l2_regularization", 0, 1.0)
            max_bins = trial.suggest_int("max_bins", 100, 255)

            if classifier:
                hist_gradient_model = HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    max_bins=max_bins,
                )
            else:
                hist_gradient_model = HistGradientBoostingRegressor(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    max_bins=max_bins,
                )

            hist_gradient_model.fit(X_train, y_train)
            y_pred = hist_gradient_model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            return r2

        hist_gb_search_space = {
            "learning_rate": (0.01, 0.3),
            "max_iter": (50, 500),
            "max_depth": (3, 20),
            "min_samples_leaf": (1, 20),
            "max_leaf_nodes": (10, 100),
            "l2_regularization": [0.0, 0.01, 0.1, 1.0],
            "max_bins": (100, 255),
        }

        gradient_boosting_study = optuna.create_study(
            sampler=optuna.samplers.GridSampler(search_space=hist_gb_search_space),
            direction="maximize",
        )
        gradient_boosting_study.optimize(gradient_boosting_optuna, n_trials=n_trials)

        (
            learning_rate,
            max_iter,
            max_depth,
            min_samples_leaf,
            max_leaf_nodes,
            l2_regularization,
            max_bins,
        ) = (
            gradient_boosting_study.best_params["learning_rate"],
            gradient_boosting_study.best_params["max_iter"],
            gradient_boosting_study.best_params["max_depth"],
            gradient_boosting_study.best_params["min_samples_leaf"],
            gradient_boosting_study.best_params["max_leaf_nodes"],
            gradient_boosting_study.best_params["l2_regularization"],
            gradient_boosting_study.best_params["max_bins"],
        )

        return (
            learning_rate,
            max_iter,
            max_depth,
            min_samples_leaf,
            max_leaf_nodes,
            l2_regularization,
            max_bins,
        )

    if algorithm == "kmeans":

        def kmeans_optuna(trial):
            n_clusters = trial.suggest_int("n_clusters", 2, n_groups)
            max_iter = trial.suggest_int("max_iter", 200, 500)
            n_init = trial.suggest_int("n_init", 10, 30)
            model_kmeans = KMeans(
                n_clusters=n_clusters, max_iter=max_iter, n_init=n_init
            )

            # if it is collinear, it will use X_pca, which overwrites X_transformed; if it is not collinear, it uses the normal X_transformed.
            labels = model_kmeans.fit_predict(X_transformed)
            sillhouette_avg = silhouette_score(X_transformed, labels=labels)

            return sillhouette_avg

        kmeans_study = optuna.create_study(
            sampler=optuna.samplers.GridSampler(
                search_space={
                    "n_clusters": (2, n_groups),
                    "max_iter": (200, 500),
                    "n_init": (10, 30),
                }
            ),
            direction="maximize",
        )
        kmeans_study.optimize(kmeans_optuna, n_trials=20)
        kmeans_best_params = kmeans_study.best_params

        return (
            kmeans_best_params["n_clusters"],
            kmeans_best_params["max_iter"],
            kmeans_best_params["n_init"],
        )

    if algorithm == "hierarchical":

        def hierarchical_agg_optuna(trial):
            n_clusters = trial.suggest_int(
                "n_clusters",
                2,
                ceil(0.1 * num_rows) if ceil(0.1 * num_rows) < 200 else 200,
            )  # 2 cluster min, 200 clusters max
            linkage = trial.suggest_categorical(
                "linkage", ["ward", "single", "complete", "average"]
            )

            model_hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters, linkage=linkage
            )

            # if it is collinear, it will use X_pca, which overwrites X_transformed; if it is not collinear, it uses the normal X_transformed.
            labels = model_hierarchical.fit_predict(X_transformed)
            sillhouette_avg = silhouette_score(X_transformed, labels=labels)

            return sillhouette_avg

        hierarchical_agg_study = optuna.create_study(
            sampler=optuna.samplers.GridSampler(
                search_space={
                    "n_clusters": (
                        2,
                        (ceil(0.1 * num_rows) if ceil(0.1 * num_rows) < 200 else 200),
                    ),
                    "linkage": ["ward", "single", "complete", "average"],
                }
            ),
            direction="maximize",
        )
        hierarchical_agg_study.optimize(hierarchical_agg_optuna, n_trials=20)
        best_params_agglomerative_clustering = hierarchical_agg_study.best_params
        linkage, hierarchical_n_clusters = (
            best_params_agglomerative_clustering["linkage"],
            best_params_agglomerative_clustering["n_clusters"],
        )

        def hierarchical_div_optuna(trial):
            n_clusters = trial.suggest_int("n_clusters", 2, ceil(0.5 * num_rows))

            hieraquical_model = BisectingKMeans(n_clusters=n_clusters)

            labels = hieraquical_model.fit_predict(X_transformed)

            silhouette_avg = silhouette_score(X_transformed, labels=labels)

            return silhouette_avg

        search_space = {"n_clusters": [n for n in range(1, 151)]}
        hierarchical_div_study = optuna.create_study(
            sampler=optuna.samplers.GridSampler(search_space=search_space),
            direction="maximize",
        )
        hierarchical_div_study.optimize(hierarchical_div_optuna, n_trials=20)

        best_params_divisive_clustering = hierarchical_div_study.best_params
        divisive_n_clusters = best_params_divisive_clustering["n_clusters"]

        # Checks which hierarchical model has the best sillhouette_score
        agg_model = AgglomerativeClustering(
            n_clusters=best_params_agglomerative_clustering["n_clusters"],
            linkage=best_params_agglomerative_clustering["linkage"],
        )
        agg_labels = agg_model.fit_predict(X_transformed)

        div_model = BisectingKMeans(
            n_clusters=best_params_divisive_clustering["n_clusters"]
        )
        div_labels = div_model.fit_predict(X_transformed)

        agg_best_sillhouette = silhouette_score(X_transformed, labels=agg_labels)
        div_best_sillhouette = silhouette_score(X_transformed, labels=div_labels)
        best_model = (
            agg_model if agg_best_sillhouette >= div_best_sillhouette else div_model
        )
        return best_model, linkage, hierarchical_n_clusters, divisive_n_clusters
