from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from model_tests.optuna import optuna_test
from utils.dataframes import apply_pca, make_preprocessor
from checks.statistics import check_collinearity


def test_clustering_algorithms(
    cluster_method: str,
    df: DataFrame,
    numericals: list = [],
    ordinals: list = [],
    n_groups: int = None,
):
    # Normalization
    X = df.copy()
    preprocessor = make_preprocessor(numericals, ordinals)

    # Start KNN to predict the cluster of new items
    knn = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    df_transformed = pd.DataFrame(
        X_transformed, columns=preprocessor.get_feature_names_out()
    )
    collinear = check_collinearity(df_transformed)

    # Cluster methods
    n_components = None
    df_high_corr = pd.DataFrame()
    df_all_correlations = pd.DataFrame()

    if (
        collinear
    ):  # If strong collinearity, use PCA -> convert to n_components -> choose the best model by "brute force" or Optuna
        df_all_correlations, df_high_corr, n_components, X_pca = apply_pca(
            X_transformed, df_transformed
        )  # Return the dataframe with only PCA columns, to calculate the correlation after.
        X_transformed = X_pca

    if cluster_method == "k-means":
        if (
            n_components and n_components * n_groups > 50
        ):  # Apparently is a good complexity scenario to start use optuna.
            best_n_clusters, best_max_iter, best_n_init = optuna_test(
                algorithm="k-means", X_transformed=X_transformed, n_groups=n_groups
            )
            best_model = KMeans(
                n_clusters=best_n_clusters,
                max_iter=best_max_iter,
                n_init=best_n_init,
            )

            best_model.fit(X_transformed)
            df["cluster"] = best_model.labels_

        # If complexity is not high enough, being collinear or not, just use some "brute force" to decide the best model
        else:
            best_n_clusters = n_groups
            best_model = None
            best_score = -1
            best_n_init = None
            best_max_iter = None

            for n_init in [10, 20, 30]:
                for max_iter in [200, 500]:
                    model = KMeans(
                        n_clusters=n_groups, max_iter=max_iter, n_init=n_init
                    )
                    labels = model.fit_predict(X_transformed)

                    score = silhouette_score(X_transformed, labels)

                    if score > best_score:
                        best_max_iter = max_iter
                        best_n_init = n_init
                        best_score = score
                        best_model = model
            best_model.fit(X_transformed)
        hiperparameter_df = DataFrame(
            [["kmeans", best_n_init, best_max_iter, best_n_clusters]],
            columns=["model_type", "n_init", "max_iter", "n_clusters"],
        )

    if cluster_method == "hierarchical":

        best_model, linkage, hierarchical_n_clusters, divisive_n_clusters = optuna_test(
            algorithm="hierarchical", X_transformed=X_transformed, num_rows=len(df)
        )

        if isinstance(best_model, AgglomerativeClustering):
            hiperparameter_df = DataFrame(
                [["agglomerative_clustering", linkage, hierarchical_n_clusters]],
                columns=["model_type", "linkage", "n_clusters"],
            )
        if isinstance(best_model, BisectingKMeans):
            hiperparameter_df = DataFrame(
                [["divisive_clustering", divisive_n_clusters]],
                columns=["model_type", "n_clusters"],
            )

    df_knn = df.copy()
    df_knn["cluster"] = best_model.labels_
    y = df_knn["cluster"]
    knn.fit(X, y)
    return (
        hiperparameter_df,
        knn,
        df_high_corr.to_csv(index=False),
        df_all_correlations.to_csv(index=False),
        best_model,
        preprocessor,
    )
