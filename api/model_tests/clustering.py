from math import ceil

import optuna
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans

from checks import check_collinearity
from utils import apply_pca, make_preprocessor

def test_clustering_algorithms(cluster_method: str, df: DataFrame, numericals: list, categoricals: list, ordinals: list, n_groups: int = None):    
    # Normalization
    X = df.copy()
    preprocessor = make_preprocessor(numericals, categoricals, ordinals)
        
    X_transformed = preprocessor.fit_transform(X)
    df_transformed = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
    collinear = check_collinearity(df_transformed)
    
    # Cluster methods
    n_components = None
    df_high_corr = pd.DataFrame()
    df_corr = pd.DataFrame()
    
    if collinear: # If strong collinearity, use PCA -> convert to n_components -> choose the best model by "brute force" or Optuna
        df_corr, df_high_corr, n_components, X_pca = apply_pca(X_transformed, df_transformed) # Return the dataframe with only PCA columns, to calculate the correlation after.
        X_transformed = X_pca 
        
    if cluster_method == "k-means": 
        if n_components and n_components * n_groups > 50: # Apparently is a good complexity scenario to start use optuna.
            def kmeans_optuna(trial):
                n_clusters = trial.suggest_int("n_clusters", 2, n_groups)
                max_iter = trial.suggest_int("max_iter", 200, 500)
                n_init = trial.suggest_int("n_init", 10, 30)
                model_kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init)
                
                # if it is collinear, it will use X_pca, which overwrites X_transformed; if it is not collinear, it uses the normal X_transformed.
                labels = model_kmeans.fit_predict(X_transformed)
                sillhouette_avg = silhouette_score(X_transformed, labels=labels)
                
                return sillhouette_avg
            
            kmeans_study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space={"n_clusters": [n for n in range(2, n_groups+1)], "max_iter": [n for n in range(200, 501)], "n_init": [n for n in range(10, 31)]}), direction="maximize")
            kmeans_study.optimize(kmeans_optuna, n_trials=20)
            bp = kmeans_study.best_params
            best_model = KMeans(n_clusters=bp["n_clusters"], max_iter=bp["max_iter"], n_init=bp["n_init"])
            best_model.fit(X_transformed)
            df['groups'] = best_model.labels_
        
        # If complexity is not high enough, being collinear or not, just use some "brute force" to decide the best model
        else:
            best_model = None
            best_score = -1
            
            for n_init in [10, 20, 30]:
                for max_iter in [200, 500]:
                    model = KMeans(n_clusters=n_groups, max_iter=max_iter, n_init=n_init)
                    labels = model.fit_predict(X_transformed)
                    
                    score = silhouette_score(X_transformed, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
            best_model.fit(X_transformed)
            df["cluster"] = best_model.labels_
            

    if cluster_method == "hierarquical":
        def hierarquical_agg_optuna(trial):
            n_clusters = trial.suggest_int("n_clusters", 2, ceil(0.5 * len(df))) # 2 cluster min, half the df size max (so, in the worst case, we will have x clusters with 2 items each
            linkage = trial.suggest_categorical("linkage", ["ward", "single", "complete", "average"])
            
            model_hierarquical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            
            # if it is collinear, it will use X_pca, which overwrites X_transformed; if it is not collinear, it uses the normal X_transformed.
            labels = model_hierarquical.fit_predict(X_transformed)
            sillhouette_avg = silhouette_score(X_transformed, labels=labels)
            
            return sillhouette_avg
        
        hierarquical_agg_study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space={"n_clusters": [n for n in range(2, ceil(0.5 * len(df)) + 1)], "linkage": ["ward", "single", "complete", "average"]}), direction="maximize")
        hierarquical_agg_study.optimize(hierarquical_agg_optuna, n_trials=20)
        bp_agg = hierarquical_agg_study.best_params

    
        def hierarquical_div_optuna(trial):
            n_clusters = trial.suggest_int("n_clusters", 2, ceil(0.5 * len(df)))
            
            hieraquical_model = BisectingKMeans(n_clusters=n_clusters)
            
            labels = hieraquical_model.fit_predict(X_transformed)
            
            silhouette_avg = silhouette_score(X_transformed, labels=labels)
            
            return silhouette_avg
        
        search_space ={"n_clusters": [n for n in range(1, 151)]}
        hierarquical_div_study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space=search_space), direction="maximize")
        hierarquical_div_study.optimize(hierarquical_div_optuna, n_trials=20)
        
        bp_div = hierarquical_div_study.best_params
        
        # Checks which hierarquical model has the best sillhouette_score
        agg_model = AgglomerativeClustering(n_clusters=bp_agg["n_clusters"], linkage=bp_agg["linkage"])
        agg_labels = agg_model.fit_predict(X_transformed)
        
        div_model = BisectingKMeans(n_clusters=bp_div["n_clusters"])
        div_labels = div_model.fit_predict(X_transformed)
        
        agg_best_sillhouette = silhouette_score(X_transformed, labels=agg_labels)
        div_best_sillhouette = silhouette_score(X_transformed, labels=div_labels)
        best_model = agg_model if agg_best_sillhouette >= div_best_sillhouette else div_model
        best_model_labels = agg_labels if best_model == agg_model else div_labels
        
        df["cluster"] = best_model_labels
        
    # Final DF (with group column), (if collinear) -> DF with only correlations > 0.6 and all correlations
    return df.to_csv(index=False), df_high_corr.to_csv(index=False), df_corr.to_csv(index=False), best_model, preprocessor