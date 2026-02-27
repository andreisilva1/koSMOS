from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd

from utils import extract_correlation_pairs

def test_clustering_algorithms(cluster_method: str, df: DataFrame, dict_values: dict, numericals: list, categoricals: list, ordinals: list, n_groups: int = None):    
    # Normalization
    list_transformers = []
    X = df.copy()
    if numericals:
        list_transformers.append(("num", StandardScaler(), numericals))
    if categoricals:
        list_transformers.append(("cat", OneHotEncoder(), categoricals))
    if ordinals:
        list_transformers.append(("ord", OrdinalEncoder(), ordinals))
    if list_transformers:
        preprocessor = ColumnTransformer(transformers=list_transformers, remainder="passthrough")
        
    X_transformed = preprocessor.fit_transform(X)
    df_transformed = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
    
    # Cluster methods
    if cluster_method == "k-means":
        model_cluster = KMeans(n_clusters=n_groups)
        model_cluster.fit(X_transformed)
        df["group"] = model_cluster.labels_
        corr_pairs = extract_correlation_pairs(df_transformed)
        
        # Return the correlations where the absolute value >= 0.6 (the most "evident" ones)
        df_corr = pd.DataFrame(corr_pairs)
        df_corr_filtered =  df_corr[abs(df_corr["correlation"]) >= 0.6]      
        return {df.to_csv(index=False), df_corr_filtered.to_csv(index=False)}