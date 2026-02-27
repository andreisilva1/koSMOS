from pandas import DataFrame
from sklearn.cluster import KMeans

def test_clustering_algorithms(cluster_method: str, df: DataFrame, dict_values: dict, numericals: list, categoricals: list, ordinals: list, n_groups: int = None):
    converted_df = df.convert_dtypes()
    data = converted_df.to_dict(orient="records")
    
    if cluster_method == "k-means":
        model_cluster = KMeans(n_clusters=n_groups)