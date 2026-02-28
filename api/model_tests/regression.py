from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from utils import make_preprocessor
from checks import check_linearity
from sklearn.metrics import r2_score

def test_regression_algorithms(target: str, df: DataFrame, numericals: list, categoricals: list, ordinals: list):
    num_cols = len(df.columns)
    num_rows = len(df)
    is_linear = check_linearity(df, target)
    
    preprocessor = make_preprocessor(numericals, categoricals, ordinals)
    
    X = df.drop(columns=target)
    y = df[target]
    
    X_transformed = preprocessor.fit_transform(X)
    
    # Clearly linear and few columns -> LinearRegression
    if is_linear and num_cols <= 10:
        model, accuracy = train_linear_model(X_transformed, y)
        return model, accuracy

    # Not linear and few columns -> PolynomialRegression
    elif num_cols <= 10 and not is_linear:
        model, accuracy = train_polynomial_model(X_transformed, y)
        return model, accuracy

    # Much rows and much columns -> RandomForestRegressor
    elif num_rows > 1000 and num_cols > 10:
        pass
        # model = train_random_forest_model(X_transformed, y)

    # Fallback -> GradientBoostingRegressor
    else:
        pass
        # model = train_gradient_boosting_model(X_transformed, y)

def train_linear_model(X, y):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51, shuffle=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = (abs(y_pred - y_test) / y_test) * 100
    
    return model, accuracy
            
def train_polynomial_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51, shuffle=True)
    polynomial_degrees = [n for n in range(1, 11)]
    best_r2 = -float("inf")
    best_degree = 1
    for degree in polynomial_degrees:
        poly_feat = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly, X_test_poly = poly_feat.fit_transform(X_train), poly_feat.fit_transform(X_test)
        model = LinearRegression()
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
    model = Pipeline(steps=[
        ("poly_feat", poly_feat),
        ("regressor", LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = (abs(y_pred - y_test) / y_test) * 100
    return model, accuracy