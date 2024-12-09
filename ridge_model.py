from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


def train_ridge_model(X_train, y_train, X_test):
    # Pipeline z Ridge Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Skalowanie danych
        ('poly', PolynomialFeatures(degree=7, include_bias=False)),  # Wielomiany
        ('ridge', Ridge())  # Ridge Regression
    ])

    # Siatka parametrów do przeszukiwania
    param_grid = {
        'ridge__alpha': [0.2, 0.3, 0.35, 0.5, 1, 10, 100, 1000],  # Optymalizacja alpha
        'poly__degree': [1, 2, 3, 5]  # Stopień wielomianu
    }

    # GridSearchCV do optymalizacji hiperparametrów
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Wybór najlepszego modelu
    best_model = grid_search.best_estimator_

    # Predykcje
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    return best_model, train_pred, test_pred
