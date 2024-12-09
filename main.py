import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from ridge_model import train_ridge_model
from lstm_model import train_lstm_model


def load_data(file_path):
    df = pd.read_csv(file_path, sep=';', parse_dates=['date'])
    for column in ["temp_pow", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr"]:
        df[column] = df[column].str.replace(",", ".").astype(float)
    df = df.dropna(subset=["temp_pow"])
    df = df.fillna(df.backfill())
    return df


# 1. Wczytanie danych
file_path = 'Wroclaw.csv'
df = load_data(file_path)

# 2. Przygotowanie danych
criteria = ["kier_wiatr", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr", "wilg"]
X = df[criteria]
y = df["temp_pow"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Trenowanie modelu Ridge Regression
ridge_model, ridge_train_pred, ridge_test_pred = train_ridge_model(X_train, y_train, X_test)

# 4. Trenowanie modelu LSTM
lstm_model, lstm_train_pred, lstm_test_pred = train_lstm_model(X_train, y_train, X_test)

# 5. Obliczenie metryk
def calculate_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.4f}, R²: {r2:.4f}")
    return mse, r2

print("\nPorównanie wyników:")
ridge_train_mse, ridge_train_r2 = calculate_metrics(y_train, ridge_train_pred, "Ridge Train")
ridge_test_mse, ridge_test_r2 = calculate_metrics(y_test, ridge_test_pred, "Ridge Test")
lstm_train_mse, lstm_train_r2 = calculate_metrics(y_train, lstm_train_pred, "LSTM Train")
lstm_test_mse, lstm_test_r2 = calculate_metrics(y_test, lstm_test_pred, "LSTM Test")

# 6. Wizualizacja wyników
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Rzeczywiste wartości', color='blue')
plt.plot(ridge_test_pred, label='Ridge Regression', color='green')
plt.plot(lstm_test_pred, label='LSTM', color='red')
plt.title('Porównanie prognoz na zbiorze testowym')
plt.xlabel('Indeks')
plt.ylabel('Temperatura')
plt.legend()
plt.show()

# 7. Zapis modeli (opcjonalnie)
import joblib
ridge_model_filename = 'ridge_model.pkl'
lstm_model_filename = 'lstm_model.h5'

joblib.dump(ridge_model, ridge_model_filename)
lstm_model.save(lstm_model_filename)
print(f"Modele zapisane: {ridge_model_filename}, {lstm_model_filename}")
