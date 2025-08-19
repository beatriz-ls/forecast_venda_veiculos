import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

df = pd.read_excel("Venda Veiculos.xlsx")
df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values('Data')
df = df.set_index('Data')

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Unidades'], label='Unidades Vendidas')
plt.title('Série Temporal de Unidades Vendidas')
plt.xlabel('Data')
plt.ylabel('Unidades')
plt.legend()
plt.grid(True)
plt.show()

# Criar defasagens
def create_lagged_features(series, lags=12):
    df_lags = pd.DataFrame({'y': series})
    for lag in range(1, lags + 1):
        df_lags[f'lag_{lag}'] = df_lags['y'].shift(lag)
    return df_lags.dropna()

lags = 12
df_lagged = create_lagged_features(df['Unidades'], lags=lags)
X = df_lagged.drop(columns='y').values
y = df_lagged['y'].values

# Escalonador (fit no dataset completo só para exemplo)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TimeSeriesSplit para criar folds
tscv = TimeSeriesSplit(n_splits=3, test_size=12)

splits = list(tscv.split(X_scaled))
for fold, (train_index, val_index) in enumerate(splits, 1):
    X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # Plotando as séries dos folds
    plt.figure(figsize=(10, 4))
    plt.plot(val_index, y_val_fold, label='Validação')
    plt.plot(train_index, y_train_fold, label='Treino')
    plt.title(f'Fold {fold}')
    plt.xlabel('Índice temporal')
    plt.ylabel('Unidades')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Separar manualmente os folds
folds = list(tscv.split(X_scaled))

# Unir os folds 0 e 1 para treino
train_idx = np.concatenate([folds[0][0], folds[1][0], folds[0][1], folds[1][1]])
test_idx = folds[2][1]  # Apenas o conjunto de teste do fold 3

# Separar os dados
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Treinar MLP
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                     solver='adam', random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Prever no fold 3
y_pred_mlp = model.predict(X_test)

folds[2][1]

# ---------- Gráfico - MLP ----------
data_test = df.index[test_idx]

plt.figure(figsize=(12, 6))
plt.plot(data_test, y_test, label='Real', color='black')
plt.plot(data_test, y_pred_mlp, label='MLP', color='blue')
plt.title(f'Previsão de Unidades (MLP) - Fold {fold + 1}')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))

# Série real completa (eixo x = índices inteiros da série)
plt.plot(np.arange(len(y)), y, label='Real', color='black')

# Previsão no fold 3 (test_idx são os índices absolutos na série)
plt.plot(test_idx, y_pred_mlp, label='Previsão MLP (fold 3)', color='blue')

plt.title('Previsão de Unidades (MLP) - Série Completa')
plt.legend()
plt.grid(True)
plt.show()

# Loop de validação cruzada
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Reformatar para 3D: (amostras, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # RNN com Keras
    model = Sequential([
      LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
      Dropout(0.2),
      LSTM(32),
      Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

    model.fit(X_train, y_train, epochs=100, batch_size=8,
              validation_split=0.1, verbose=0,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    y_pred_rnn = model.predict(X_test)

from sklearn.preprocessing import StandardScaler

# Exemplo para escalonamento
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1)

X_test_scaled = scaler_X.transform(X_test.reshape(X_test.shape[0], -1))
X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

model.fit(X_train_scaled, y_train_scaled, epochs=200, batch_size=16,
          validation_split=0.1, verbose=1,
          callbacks=[EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)])

# Previsão
y_pred_scaled = model.predict(X_test_scaled).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()


# ---------- Gráfico - MLP ----------
plt.figure(figsize=(12, 6))
plt.plot(data_test, y_test, label='Real', color='black')
plt.plot(data_test, y_pred, label='RNN', color='red')
plt.title(f'Previsão de Unidades (MLP) - Fold {fold + 1}')
plt.legend()
plt.grid(True)
plt.show()