import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Visualização
print("df: (linha, coluna) =", df.shape)
df.head()

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

# Visualização
print("df_lagged: (linha, coluna) =", df_lagged.shape)
df_lagged.head()

X = df_lagged.drop(columns='y').values
y = df_lagged['y'].values

print("X: (linha, coluna) =", X.shape)
print("y: (linha, coluna) =", y.shape)

# Escalonador (fit no dataset completo só para exemplo)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TimeSeriesSplit para criar os folds
tscv = TimeSeriesSplit(n_splits=3, test_size=12)
splits = list(tscv.split(X_scaled))

# Plotando as séries dos folds
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for fold, (train_index, val_index) in enumerate(splits, 1):
    X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # Plot no subplot correspondente
    ax = axes[fold-1]
    ax.plot(val_index, y_val_fold, label='Validação', color='orange', linewidth=2)
    ax.plot(train_index, y_train_fold, label='Treino', alpha=0.7)
    ax.set_title(f'Fold {fold}')
    ax.set_xlabel('Índice temporal')
    ax.set_ylabel('Unidades')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Separar manualmente os folds
folds = list(tscv.split(X_scaled))

# Unir os folds 0 e 1 para treino
train_idx = np.concatenate([folds[0][0], folds[1][0], folds[0][1], folds[1][1]])
test_idx = folds[2][1]  # Apenas o conjunto de teste do fold 3

# Separar os dados
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# --- Estatísticas básicas ---
print("Informações gerais:")
print(df.info())
print("\nEstatísticas descritivas:")
print(df['Unidades'].describe())

# --- Valores ausentes ---
print("\nValores ausentes:")
print(df.isna().sum())

# --- Série temporal bruta ---
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Unidades'], marker='o', linestyle='-')
plt.title("Série Temporal - Unidades Vendidas")
plt.xlabel("Data")
plt.ylabel("Unidades")
plt.grid(True)
plt.show()

# --- Histograma da distribuição ---
plt.figure(figsize=(8,5))
sns.histplot(df['Unidades'], bins=20, kde=True)
plt.title("Distribuição das Unidades Vendidas")
plt.xlabel("Unidades")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()

# --- Boxplot por ano ---
df['Ano'] = df.index.year
plt.figure(figsize=(10,5))
sns.boxplot(x='Ano', y='Unidades', data=df)
plt.title("Boxplot de Unidades Vendidas por Ano")
plt.xlabel("Ano")
plt.ylabel("Unidades")
plt.grid(True)
plt.show()

# --- Médias mensais ---
df['Mes'] = df.index.month
media_mensal = df.groupby('Mes')['Unidades'].mean()

plt.figure(figsize=(10,5))
media_mensal.plot(kind='bar', color='skyblue')
plt.title("Média de Unidades Vendidas por Mês")
plt.xlabel("Mês")
plt.ylabel("Unidades")
plt.grid(True)
plt.show()

# --- Correlação entre defasagens ---
lags = 12
fig, axes = plt.subplots(3, 4, figsize=(15,8))
for i, ax in enumerate(axes.flat, 1):
    if i <= lags:
        ax.scatter(df['Unidades'].shift(i), df['Unidades'])
        ax.set_title(f"Lag {i}")
        ax.set_xlabel(f"Unidades (t-{i})")
        ax.set_ylabel("Unidades (t)")
plt.tight_layout()
plt.show()

rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
print(f"RMSE: {rmse_mlp:.4f}")

# ---------- Gráfico - MLP ----------
data_test = df.index[test_idx]

plt.figure(figsize=(12, 6))
plt.plot(data_test, y_test, label='Real')
plt.plot(data_test, y_pred_mlp, label='MLP', color='black')
plt.title(f'Previsão de Unidades (MLP) - Fold {fold + 1}')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))

# Série real completa (eixo x = índices inteiros da série)
plt.plot(np.arange(len(y)), y, label='Real')

# Previsão no fold 3 (test_idx são os índices absolutos na série)
plt.plot(test_idx, y_pred_mlp, label='Previsão MLP (fold 3)', color='black')

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
    model_rnn = Sequential([
      LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
      Dropout(0.2),
      LSTM(32),
      Dense(1)
    ])
    model_rnn.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

    model_rnn.fit(X_train, y_train, epochs=100, batch_size=8,
              validation_split=0.1, verbose=0,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    y_pred_rnn = model_rnn.predict(X_test)

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
y_pred_rnn = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()


rmse_rnn = np.sqrt(mean_squared_error(y_test, y_pred_rnn))
print(f"RMSE: {rmse_rnn:.4f}")

# ---------- Gráfico - RNN ----------
plt.figure(figsize=(12, 6))
plt.plot(data_test, y_test, label='Real')
plt.plot(data_test, y_pred_rnn, label='RNN', color='black')
plt.title(f'Previsão de Unidades (RNN) - Fold {fold + 1}')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 6))

# Série real completa (eixo x = índices inteiros da série)
plt.plot(np.arange(len(y)), y, label='Real')

# Previsão no fold 3 (test_idx são os índices absolutos na série)
plt.plot(test_idx, y_pred_rnn, label='Previsão RNN (fold 3)', color='black')

plt.title('Previsão de Unidades (RNN) - Série Completa')
plt.legend()
plt.grid(True)
plt.show()

