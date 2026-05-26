import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CELDA 1: CARGAR DATOS (sin limpiar todavía)
# ============================================================================
df = pd.read_csv('housing_clean_FINAL.csv')
print(f"Registros originales: {len(df)}")

# Remover SOLO valores negativos (limpieza básica, segura)
df = df[(df['price'] > 0) & (df['surface'] > 0)]
print(f"Registros después de limpiar negativos: {len(df)}")

# ============================================================================
# CELDA 2: TRAIN/TEST SPLIT (PRIMERO, antes de cualquier outlier removal)
# ============================================================================
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

print(f"\nTamaños ANTES de remover outliers:")
print(f"Train size: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")

# ============================================================================
# CELDA 3: CALCULAR Q99 SOLO EN TRAIN (SIN DATA LEAKAGE)
# ============================================================================
Q99_train = y_train.quantile(0.99)
print(f"\nQ99 calculado SOLO en TRAIN: {Q99_train:.2f}")

# Remover outliers solo en train
mask_train = y_train <= Q99_train
X_train_clean = X_train[mask_train]
y_train_clean = y_train[mask_train]

n_removed = (~mask_train).sum()
print(f"Registros removidos del train (outliers): {n_removed}")
print(f"\nTamaños DESPUÉS de remover outliers:")
print(f"Train size: {len(X_train_clean)} ({len(X_train_clean)/(len(X_train_clean)+len(X_test))*100:.1f}%)")
print(f"Test size: {len(X_test)} (sin modificar)")

# ============================================================================
# CELDA 4: ENTRENAR MODELO CON MEJORES PARÁMETROS
# ============================================================================
best_params = {
    'colsample_bytree': 0.8,
    'gamma': 0,
    'learning_rate': 0.01,
    'max_depth': 9,
    'min_child_weight': 1,
    'n_estimators': 500,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

print("\n" + "="*70)
print("ENTRENANDO MODELO DEFINITIVO")
print("="*70)
print("Parámetros:")
for key, value in best_params.items():
    if key not in ['random_state', 'n_jobs', 'tree_method']:
        print(f"  {key}: {value}")

modelo_final = xgb.XGBRegressor(**best_params)
modelo_final.fit(X_train_clean, y_train_clean)
print("\n✓ Modelo entrenado\n")

# ============================================================================
# CELDA 5: PREDICCIONES EN TRAIN Y TEST
# ============================================================================
y_train_pred = modelo_final.predict(X_train_clean)
y_test_pred = modelo_final.predict(X_test)

# ============================================================================
# CELDA 6: CALCULAR MÉTRICAS (incluyendo MSE)
# ============================================================================
# TRAIN
train_mse = mean_squared_error(y_train_clean, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train_clean, y_train_pred)
train_r2 = r2_score(y_train_clean, y_train_pred)

# TEST
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# ============================================================================
# CELDA 7: MOSTRAR COMPARACIÓN TRAIN/TEST
# ============================================================================
print("="*70)
print("COMPARACIÓN MÉTRICAS TRAIN vs TEST")
print("="*70)
print(f"{'Métrica':<12} {'Train':<15} {'Test':<15} {'Diferencia %':<15}")
print("-"*70)
print(f"{'MSE':<12} {train_mse:<15.2f} {test_mse:<15.2f} {((test_mse - train_mse) / train_mse * 100):>13.2f}%")
print(f"{'RMSE':<12} {train_rmse:<15.2f} {test_rmse:<15.2f} {((test_rmse - train_rmse) / train_rmse * 100):>13.2f}%")
print(f"{'MAE':<12} {train_mae:<15.2f} {test_mae:<15.2f} {((test_mae - train_mae) / train_mae * 100):>13.2f}%")
print(f"{'R²':<12} {train_r2:<15.4f} {test_r2:<15.4f} {((test_r2 - train_r2) / train_r2 * 100):>13.2f}%")
print("="*70)

# ============================================================================
# CELDA 8: ANÁLISIS DE OVERFITTING
# ============================================================================
print("\n" + "="*70)
print("ANÁLISIS DE OVERFITTING")
print("="*70)
rmse_diff = test_rmse - train_rmse
mae_diff = test_mae - train_mae
rmse_diff_pct = (rmse_diff / train_rmse * 100)

if rmse_diff_pct < 5:
    status = "✓ EXCELENTE: Sin overfitting"
elif rmse_diff_pct < 15:
    status = "✓ BIEN: Bajo overfitting"
elif rmse_diff_pct < 30:
    status = "⚠ MODERADO: Overfitting aceptable"
else:
    status = "✗ ALTO: Overfitting significativo"

print(f"RMSE diferencia (Test - Train): {rmse_diff:.2f} ({rmse_diff_pct:.2f}%) → {status}")
print(f"MAE diferencia (Test - Train): {mae_diff:.2f} ({(mae_diff / train_mae * 100):.2f}%)")
print("="*70)

# ============================================================================
# CELDA 9: FEATURE IMPORTANCE
# ============================================================================
importances = pd.DataFrame({
    'feature': X_train_clean.columns,
    'importance': modelo_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 10 FEATURES:")
print(importances.head(10).to_string(index=False))

# ============================================================================
# CELDA 10: PLOT COMPARACIÓN TRAIN/TEST (RMSE, MAE, MSE)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Métricas comparación
metrics = ['MSE', 'RMSE', 'MAE']
train_vals = [train_mse, train_rmse, train_mae]
test_vals = [test_mse, test_rmse, test_mae]

x = np.arange(len(metrics))
width = 0.35

bars1 = axes[0].bar(x - width/2, train_vals, width, label='Train', color='steelblue', alpha=0.8)
bars2 = axes[0].bar(x + width/2, test_vals, width, label='Test', color='coral', alpha=0.8)

axes[0].set_xlabel('Métrica', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Valor', fontsize=12, fontweight='bold')
axes[0].set_title('Comparación MSE, RMSE y MAE (Train vs Test)', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics, fontsize=11)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3, axis='y')

# Añadir valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Feature Importance
importances.head(10).plot(x='feature', y='importance', kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Top 10 Feature Importance', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Importance', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Feature', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('modelo_definitivo_comparacion.png', dpi=100, bbox_inches='tight')
print("\n✓ Gráfico guardado: modelo_definitivo_comparacion.png")
plt.show()

# ============================================================================
# CELDA 11: DIAGNÓSTICO DE RESIDUOS
# ============================================================================
residuos_train = y_train_clean - y_train_pred
residuos_test = y_test - y_test_pred

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Residuos Test vs Predicciones
axes[0, 0].scatter(y_test_pred, residuos_test, alpha=0.3, s=10, color='coral')
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicciones', fontsize=11)
axes[0, 0].set_ylabel('Residuos', fontsize=11)
axes[0, 0].set_title('TEST: Residuos vs Predicciones', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Residuos Train vs Predicciones
axes[0, 1].scatter(y_train_pred, residuos_train, alpha=0.3, s=10, color='steelblue')
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicciones', fontsize=11)
axes[0, 1].set_ylabel('Residuos', fontsize=11)
axes[0, 1].set_title('TRAIN: Residuos vs Predicciones', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Distribución Residuos Test
axes[1, 0].hist(residuos_test, bins=50, edgecolor='black', color='coral', alpha=0.7)
axes[1, 0].set_xlabel('Residuos', fontsize=11)
axes[1, 0].set_ylabel('Frecuencia', fontsize=11)
axes[1, 0].set_title(f'TEST: Distribución Residuos (Mean={residuos_test.mean():.2f}, Std={residuos_test.std():.2f})', 
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Distribución Residuos Train
axes[1, 1].hist(residuos_train, bins=50, edgecolor='black', color='steelblue', alpha=0.7)
axes[1, 1].set_xlabel('Residuos', fontsize=11)
axes[1, 1].set_ylabel('Frecuencia', fontsize=11)
axes[1, 1].set_title(f'TRAIN: Distribución Residuos (Mean={residuos_train.mean():.2f}, Std={residuos_train.std():.2f})', 
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('diagnostico_residuos.png', dpi=100, bbox_inches='tight')
print("✓ Gráfico de residuos guardado: diagnostico_residuos.png")
plt.show()

# ============================================================================
# CELDA 12: EXPORTAR MODELO Y METADATA A PKL (con joblib)
# ============================================================================
print("\n" + "="*70)
print("EXPORTANDO MODELO A PKL")
print("="*70)

# Crear diccionario con metadata
model_metadata = {
    'model': modelo_final,
    'Q99_threshold': Q99_train,
    'features': X_train_clean.columns.tolist(),
    'feature_importances': importances.to_dict('records'),
    'training_metrics': {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    },
    'data_info': {
        'train_size': len(X_train_clean),
        'test_size': len(X_test),
        'n_outliers_removed': n_removed,
        'original_size': len(df)
    },
    'model_params': best_params
}

# Guardar con joblib (mejor que pickle)
dump(model_metadata, 'modelo_housing_final.pkl')
print("✓ Modelo + metadata guardado en: modelo_housing_final.pkl")

# Guardar SOLO el modelo si lo necesitas
dump(modelo_final, 'modelo_housing_xgb_solo.pkl')
print("✓ Modelo solo guardado en: modelo_housing_xgb_solo.pkl")

# ============================================================================
# CELDA 13: RESUMEN FINAL
# ============================================================================
print("\n" + "="*70)
print("RESUMEN FINAL DEL MODELO")
print("="*70)
print(f"Dataset original: {len(df):,} registros")
print(f"Dataset limpio (después train/test): Train={len(X_train_clean):,}, Test={len(X_test):,}")
print(f"Outliers removidos del train (Q99={Q99_train:.0f}): {n_removed:,}")
print(f"\n{'Métrica':<12} {'Train':<12} {'Test':<12} {'Diferencia':<12}")
print(f"{'─'*50}")
print(f"{'MSE':<12} {train_mse:<12.2f} {test_mse:<12.2f} {(test_mse - train_mse):>+10.2f}")
print(f"{'RMSE':<12} {train_rmse:<12.2f} {test_rmse:<12.2f} {(test_rmse - train_rmse):>+10.2f}")
print(f"{'MAE':<12} {train_mae:<12.2f} {test_mae:<12.2f} {(test_mae - train_mae):>+10.2f}")
print(f"{'R²':<12} {train_r2:<12.4f} {test_r2:<12.4f} {(test_r2 - train_r2):>+10.4f}")
print("="*70)
print("\n✓ Modelo definitivo entrenado y guardado en PKL")
print("✓ Listo para producción")
