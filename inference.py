"""
SCRIPT PARA CARGAR Y USAR EL MODELO GUARDADO EN PKL
Uso: python inference.py
"""

from joblib import load
import pandas as pd
import numpy as np

# ============================================================================
# CARGAR MODELO Y METADATA
# ============================================================================
model_data = load('modelo_housing_final.pkl')

modelo = model_data['model']
Q99_threshold = model_data['Q99_threshold']
features_list = model_data['features']
metrics = model_data['training_metrics']
data_info = model_data['data_info']

print("="*70)
print("MODELO CARGADO EXITOSAMENTE")
print("="*70)
print(f"\nMetadata del modelo:")
print(f"  - Threshold Q99 (outliers): {Q99_threshold:.2f}")
print(f"  - Features esperadas: {len(features_list)}")
print(f"  - Datos de entrenamiento: {data_info['train_size']:,} registros")
print(f"  - Outliers removidos: {data_info['n_outliers_removed']:,}")

print(f"\nMétricas de test:")
print(f"  - Test RMSE: {metrics['test_rmse']:.2f}")
print(f"  - Test MAE: {metrics['test_mae']:.2f}")
print(f"  - Test R²: {metrics['test_r2']:.4f}")

# ============================================================================
# EJEMPLO: HACER PREDICCIONES
# ============================================================================
print("\n" + "="*70)
print("EJEMPLO DE PREDICCIÓN")
print("="*70)

# Crear datos de ejemplo (AJUSTA ESTO A TUS FEATURES REALES)
data_ejemplo = pd.DataFrame({
    'surface': [100, 150, 200],
    'rooms': [3, 4, 5],
    # Añade más features según tu modelo
})

# Asegurarse de que tienen todos los features en el orden correcto
data_ejemplo = data_ejemplo[features_list]

# Hacer predicción
predicciones = modelo.predict(data_ejemplo)

print(f"\nDatos de entrada:")
print(data_ejemplo)
print(f"\nPredicciones (price):")
for i, pred in enumerate(predicciones):
    print(f"  Fila {i+1}: ${pred:,.2f}")

# ============================================================================
# IMPORTANTE: VALIDACIÓN EN PRODUCCIÓN
# ============================================================================
print("\n" + "="*70)
print("CHECKLIST ANTES DE USAR EN PRODUCCIÓN")
print("="*70)
print(f"✓ Modelo cargado desde: modelo_housing_final.pkl")
print(f"✓ Q99 threshold guardado: {Q99_threshold:.2f}")
print(f"✓ Features ordenadas y validadas")
print(f"✓ RECUERDA: Remover outliers en TRAIN con Q99={Q99_threshold:.2f}")
print(f"✓ NO modificar test/nuevos datos con este threshold")
print("="*70)
