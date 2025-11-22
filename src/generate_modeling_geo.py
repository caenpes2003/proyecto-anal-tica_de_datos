"""
Re-entrenamiento de modelos con features geoespaciales - Seccion 7
"""

import pandas as pd
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RE-ENTRENAMIENTO DE MODELOS CON FEATURES GEOESPACIALES")
print("="*80)

# 1. Cargar dataset con features geoespaciales
print("\n[1/6] Cargando dataset con features geoespaciales...")
df = pd.read_csv('data/processed/siniestros_features_geo.csv')
print(f"Dataset cargado: {len(df):,} registros x {df.shape[1]} columnas")

# Filtrar solo registros con coordenadas validas
df_geo = df[df['latitud'].notna()].copy()
print(f"Registros con features geoespaciales: {len(df_geo):,}")

# 2. Preparar features
print("\n[2/6] Preparando features para modelado...")

# Target (recodificar para XGBoost: 1,2,3 -> 0,1,2)
y = df_geo['GRAVEDAD'] - 1  # Ahora: 0=solo danos, 1=heridos, 2=muertos

# Features base (numericas del modelo original)
features_base = [
    'anio', 'mes', 'dia_semana', 'hora_num', 'es_fin_semana',
    'CODIGO_LOCALIDAD', 'CLASE', 'CHOQUE',
    'INVOLUCRA_OBJETO_FIJO', 'trimestre', 'semana_anio',
    'riesgo_alto', 'choque_multiple', 'franja_horaria_bin'
]

# Features geoespaciales (nuevas)
features_geo = [
    'dist_centros_comerciales', 'dist_estadios', 'dist_bares', 'dist_transmilenio',
    'cerca_centros_comerciales', 'cerca_estadios', 'cerca_bares', 'cerca_transmilenio',
    'num_centros_comerciales_1km', 'num_estadios_1km', 'num_bares_1km', 'num_transmilenio_1km'
]

# Verificar features disponibles
features_disponibles_base = [f for f in features_base if f in df_geo.columns]
features_disponibles_geo = [f for f in features_geo if f in df_geo.columns]

print(f"  Features base disponibles: {len(features_disponibles_base)}")
print(f"  Features geoespaciales disponibles: {len(features_disponibles_geo)}")

# Modelo SIN features geoespaciales
X_sin_geo = df_geo[features_disponibles_base].copy()

# Modelo CON features geoespaciales
features_completas = features_disponibles_base + features_disponibles_geo
X_con_geo = df_geo[features_completas].copy()

# Rellenar NaN en features geoespaciales
for col in features_disponibles_geo:
    if 'dist_' in col:
        X_con_geo[col].fillna(X_con_geo[col].median(), inplace=True)
    else:
        X_con_geo[col].fillna(0, inplace=True)

print(f"\n  Modelo SIN geo: {X_sin_geo.shape[1]} features")
print(f"  Modelo CON geo: {X_con_geo.shape[1]} features ({X_con_geo.shape[1] - X_sin_geo.shape[1]} nuevas)")

# 3. Split estratificado
print("\n[3/6] Dividiendo en train/test (80/20)...")
X_sin_train, X_sin_test, y_sin_train, y_sin_test = train_test_split(
    X_sin_geo, y, test_size=0.2, random_state=42, stratify=y
)
X_con_train, X_con_test, y_con_train, y_con_test = train_test_split(
    X_con_geo, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {len(X_sin_train):,} | Test: {len(X_sin_test):,}")
print(f"  Distribucion y_train: {y_sin_train.value_counts().to_dict()}")

# 4. Entrenar modelos
print("\n[4/6] Entrenando modelos...")

resultados = {}

modelos_config = {
    'Logistic_Regression': LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
    'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss', n_jobs=-1)
}

for nombre, modelo in modelos_config.items():
    print(f"\n  [{nombre}]")

    # MODELO SIN GEO
    print(f"    Entrenando SIN features geoespaciales...")
    start = time.time()
    modelo_sin = modelo.__class__(**modelo.get_params())
    modelo_sin.fit(X_sin_train, y_sin_train)
    tiempo_sin = time.time() - start

    y_pred_sin = modelo_sin.predict(X_sin_test)
    acc_sin = accuracy_score(y_sin_test, y_pred_sin)
    f1_sin = f1_score(y_sin_test, y_pred_sin, average='macro')

    print(f"      Accuracy: {acc_sin:.4f} | F1-macro: {f1_sin:.4f} | Tiempo: {tiempo_sin:.2f}s")

    # MODELO CON GEO
    print(f"    Entrenando CON features geoespaciales...")
    start = time.time()
    modelo_con = modelo.__class__(**modelo.get_params())
    modelo_con.fit(X_con_train, y_con_train)
    tiempo_con = time.time() - start

    y_pred_con = modelo_con.predict(X_con_test)
    acc_con = accuracy_score(y_con_test, y_pred_con)
    f1_con = f1_score(y_con_test, y_pred_con, average='macro')

    print(f"      Accuracy: {acc_con:.4f} | F1-macro: {f1_con:.4f} | Tiempo: {tiempo_con:.2f}s")

    # Mejora
    mejora_f1 = ((f1_con - f1_sin) / f1_sin) * 100
    print(f"      MEJORA F1: {mejora_f1:+.2f}%")

    resultados[nombre] = {
        'sin_geo': {'accuracy': acc_sin, 'f1_macro': f1_sin, 'tiempo': tiempo_sin},
        'con_geo': {'accuracy': acc_con, 'f1_macro': f1_con, 'tiempo': tiempo_con},
        'mejora_f1_pct': mejora_f1,
        'modelo_sin': modelo_sin,
        'modelo_con': modelo_con
    }

# 5. Visualizaciones comparativas
print("\n[5/6] Generando visualizaciones comparativas...")

# 5.1 Comparacion F1-Score
fig, ax = plt.subplots(figsize=(10, 6))

modelos = list(resultados.keys())
f1_sin = [resultados[m]['sin_geo']['f1_macro'] for m in modelos]
f1_con = [resultados[m]['con_geo']['f1_macro'] for m in modelos]

x = np.arange(len(modelos))
width = 0.35

bars1 = ax.bar(x - width/2, f1_sin, width, label='Sin Features Geo', color='#FF6B6B', edgecolor='black')
bars2 = ax.bar(x + width/2, f1_con, width, label='Con Features Geo', color='#4ECDC4', edgecolor='black')

# Valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Modelos', fontsize=12)
ax.set_ylabel('F1-Score Macro', fontsize=12)
ax.set_title('Comparacion F1-Score: SIN vs CON Features Geoespaciales', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ') for m in modelos])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/eda_visualizaciones/modelo_geo_01_comparacion_f1.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Guardado: modelo_geo_01_comparacion_f1.png")

# 5.2 Mejora porcentual
fig, ax = plt.subplots(figsize=(10, 6))

mejoras = [resultados[m]['mejora_f1_pct'] for m in modelos]
colors = ['#2ECC71' if m > 0 else '#E74C3C' for m in mejoras]

bars = ax.bar(modelos, mejoras, color=colors, edgecolor='black', linewidth=1.5)

# Valores en las barras
for bar, mejora in zip(bars, mejoras):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mejora:+.2f}%',
            ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')

ax.axhline(0, color='black', linewidth=1)
ax.set_ylabel('Mejora F1-Score (%)', fontsize=12)
ax.set_title('Impacto de Features Geoespaciales en F1-Score', fontsize=14, fontweight='bold')
ax.set_xticklabels([m.replace('_', ' ') for m in modelos])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/eda_visualizaciones/modelo_geo_02_mejora_porcentual.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Guardado: modelo_geo_02_mejora_porcentual.png")

# 6. Reporte final
print("\n[6/6] Generando reporte...")

reporte = "# RESULTADOS: MODELOS CON FEATURES GEOESPACIALES\\n\\n"
reporte += "## Comparacion de Rendimiento\\n\\n"

for nombre, res in resultados.items():
    reporte += f"### {nombre.replace('_', ' ')}\\n\\n"
    reporte += f"**SIN Features Geoespaciales:**\\n"
    reporte += f"- Accuracy: {res['sin_geo']['accuracy']:.4f}\\n"
    reporte += f"- F1-macro: {res['sin_geo']['f1_macro']:.4f}\\n"
    reporte += f"- Tiempo: {res['sin_geo']['tiempo']:.2f}s\\n\\n"

    reporte += f"**CON Features Geoespaciales:**\\n"
    reporte += f"- Accuracy: {res['con_geo']['accuracy']:.4f}\\n"
    reporte += f"- F1-macro: {res['con_geo']['f1_macro']:.4f}\\n"
    reporte += f"- Tiempo: {res['con_geo']['tiempo']:.2f}s\\n\\n"

    reporte += f"**MEJORA:** {res['mejora_f1_pct']:+.2f}% en F1-Score\\n\\n"
    reporte += "---\\n\\n"

# Mejor modelo
mejor_modelo = max(resultados.items(), key=lambda x: x[1]['con_geo']['f1_macro'])
reporte += f"## Mejor Modelo (CON geo): {mejor_modelo[0].replace('_', ' ')}\\n"
reporte += f"- F1-Score: {mejor_modelo[1]['con_geo']['f1_macro']:.4f}\\n"
reporte += f"- Accuracy: {mejor_modelo[1]['con_geo']['accuracy']:.4f}\\n"

with open('data/processed/resultados_modelos_geo.txt', 'w', encoding='utf-8') as f:
    f.write(reporte)

print("\n" + "="*80)
print("RESUMEN DE RESULTADOS")
print("="*80)
for nombre, res in resultados.items():
    print(f"\n{nombre.replace('_', ' ')}:")
    print(f"  SIN geo: F1={res['sin_geo']['f1_macro']:.4f}")
    print(f"  CON geo: F1={res['con_geo']['f1_macro']:.4f}")
    print(f"  Mejora: {res['mejora_f1_pct']:+.2f}%")

print("\n" + "="*80)
print("RE-ENTRENAMIENTO COMPLETADO")
print("="*80)
