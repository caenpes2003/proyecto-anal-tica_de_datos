"""
Script SIMPLE para completar modelado - SIN re-entrenar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Los modelos ya están entrenados, solo falta generar visualizaciones y reporte

base_dir = Path('.')
viz_dir = base_dir / 'reports' / 'visualizaciones_modelos'
class_dir = viz_dir / 'clasificacion'
comp_dir = viz_dir / 'comparacion'

# Métricas YA obtenidas (del output anterior)
resultados = {
    'Logistic_Regression': {
        'accuracy': 0.7822,
        'precision_macro': 0.5598,
        'recall_macro': 0.4650,
        'f1_macro': 0.4727,
        'tiempo': 7.16
    },
    'Random_Forest': {
        'accuracy': 0.7564,
        'precision_macro': 0.5225,
        'recall_macro': 0.4976,
        'f1_macro': 0.5023,
        'tiempo': 264.66
    },
    'XGBoost': {
        'accuracy': 0.7835,
        'precision_macro': 0.5383,
        'recall_macro': 0.4741,
        'f1_macro': 0.4823,
        'tiempo': 85.27
    }
}

print("Generando visualizaciones faltantes...")

# Gráfico de comparación
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
modelos = list(resultados.keys())
metricas_nombres = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
metricas_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, (metrica, label) in enumerate(zip(metricas_nombres, metricas_labels)):
    ax = axes[idx // 2, idx % 2]
    valores = [resultados[m][metrica] for m in modelos]

    bars = ax.bar(modelos, valores, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    ax.set_ylabel(label, fontweight='bold')
    ax.set_title(f'Comparacion: {label}', fontweight='bold', fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(comp_dir / 'comparacion_modelos.png', bbox_inches='tight', dpi=300)
plt.close()
print("OK Grafico de comparacion generado")

# Tabla resumen
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

tabla_data = [['MODELO', 'ACCURACY', 'PRECISION', 'RECALL', 'F1-SCORE', 'TIEMPO (s)']]
for nombre in modelos:
    m = resultados[nombre]
    fila = [
        nombre,
        f"{m['accuracy']:.4f}",
        f"{m['precision_macro']:.4f}",
        f"{m['recall_macro']:.4f}",
        f"{m['f1_macro']:.4f}",
        f"{m['tiempo']:.2f}"
    ]
    tabla_data.append(fila)

table = ax.table(cellText=tabla_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(len(tabla_data[0])):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Resaltar mejor modelo (Random Forest tiene mejor F1)
mejor_idx = 2  # Random Forest
for i in range(len(tabla_data[0])):
    table[(mejor_idx, i)].set_facecolor('#2ecc71')
    table[(mejor_idx, i)].set_text_props(weight='bold')

ax.set_title('Comparación de Modelos - Métricas de Evaluación',
            fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(comp_dir / 'tabla_resumen.png', bbox_inches='tight', dpi=300)
plt.close()
print("OK Tabla resumen generada")

# Generar informe markdown
informe = f"""# Seccion 7: Metodologia de Modelado

## Analisis de Siniestros Viales - Bogota

---

## Resumen Ejecutivo

Se implementaron **3 modelos de clasificacion** para predecir la gravedad de siniestros viales.

**Mejor modelo:** Random Forest (F1-Score: 0.5023)

**Dataset:**
- Train set: 156,921 registros
- Test set: 39,231 registros
- Features: 15 variables

---

## 7.1. Seleccion del Modelo

### Problema de Negocio

Predecir la **gravedad de siniestros viales** (Con Muertos / Con Heridos / Solo Danos) basado en:
- Factores temporales (hora, dia, mes)
- Factores geograficos (localidad, zona, tipo de via)
- Caracteristicas del siniestro (clase, choque, diseno)

### Tipo de Problema: Clasificacion Multiclase

**Variable objetivo:** GRAVEDAD
- Clase 1: Con Muertos (1.5% del dataset)
- Clase 2: Con Heridos (33.3% del dataset)
- Clase 3: Solo Danos (65.2% del dataset)

**Desafio:** Clases muy desbalanceadas

---

## 7.2. Algoritmos y Tecnicas Utilizadas

### 1. Regresion Logistica

**Descripcion:**
- Modelo lineal para clasificacion multiclase
- Usa regresion logistica multinomial (softmax)
- Interpretable y rapido

**Resultados:**
- Accuracy: {resultados['Logistic_Regression']['accuracy']:.4f}
- Precision: {resultados['Logistic_Regression']['precision_macro']:.4f}
- Recall: {resultados['Logistic_Regression']['recall_macro']:.4f}
- F1-Score: {resultados['Logistic_Regression']['f1_macro']:.4f}
- Tiempo: {resultados['Logistic_Regression']['tiempo']:.2f}s

### 2. Random Forest

**Descripcion:**
- Ensemble de arboles de decision
- Reduce overfitting mediante bagging
- Captura interacciones no lineales

**Resultados:**
- Accuracy: {resultados['Random_Forest']['accuracy']:.4f}
- Precision: {resultados['Random_Forest']['precision_macro']:.4f}
- Recall: {resultados['Random_Forest']['recall_macro']:.4f}
- F1-Score: {resultados['Random_Forest']['f1_macro']:.4f}
- Tiempo: {resultados['Random_Forest']['tiempo']:.2f}s

### 3. XGBoost

**Descripcion:**
- Gradient Boosting optimizado
- Construye arboles secuencialmente
- Alta performance

**Resultados:**
- Accuracy: {resultados['XGBoost']['accuracy']:.4f}
- Precision: {resultados['XGBoost']['precision_macro']:.4f}
- Recall: {resultados['XGBoost']['recall_macro']:.4f}
- F1-Score: {resultados['XGBoost']['f1_macro']:.4f}
- Tiempo: {resultados['XGBoost']['tiempo']:.2f}s

---

## 7.3. Justificacion de Hiperparametros

### Metodo: RandomizedSearchCV

**Configuracion:**
- Iteraciones: 10 por modelo
- Validacion cruzada: 3-fold estratificado
- Metrica: F1-Score macro (por desbalance de clases)
- Paralelizacion: n_jobs=-1

**¿Por que RandomizedSearchCV?**
- Dataset grande (196k registros)
- Balance entre velocidad y calidad
- Explora eficientemente el espacio de parametros

---

## 7.4. Validacion Cruzada

### Estrategia

**1. Train-Test Split Estratificado (80/20)**
- Mantiene proporcion de clases
- Random state: 42 (reproducibilidad)

**2. Validacion Cruzada 3-Fold Estratificada**
- Durante optimizacion de hiperparametros
- Metrica: F1-Score macro

### Manejo de Desbalance

**Tecnicas aplicadas:**
1. Stratified sampling en todos los splits
2. Class weight balancing probado en modelos
3. Metrica F1-macro (trata todas las clases igual)

---

## Comparacion de Modelos

![Comparacion](reports/visualizaciones_modelos/comparacion/comparacion_modelos.png)

![Tabla](reports/visualizaciones_modelos/comparacion/tabla_resumen.png)

### Modelo Recomendado: Random Forest

**Justificacion:**
- Mejor F1-Score macro: 0.5023
- Mejor balance precision-recall
- Captura interacciones complejas
- Robusto ante clases desbalanceadas

---

## Conclusiones

### Hallazgos Principales

1. **Desafio del desbalance:**
   - Clase minoritaria (muertos 1.5%) muy dificil de predecir
   - Todos los modelos priorizan clase mayoritaria
   - F1-Score ~0.50 es razonable dado el desbalance extremo

2. **Mejor modelo: Random Forest**
   - F1-Score: 0.5023
   - Accuracy: 75.64%
   - Tiempo razonable: 4.4 minutos

3. **Features importantes:**
   - Hora del dia
   - Localidad
   - Tipo de via
   - Dia de la semana

### Aplicaciones Practicas

1. **Clasificacion de riesgo en tiempo real**
2. **Priorizacion de recursos de emergencia**
3. **Identificacion de escenarios de alto riesgo**
4. **Planificacion de intervenciones preventivas**

---

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Modelos entrenados:** 3
**Total registros:** 196,152
"""

informe_path = base_dir / 'INFORME_MODELADO.md'
with open(informe_path, 'w', encoding='utf-8') as f:
    f.write(informe)

print("OK Informe generado: INFORME_MODELADO.md")

print("\n" + "="*70)
print("MODELADO COMPLETADO")
print("="*70)
print(f"Mejor modelo: Random Forest (F1={resultados['Random_Forest']['f1_macro']:.4f})")
print(f"Visualizaciones: reports/visualizaciones_modelos/")
print(f"Informe: INFORME_MODELADO.md")
