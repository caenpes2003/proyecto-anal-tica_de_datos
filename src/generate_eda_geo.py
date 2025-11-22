"""
Script para generar visualizaciones y analisis geoespacial - Seccion 6
Actualizacion con POIs y features geoespaciales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuracion de visualizacion
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ANALISIS EXPLORATORIO GEOESPACIAL - SECCION 6")
print("="*80)

# 1. Cargar datos
print("\n[1/5] Cargando dataset geoespacial...")
df = pd.read_csv('data/processed/siniestros_features_geo.csv')
print(f"Dataset cargado: {len(df):,} registros x {df.shape[1]} columnas")

# Filtrar solo registros con coordenadas validas
df_geo = df[df['latitud'].notna()].copy()
print(f"Registros con datos geoespaciales: {len(df_geo):,}")

# 2. Visualizaciones geoespaciales
print("\n[2/5] Generando visualizaciones geoespaciales...")

# 2.1 Distribucion de distancias a POIs
print("  [2.1] Histogramas de distancias a POIs...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
categorias = ['centros_comerciales', 'estadios', 'bares', 'transmilenio']
nombres = ['Centros Comerciales', 'Estadios', 'Bares/Pubs/Discotecas', 'TransMilenio']

for idx, (cat, nombre) in enumerate(zip(categorias, nombres)):
    ax = axes[idx//2, idx%2]
    col = f'dist_{cat}'

    # Filtrar valores validos
    distancias = df_geo[col].dropna() / 1000  # Convertir a km

    ax.hist(distancias, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(1, color='red', linestyle='--', linewidth=2, label='Umbral 1km')
    ax.set_xlabel('Distancia (km)')
    ax.set_ylabel('Frecuencia de Siniestros')
    ax.set_title(f'Distribucion de Distancias a {nombre}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Estadisticas en el grafico
    media = distancias.mean()
    mediana = distancias.median()
    ax.text(0.98, 0.95, f'Media: {media:.2f}km\\nMediana: {mediana:.2f}km',
            transform=ax.transAxes, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('data/processed/eda_visualizaciones/geo_01_distancias_pois.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Guardado: geo_01_distancias_pois.png")

# 2.2 Porcentaje de siniestros cerca de POIs
print("  [2.2] Grafico de barras: % siniestros cerca de POIs...")
fig, ax = plt.subplots(figsize=(10, 6))

porcentajes = []
for cat in categorias:
    cerca_col = f'cerca_{cat}'
    pct = (df_geo[cerca_col].sum() / len(df_geo)) * 100
    porcentajes.append(pct)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax.bar(nombres, porcentajes, color=colors, edgecolor='black', linewidth=1.5)

# Valores en las barras
for bar, pct in zip(bars, porcentajes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50% umbral')
ax.set_ylabel('Porcentaje de Siniestros (%)', fontsize=12)
ax.set_title('Siniestros Cercanos (<1km) a Puntos de Interes', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('data/processed/eda_visualizaciones/geo_02_porcentaje_cerca_pois.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Guardado: geo_02_porcentaje_cerca_pois.png")

# 2.3 Gravedad vs Proximidad a POIs
print("  [2.3] Boxplots: Gravedad vs proximidad a POIs...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (cat, nombre) in enumerate(zip(categorias, nombres)):
    ax = axes[idx//2, idx%2]
    cerca_col = f'cerca_{cat}'

    # Preparar datos
    data_plot = df_geo[[cerca_col, 'GRAVEDAD']].copy()
    data_plot[cerca_col] = data_plot[cerca_col].map({0: 'Lejos (>=1km)', 1: 'Cerca (<1km)'})

    # Boxplot
    sns.boxplot(data=data_plot, x=cerca_col, y='GRAVEDAD', ax=ax, palette='Set2')

    ax.set_xlabel('Proximidad', fontsize=11)
    ax.set_ylabel('Gravedad (1=Solo danos, 2=Heridos, 3=Muertos)', fontsize=11)
    ax.set_title(f'Gravedad vs Proximidad a {nombre}', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # T-test
    cerca = df_geo[df_geo[cerca_col] == 1]['GRAVEDAD']
    lejos = df_geo[df_geo[cerca_col] == 0]['GRAVEDAD']
    if len(cerca) > 0 and len(lejos) > 0:
        t_stat, p_value = stats.ttest_ind(cerca, lejos)
        ax.text(0.5, 0.95, f'p-value: {p_value:.4f}',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('data/processed/eda_visualizaciones/geo_03_gravedad_proximidad.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Guardado: geo_03_gravedad_proximidad.png")

# 2.4 Densidad de POIs vs Frecuencia de Siniestros
print("  [2.4] Scatter plots: Densidad de POIs vs frecuencia...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (cat, nombre) in enumerate(zip(categorias, nombres)):
    ax = axes[idx//2, idx%2]
    num_col = f'num_{cat}_1km'

    # Agrupar por densidad y contar siniestros
    densidad_counts = df_geo[num_col].value_counts().sort_index()

    ax.scatter(densidad_counts.index, densidad_counts.values, alpha=0.6, s=100)
    ax.set_xlabel(f'Cantidad de {nombre} en 1km', fontsize=11)
    ax.set_ylabel('Frecuencia de Siniestros', fontsize=11)
    ax.set_title(f'Siniestros vs Densidad de {nombre}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Correlacion
    if len(densidad_counts) > 1:
        corr = np.corrcoef(densidad_counts.index, densidad_counts.values)[0, 1]
        ax.text(0.05, 0.95, f'Correlacion: {corr:.3f}',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('data/processed/eda_visualizaciones/geo_04_densidad_pois.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Guardado: geo_04_densidad_pois.png")

# 3. Tests estadisticos para hipotesis geoespaciales
print("\n[3/5] Ejecutando tests estadisticos...")

resultados_tests = {}

# H4: Siniestros ocurren mas frecuentemente cerca de centros comerciales
print("  [H4] Siniestros cerca de centros comerciales...")
cerca_cc = df_geo[df_geo['cerca_centros_comerciales'] == 1]
lejos_cc = df_geo[df_geo['cerca_centros_comerciales'] == 0]
chi2, p_h4 = stats.chi2_contingency(pd.crosstab(
    df_geo['cerca_centros_comerciales'],
    df_geo['GRAVEDAD']
))[:2]
resultados_tests['H4'] = {
    'hipotesis': 'Siniestros ocurren mas frecuentemente cerca de centros comerciales',
    'test': 'Chi-cuadrado',
    'estadistico': chi2,
    'p_value': p_h4,
    'resultado': 'ACEPTADA' if p_h4 < 0.05 else 'RECHAZADA',
    'interpretacion': f'{len(cerca_cc)} ({len(cerca_cc)/len(df_geo)*100:.1f}%) siniestros cerca vs {len(lejos_cc)} lejos'
}

# H5: Siniestros nocturnos mas frecuentes cerca de bares
print("  [H5] Siniestros nocturnos cerca de bares...")
df_geo['es_noche'] = (df_geo['periodo_dia'] == 'NOCHE').astype(int) if 'periodo_dia' in df_geo.columns else 0
noche_cerca_bares = df_geo[(df_geo['es_noche'] == 1) & (df_geo['cerca_bares'] == 1)]
noche_lejos_bares = df_geo[(df_geo['es_noche'] == 1) & (df_geo['cerca_bares'] == 0)]
if len(noche_cerca_bares) > 0 and len(noche_lejos_bares) > 0:
    chi2_h5, p_h5 = stats.chi2_contingency(pd.crosstab(
        df_geo[df_geo['es_noche'] == 1]['cerca_bares'],
        df_geo[df_geo['es_noche'] == 1]['GRAVEDAD']
    ))[:2]
else:
    chi2_h5, p_h5 = 0, 1.0

resultados_tests['H5'] = {
    'hipotesis': 'Siniestros nocturnos mas frecuentes cerca de bares',
    'test': 'Chi-cuadrado (solo periodo nocturno)',
    'estadistico': chi2_h5,
    'p_value': p_h5,
    'resultado': 'ACEPTADA' if p_h5 < 0.05 else 'RECHAZADA',
    'interpretacion': f'{len(noche_cerca_bares)} siniestros nocturnos cerca de bares vs {len(noche_lejos_bares)} lejos'
}

# H6: Gravedad mayor cerca de TransMilenio (trafico mixto)
print("  [H6] Gravedad cerca de TransMilenio...")
cerca_tm = df_geo[df_geo['cerca_transmilenio'] == 1]['GRAVEDAD']
lejos_tm = df_geo[df_geo['cerca_transmilenio'] == 0]['GRAVEDAD']
if len(cerca_tm) > 0 and len(lejos_tm) > 0:
    t_stat_h6, p_h6 = stats.ttest_ind(cerca_tm, lejos_tm)
else:
    t_stat_h6, p_h6 = 0, 1.0

resultados_tests['H6'] = {
    'hipotesis': 'Gravedad mayor cerca de estaciones TransMilenio',
    'test': 't-test',
    'estadistico': t_stat_h6,
    'p_value': p_h6,
    'resultado': 'ACEPTADA' if p_h6 < 0.05 else 'RECHAZADA',
    'interpretacion': f'Gravedad media cerca: {cerca_tm.mean():.2f} vs lejos: {lejos_tm.mean():.2f}'
}

# 4. Generar reporte de tests
print("\n[4/5] Generando reporte de hipotesis geoespaciales...")
reporte = "# RESULTADOS DE TESTS ESTADISTICOS - HIPOTESIS GEOESPACIALES\\n\\n"
for key, resultado in resultados_tests.items():
    reporte += f"## {key}: {resultado['hipotesis']}\\n\\n"
    reporte += f"- **Test:** {resultado['test']}\\n"
    reporte += f"- **Estadistico:** {resultado['estadistico']:.4f}\\n"
    reporte += f"- **P-value:** {resultado['p_value']:.6f}\\n"
    reporte += f"- **Resultado:** {resultado['resultado']} (alpha=0.05)\\n"
    reporte += f"- **Interpretacion:** {resultado['interpretacion']}\\n\\n"

with open('data/processed/tests_geoespaciales.txt', 'w', encoding='utf-8') as f:
    f.write(reporte)

print("    Guardado: tests_geoespaciales.txt")

# 5. Resumen final
print("\n[5/5] Generando resumen estadistico...")
resumen = {
    'Total registros con coordenadas': len(df_geo),
    'Siniestros cerca centros comerciales': df_geo['cerca_centros_comerciales'].sum(),
    'Siniestros cerca estadios': df_geo['cerca_estadios'].sum(),
    'Siniestros cerca bares': df_geo['cerca_bares'].sum(),
    'Siniestros cerca TransMilenio': df_geo['cerca_transmilenio'].sum(),
}

print("\\n" + "="*80)
print("RESUMEN DE ANALISIS GEOESPACIAL")
print("="*80)
for key, value in resumen.items():
    print(f"  {key}: {value:,}")

print(f"\\n Visualizaciones generadas: 4")
print(f"  Hipotesis geoespaciales evaluadas: 3")
print(f"\\n Archivos guardados en: data/processed/eda_visualizaciones/")
print("="*80)

print("\\nANALISIS GEOESPACIAL COMPLETADO")
