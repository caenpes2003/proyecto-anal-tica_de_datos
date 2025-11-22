"""
Analisis temporal-geoespacial cruzado
Identifica zonas criticas por hora del dia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("ANALISIS TEMPORAL-GEOESPACIAL CRUZADO")
print("="*80)

# 1. Cargar datos
print("\n[1/3] Cargando dataset geoespacial...")
df = pd.read_csv('data/processed/siniestros_features_geo.csv')
df_geo = df[df['latitud'].notna()].copy()
print(f"Registros: {len(df_geo):,}")

# Extraer hora si no existe
if 'hora_num' not in df_geo.columns:
    df_geo['hora_num'] = pd.to_datetime(df_geo['HORA']).dt.hour

# 2. Analisis: Bares x Horario
print("\n[2/3] Analizando: Siniestros cerca de bares por hora...")

# Crear matriz: Hora x Proximidad a Bares
df_bares_hora = df_geo.groupby(['hora_num', 'cerca_bares']).size().unstack(fill_value=0)
df_bares_hora_pct = df_bares_hora.div(df_bares_hora.sum(axis=1), axis=0) * 100

# Visualizacion 1: Siniestros cerca de bares por hora
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Grafico 1: Frecuencia absoluta
ax1 = axes[0]
horas = df_bares_hora.index
cerca = df_bares_hora[1] if 1 in df_bares_hora.columns else pd.Series(0, index=horas)
lejos = df_bares_hora[0] if 0 in df_bares_hora.columns else pd.Series(0, index=horas)

ax1.bar(horas, cerca, label='Cerca de bares (<1km)', color='#FF6B6B', alpha=0.8)
ax1.bar(horas, lejos, bottom=cerca, label='Lejos de bares (>=1km)', color='#4ECDC4', alpha=0.8)

ax1.set_xlabel('Hora del Dia', fontsize=12)
ax1.set_ylabel('Cantidad de Siniestros', fontsize=12)
ax1.set_title('Distribucion Horaria de Siniestros por Proximidad a Bares', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(range(0, 24))

# Grafico 2: Porcentaje
ax2 = axes[1]
pct_cerca = df_bares_hora_pct[1] if 1 in df_bares_hora_pct.columns else pd.Series(0, index=horas)

ax2.plot(horas, pct_cerca, marker='o', linewidth=2, markersize=6, color='#E74C3C')
ax2.axhline(pct_cerca.mean(), color='black', linestyle='--', alpha=0.5, label=f'Promedio: {pct_cerca.mean():.1f}%')
ax2.fill_between(horas, pct_cerca, alpha=0.3, color='#E74C3C')

ax2.set_xlabel('Hora del Dia', fontsize=12)
ax2.set_ylabel('% Siniestros Cerca de Bares', fontsize=12)
ax2.set_title('Porcentaje de Siniestros Cercanos a Bares por Hora', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 24))
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('data/processed/eda_visualizaciones/geo_05_bares_por_hora.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Guardado: geo_05_bares_por_hora.png")

# 3. Analisis: Centros comerciales x Horario
print("\n  Analizando: Centros comerciales por hora...")

df_cc_hora = df_geo.groupby(['hora_num', 'cerca_centros_comerciales']).size().unstack(fill_value=0)
df_cc_hora_pct = df_cc_hora.div(df_cc_hora.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 6))

cerca_cc = df_cc_hora[1] if 1 in df_cc_hora.columns else pd.Series(0, index=horas)
pct_cerca_cc = df_cc_hora_pct[1] if 1 in df_cc_hora_pct.columns else pd.Series(0, index=horas)

# Barras
ax.bar(horas, cerca_cc, color='#3498DB', alpha=0.6, label='Siniestros cerca CC')

# Linea de porcentaje (eje secundario)
ax2 = ax.twinx()
ax2.plot(horas, pct_cerca_cc, color='#E67E22', marker='s', linewidth=2, markersize=6, label='% cerca CC')
ax2.set_ylabel('Porcentaje Cerca (%)', fontsize=12, color='#E67E22')
ax2.tick_params(axis='y', labelcolor='#E67E22')
ax2.set_ylim(0, 100)

ax.set_xlabel('Hora del Dia', fontsize=12)
ax.set_ylabel('Cantidad de Siniestros', fontsize=12)
ax.set_title('Siniestros Cerca de Centros Comerciales por Hora', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(0, 24))

plt.tight_layout()
plt.savefig('data/processed/eda_visualizaciones/geo_06_centros_comerciales_hora.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Guardado: geo_06_centros_comerciales_hora.png")

# 4. Heatmap: Hora x Tipo de POI cercano
print("\n[3/3] Generando heatmap: Hora x Tipo de POI...")

# Crear matriz de datos
pois_categorias = ['cerca_centros_comerciales', 'cerca_estadios', 'cerca_bares', 'cerca_transmilenio']
nombres_cortos = ['C.Comerciales', 'Estadios', 'Bares', 'TransMilenio']

matriz_pois_hora = pd.DataFrame()
for poi_col, nombre in zip(pois_categorias, nombres_cortos):
    if poi_col in df_geo.columns:
        conteos = df_geo[df_geo[poi_col] == 1].groupby('hora_num').size()
        matriz_pois_hora[nombre] = conteos

# Normalizar por fila (porcentaje del total de cada hora)
matriz_pois_hora_norm = matriz_pois_hora.div(matriz_pois_hora.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(matriz_pois_hora_norm.T, annot=True, fmt='.1f', cmap='YlOrRd',
            cbar_kws={'label': '% del total por hora'}, ax=ax, linewidths=0.5)

ax.set_xlabel('Hora del Dia', fontsize=12)
ax.set_ylabel('Tipo de POI Cercano', fontsize=12)
ax.set_title('Distribucion Horaria de Siniestros por Tipo de POI Cercano', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('data/processed/eda_visualizaciones/geo_07_heatmap_hora_poi.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Guardado: geo_07_heatmap_hora_poi.png")

# 5. Resumen de hallazgos
print("\n" + "="*80)
print("HALLAZGOS PRINCIPALES")
print("="*80)

# Hora con mas siniestros cerca de bares
hora_max_bares = pct_cerca.idxmax()
pct_max_bares = pct_cerca.max()
print(f"\nBares:")
print(f"  - Hora con mayor % cerca de bares: {hora_max_bares}:00 ({pct_max_bares:.1f}%)")
print(f"  - Promedio general: {pct_cerca.mean():.1f}%")

# Hora con mas siniestros cerca de centros comerciales
hora_max_cc = cerca_cc.idxmax()
cant_max_cc = cerca_cc.max()
print(f"\nCentros Comerciales:")
print(f"  - Hora con mas siniestros cerca CC: {hora_max_cc}:00 ({cant_max_cc:.0f} siniestros)")
print(f"  - % promedio cerca CC: {pct_cerca_cc.mean():.1f}%")

# Periodo critico (20:00 - 04:00)
periodo_nocturno = pct_cerca[(pct_cerca.index >= 20) | (pct_cerca.index <= 4)]
print(f"\nPeriodo Nocturno (20:00-04:00):")
print(f"  - % promedio cerca bares: {periodo_nocturno.mean():.1f}%")
print(f"  - Diferencia vs promedio general: {periodo_nocturno.mean() - pct_cerca.mean():+.1f} puntos")

print("\n" + "="*80)
print("ANALISIS COMPLETADO")
print("="*80)
