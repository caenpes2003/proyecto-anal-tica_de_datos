"""
Script para corregir coordenadas del Top 20 de intersecciones criticas.

Problema identificado:
- Las coordenadas en reporte_top20.csv no coinciden con las direcciones
- Ejemplo: "CL 100" aparece con coordenadas de otra zona

Solucion:
- Agrupar por interseccion y calcular promedio de coordenadas (centroide)
- Validar que coordenadas esten en rango razonable para Bogota
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Rutas
BASE_DIR = Path(r'c:\Users\caenp\OneDrive\Escritorio\analitica_siniestros')
INPUT_FILE = BASE_DIR / 'data' / 'processed' / 'siniestros_features_geo.csv'
OUTPUT_FILE = BASE_DIR / 'seccion_11_dashboard' / 'reporte_top20_corregido.csv'

print("="*80)
print("CORRECCION DE COORDENADAS TOP 20 INTERSECCIONES CRITICAS")
print("="*80)

# 1. Cargar datos
print("\n[1/5] Cargando datos geocodificados...")
df = pd.read_csv(INPUT_FILE)
print(f"   Total registros: {len(df):,}")

# Filtrar solo siniestros graves (GRAVEDAD >= 2)
df_graves = df[df['GRAVEDAD'] >= 2].copy()
print(f"   Siniestros graves (GRAVEDAD >= 2): {len(df_graves):,}")

# 2. Normalizar direcciones para agrupar
print("\n[2/5] Normalizando direcciones...")

def normalizar_direccion(direccion):
    """Normaliza direccion para agrupacion consistente"""
    if pd.isna(direccion):
        return None

    direccion = str(direccion).upper().strip()

    # Normalizar abreviaciones
    direccion = direccion.replace('AVENIDA', 'AV')
    direccion = direccion.replace('CALLE', 'CL')
    direccion = direccion.replace('CARRERA', 'KR')
    direccion = direccion.replace('AUTOPISTA', 'AK')
    direccion = direccion.replace('DIAGONAL', 'DG')
    direccion = direccion.replace('TRANSVERSAL', 'TV')

    # Quitar espacios multiples
    direccion = ' '.join(direccion.split())

    return direccion

df_graves['direccion_norm'] = df_graves['DIRECCION'].apply(normalizar_direccion)

# Filtrar direcciones validas
df_graves = df_graves[df_graves['direccion_norm'].notna()].copy()
print(f"   Registros con direccion valida: {len(df_graves):,}")

# 3. Agrupar por interseccion y calcular metricas
print("\n[3/5] Calculando metricas por interseccion...")

intersecciones = df_graves.groupby('direccion_norm').agg({
    'GRAVEDAD': ['count', 'mean'],
    'latitud': 'mean',  # PROMEDIO de todas las coordenadas
    'longitud': 'mean',
    'localidad_nombre': lambda x: x.mode()[0] if len(x) > 0 else 'DESCONOCIDO'
}).reset_index()

# Renombrar columnas
intersecciones.columns = ['Interseccion', 'Num_Siniestros', 'Gravedad_Promedio',
                          'Latitud', 'Longitud', 'Localidad']

print(f"   Total intersecciones unicas: {len(intersecciones):,}")

# 4. Validar coordenadas en rango de Bogota
print("\n[4/5] Validando coordenadas...")

# Limites aproximados de Bogota
LAT_MIN, LAT_MAX = 4.45, 4.85  # Bogota: ~4.45 a 4.85
LON_MIN, LON_MAX = -74.25, -73.95  # Bogota: ~-74.25 a -73.95

antes_validacion = len(intersecciones)
intersecciones = intersecciones[
    (intersecciones['Latitud'] >= LAT_MIN) & (intersecciones['Latitud'] <= LAT_MAX) &
    (intersecciones['Longitud'] >= LON_MIN) & (intersecciones['Longitud'] <= LON_MAX)
].copy()

descartados = antes_validacion - len(intersecciones)
print(f"   Intersecciones con coordenadas validas: {len(intersecciones):,}")
print(f"   Intersecciones descartadas (fuera de rango): {descartados}")

# 5. Calcular score de riesgo y seleccionar Top 20
print("\n[5/5] Calculando score de riesgo...")

# Formula validada: F1 = Densidad x Gravedad
# Aqui usamos Num_Siniestros como proxy de densidad
intersecciones['Score_Riesgo'] = intersecciones['Num_Siniestros'] * intersecciones['Gravedad_Promedio']

# Normalizar a escala 0-100
min_score = intersecciones['Score_Riesgo'].min()
max_score = intersecciones['Score_Riesgo'].max()
intersecciones['Score_Riesgo_Norm'] = ((intersecciones['Score_Riesgo'] - min_score) /
                                       (max_score - min_score)) * 100

# Clasificar nivel de riesgo
def clasificar_riesgo(score):
    if score >= 80:
        return 'CRITICO'
    elif score >= 60:
        return 'ALTO'
    elif score >= 40:
        return 'MEDIO'
    else:
        return 'BAJO'

intersecciones['Nivel_Riesgo'] = intersecciones['Score_Riesgo_Norm'].apply(clasificar_riesgo)

# Ordenar por score descendente
intersecciones = intersecciones.sort_values('Score_Riesgo_Norm', ascending=False)

# Seleccionar Top 20
top20 = intersecciones.head(20).copy()
top20['Ranking'] = range(1, 21)

# Reordenar columnas
top20 = top20[['Ranking', 'Interseccion', 'Localidad', 'Num_Siniestros',
               'Gravedad_Promedio', 'Score_Riesgo_Norm', 'Nivel_Riesgo',
               'Latitud', 'Longitud']]

# Guardar
print(f"\n[OK] Guardando Top 20 corregido en: {OUTPUT_FILE}")
top20.to_csv(OUTPUT_FILE, index=False)

# Mostrar resultados
print("\n" + "="*80)
print("TOP 20 INTERSECCIONES CRITICAS (COORDENADAS CORREGIDAS)")
print("="*80)
print(top20.to_string(index=False))

# Validacion adicional: verificar que "CL 100" tenga coordenadas cerca de Calle 100
print("\n" + "="*80)
print("VALIDACION: Verificando algunas intersecciones clave")
print("="*80)

# Calle 100 deberia estar cerca de latitud 4.68-4.70
calle_100 = top20[top20['Interseccion'].str.contains('CL 100', na=False)]
if not calle_100.empty:
    print(f"\nIntersecciones con 'CL 100':")
    print(calle_100[['Ranking', 'Interseccion', 'Latitud', 'Longitud']].to_string(index=False))

    for _, row in calle_100.iterrows():
        if 4.68 <= row['Latitud'] <= 4.71:
            print(f"   [OK] {row['Interseccion']} - Coordenadas consistentes con Calle 100")
        else:
            print(f"   [ADVERTENCIA] {row['Interseccion']} - Latitud {row['Latitud']:.4f} parece incorrecta")

# Calle 26 deberia estar cerca de latitud 4.62-4.65
calle_26 = top20[top20['Interseccion'].str.contains('CL 26', na=False)]
if not calle_26.empty:
    print(f"\nIntersecciones con 'CL 26':")
    print(calle_26[['Ranking', 'Interseccion', 'Latitud', 'Longitud']].to_string(index=False))

    for _, row in calle_26.iterrows():
        if 4.62 <= row['Latitud'] <= 4.65:
            print(f"   [OK] {row['Interseccion']} - Coordenadas consistentes con Calle 26")
        else:
            print(f"   [ADVERTENCIA] {row['Interseccion']} - Latitud {row['Latitud']:.4f} parece incorrecta")

# Calle 80 deberia estar cerca de latitud 4.69-4.72
calle_80 = top20[top20['Interseccion'].str.contains('CL 80', na=False)]
if not calle_80.empty:
    print(f"\nIntersecciones con 'CL 80':")
    print(calle_80[['Ranking', 'Interseccion', 'Latitud', 'Longitud']].to_string(index=False))

    for _, row in calle_80.iterrows():
        if 4.69 <= row['Latitud'] <= 4.72:
            print(f"   [OK] {row['Interseccion']} - Coordenadas consistentes con Calle 80")
        else:
            print(f"   [ADVERTENCIA] {row['Interseccion']} - Latitud {row['Latitud']:.4f} parece incorrecta")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)
print(f"\nArchivo generado: {OUTPUT_FILE}")
print("\nProximos pasos:")
print("1. Revisar el archivo reporte_top20_corregido.csv")
print("2. Si las coordenadas son correctas, reemplazar reporte_top20.csv")
print("3. Regenerar dashboard con coordenadas corregidas")
