"""
Script para calcular representatividad de la muestra de 30K vs población de 196K
usando test Chi-cuadrado de bondad de ajuste.

Autor: Camilo Peñuela Espinosa
Fecha: 2025-11-21
"""

import pandas as pd
import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("TEST DE REPRESENTATIVIDAD: CHI-CUADRADO")
print("=" * 70)

# 1. Cargar población completa (196K)
print("\n1. Cargando población completa...")
try:
    # Intentar cargar el dataset original
    df_poblacion = pd.read_excel('data/raw/ANSV_Siniestros_2015_2020.xlsx',
                                  sheet_name='SINIESTROS')
    print(f"   ✓ Población cargada: {len(df_poblacion):,} registros")
except:
    print("   ✗ No se pudo cargar Excel original")
    print("   Intentando con CSV procesado...")
    try:
        df_poblacion = pd.read_csv('data/raw/siniestros_completo.csv')
        print(f"   ✓ Población cargada: {len(df_poblacion):,} registros")
    except:
        print("   ✗ Error: No se encontró archivo de población")
        print("   Continuando con estimación basada en proporciones conocidas...")
        df_poblacion = None

# 2. Cargar muestra (30K geocodificada)
print("\n2. Cargando muestra geocodificada...")
try:
    df_muestra = pd.read_csv('data/processed/siniestros_geocodificados.csv')
    print(f"   ✓ Muestra cargada: {len(df_muestra):,} registros")
except:
    print("   ✗ No se pudo cargar muestra geocodificada")
    print("   Intentando con otros archivos...")
    try:
        df_muestra = pd.read_csv('data/processed/sample_30000_siniestros.csv')
        print(f"   ✓ Muestra cargada: {len(df_muestra):,} registros")
    except:
        print("   ✗ Error: No se encontró archivo de muestra")
        df_muestra = None

# 3. Calcular distribuciones por LOCALIDAD
print("\n3. Calculando distribuciones por LOCALIDAD...")

if df_poblacion is not None and df_muestra is not None:
    # Distribución población
    dist_poblacion = df_poblacion['LOCALIDAD'].value_counts().sort_index()
    prop_poblacion = (dist_poblacion / len(df_poblacion) * 100).round(2)

    # Distribución muestra
    dist_muestra = df_muestra['LOCALIDAD'].value_counts().sort_index()
    prop_muestra = (dist_muestra / len(df_muestra) * 100).round(2)

    # Alinear índices (por si hay localidades diferentes)
    todas_localidades = sorted(set(dist_poblacion.index) | set(dist_muestra.index))

    obs_poblacion = [dist_poblacion.get(loc, 0) for loc in todas_localidades]
    obs_muestra = [dist_muestra.get(loc, 0) for loc in todas_localidades]

    # Frecuencias esperadas en la muestra basadas en proporciones de población
    n_muestra = len(df_muestra)
    expected = [(count / len(df_poblacion)) * n_muestra for count in obs_poblacion]

    # Test Chi-cuadrado
    chi2_stat, p_value = chisquare(f_obs=obs_muestra, f_exp=expected)

    print(f"\n   Localidades analizadas: {len(todas_localidades)}")
    print(f"   Chi-cuadrado (χ²): {chi2_stat:.4f}")
    print(f"   p-value: {p_value:.4f}")

    # Interpretación
    print("\n" + "=" * 70)
    print("INTERPRETACIÓN DEL TEST")
    print("=" * 70)

    if p_value > 0.05:
        print(f"✓ p-value = {p_value:.4f} > 0.05")
        print("  → NO se rechaza H₀")
        print("  → La muestra ES representativa de la población")
        print(f"  → Confianza: {p_value*100:.1f}% de que diferencias son por azar")
    else:
        print(f"✗ p-value = {p_value:.4f} < 0.05")
        print("  → Se rechaza H₀")
        print("  → La muestra NO es perfectamente representativa")
        print("  → Sin embargo, para análisis exploratorio puede ser aceptable")

    # Crear tabla comparativa
    print("\n" + "=" * 70)
    print("COMPARACIÓN DETALLADA POR LOCALIDAD")
    print("=" * 70)

    comparacion = pd.DataFrame({
        'Localidad': todas_localidades,
        'Población_n': obs_poblacion,
        'Población_%': [(c/len(df_poblacion)*100) for c in obs_poblacion],
        'Muestra_n': obs_muestra,
        'Muestra_%': [(c/len(df_muestra)*100) for c in obs_muestra],
        'Diferencia_%': [(c/len(df_muestra)*100) - (p/len(df_poblacion)*100)
                        for c, p in zip(obs_muestra, obs_poblacion)]
    })

    comparacion = comparacion.round(2)
    comparacion = comparacion.sort_values('Población_%', ascending=False)

    print(comparacion.to_string(index=False))

    # Guardar resultados
    comparacion.to_csv('data/processed/test_representatividad.csv', index=False)
    print(f"\n✓ Tabla guardada en: data/processed/test_representatividad.csv")

    # Crear visualización
    print("\n4. Generando visualización comparativa...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Barras comparativas
    x = np.arange(len(todas_localidades))
    width = 0.35

    axes[0].bar(x - width/2, comparacion['Población_%'], width,
                label='Población (196K)', alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, comparacion['Muestra_%'], width,
                label='Muestra (30K)', alpha=0.8, color='coral')

    axes[0].set_xlabel('Localidad', fontsize=10)
    axes[0].set_ylabel('Porcentaje (%)', fontsize=10)
    axes[0].set_title('Comparación de Distribuciones por Localidad', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparacion['Localidad'], rotation=45, ha='right', fontsize=8)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Gráfico 2: Scatter de proporciones
    axes[1].scatter(comparacion['Población_%'], comparacion['Muestra_%'],
                   s=100, alpha=0.6, color='purple')

    # Línea de referencia (x=y)
    max_val = max(comparacion['Población_%'].max(), comparacion['Muestra_%'].max())
    axes[1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfecta igualdad')

    axes[1].set_xlabel('Población (%)', fontsize=10)
    axes[1].set_ylabel('Muestra (%)', fontsize=10)
    axes[1].set_title(f'Correlación: Población vs Muestra\nχ²={chi2_stat:.2f}, p={p_value:.4f}',
                     fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/processed/eda_visualizaciones/test_representatividad.png',
                dpi=300, bbox_inches='tight')
    print(f"   ✓ Gráfico guardado en: data/processed/eda_visualizaciones/test_representatividad.png")

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN PARA LA PRESENTACIÓN")
    print("=" * 70)
    print(f"""
VALIDACIÓN DE MUESTRA (Test Chi-cuadrado):

• Población total: {len(df_poblacion):,} registros
• Muestra analizada: {len(df_muestra):,} registros ({len(df_muestra)/len(df_poblacion)*100:.1f}%)
• Localidades comparadas: {len(todas_localidades)}

Resultado del test:
• Chi-cuadrado (χ²): {chi2_stat:.4f}
• p-value: {p_value:.4f}

Conclusión: {'La muestra ES representativa ✓' if p_value > 0.05 else 'La muestra tiene diferencias significativas ⚠'}

Para la presentación usar:
"{p_value:.3f}" como p-value
    """)

else:
    print("\n⚠ No se pudieron cargar los archivos necesarios")
    print("Verificar rutas de archivos...")

print("\n" + "=" * 70)
print("SCRIPT FINALIZADO")
print("=" * 70)
