"""
Genera mapa HTML con Top 20 intersecciones criticas para Seccion 9.

Este mapa es una visualizacion simplificada que muestra solo el Top 20
con marcadores numerados y popups informativos.
"""

import pandas as pd
import folium
from pathlib import Path

# Configuracion
BASE_DIR = Path(r'c:\Users\caenp\OneDrive\Escritorio\analitica_siniestros')
TOP20_FILE = BASE_DIR / 'seccion_11_dashboard' / 'reporte_top20.csv'
OUTPUT_FILE = BASE_DIR / 'seccion_9_resultados' / 'visualizaciones' / '9.6_top20_intersecciones.html'

print("="*80)
print("GENERACION DE MAPA: TOP 20 INTERSECCIONES CRITICAS")
print("="*80)

# Cargar Top 20
print("\n[1/3] Cargando Top 20...")
top20 = pd.read_csv(TOP20_FILE)
print(f"   Intersecciones cargadas: {len(top20)}")

# Crear mapa centrado en Bogota
print("\n[2/3] Creando mapa...")
m = folium.Map(
    location=[4.6533, -74.0836],  # Centro de Bogota
    zoom_start=11,
    tiles='CartoDB positron'
)

# Agregar marcadores del Top 20
for idx, row in top20.iterrows():
    # Color segun nivel de riesgo
    color_map = {
        'CRITICO': '#8B0000',  # Rojo oscuro
        'ALTO': '#FF4500',     # Naranja rojizo
        'MEDIO': '#FFA500',    # Naranja
        'BAJO': '#FFD700'      # Amarillo
    }
    color = color_map.get(row['Nivel_Riesgo'], '#808080')

    # Popup con informacion detallada
    popup_html = f"""
    <div style="font-family: Arial; width: 280px;">
        <h3 style="margin:0; color:{color};">#{row['Ranking']} - {row['Nivel_Riesgo']}</h3>
        <hr style="margin:5px 0;">
        <p style="margin:5px 0;"><b>Interseccion:</b><br>{row['Interseccion']}</p>
        <p style="margin:5px 0;"><b>Localidad:</b> {row['Localidad']}</p>
        <p style="margin:5px 0;">
            <b>Siniestros graves:</b> {row['Num_Siniestros']}<br>
            <b>Gravedad promedio:</b> {row['Gravedad_Promedio']:.2f}<br>
            <b>Score de riesgo:</b> {row['Score_Riesgo_Norm']:.1f}/100
        </p>
        <hr style="margin:5px 0;">
        <p style="margin:5px 0; font-size:11px; color:grey;">
            Coordenadas: ({row['Latitud']:.5f}, {row['Longitud']:.5f})
        </p>
    </div>
    """

    # Icono personalizado con numero de ranking
    icon_html = f'''
    <div style="font-size: 14px; font-weight: bold; color: white;
                background-color: {color}; border-radius: 50%;
                width: 30px; height: 30px; display: flex;
                align-items: center; justify-content: center;
                border: 2px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
        {row['Ranking']}
    </div>
    '''

    folium.Marker(
        location=[row['Latitud'], row['Longitud']],
        popup=folium.Popup(popup_html, max_width=320),
        icon=folium.DivIcon(html=icon_html)
    ).add_to(m)

# Agregar titulo al mapa
title_html = '''
<div style="position: fixed;
            top: 10px; left: 50px; width: 400px; height: 90px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px; border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
    <h4 style="margin:0;">Top 20 Intersecciones Criticas</h4>
    <p style="margin:5px 0; font-size:12px;">
        Bogota D.C. (2015-2020)<br>
        <span style="color:#8B0000;">■</span> CRITICO
        <span style="color:#FF4500;">■</span> ALTO
        <span style="color:#FFA500;">■</span> MEDIO
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Guardar mapa
print(f"\n[3/3] Guardando mapa en: {OUTPUT_FILE}")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
m.save(str(OUTPUT_FILE))

print("\n" + "="*80)
print("MAPA GENERADO EXITOSAMENTE")
print("="*80)
print(f"\nArchivo: {OUTPUT_FILE}")
print(f"\nPara visualizar, abrir el archivo HTML en cualquier navegador.")
print("\nEstadisticas del Top 20:")
print(f"  - Total intersecciones: {len(top20)}")
print(f"  - CRITICAS: {len(top20[top20['Nivel_Riesgo'] == 'CRITICO'])}")
print(f"  - ALTAS: {len(top20[top20['Nivel_Riesgo'] == 'ALTO'])}")
print(f"  - MEDIAS: {len(top20[top20['Nivel_Riesgo'] == 'MEDIO'])}")
print(f"  - Total siniestros: {top20['Num_Siniestros'].sum()}")
print(f"  - Gravedad promedio: {top20['Gravedad_Promedio'].mean():.2f}")
print("\n" + "="*80)
