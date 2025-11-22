"""
Convertir mapas HTML interactivos a imagenes PNG para inclusion en informes
Usa selenium + webdriver para capturar screenshots
"""

print("="*80)
print("CONVERSION DE MAPAS HTML A IMAGENES PNG")
print("="*80)

print("\nEste script requiere instalar selenium y un webdriver.")
print("\nOPCION 1 - MANUAL (Recomendada):")
print("  1. Abre cada mapa HTML en tu navegador:")
print("     - mapa_01_heatmap_siniestros.html")
print("     - mapa_02_siniestros_pois.html")
print("     - mapa_03_nocturnos_bares.html")
print("\n  2. Ajusta el zoom para mostrar todo Bogota")
print("\n  3. Presiona F12 (herramientas desarrollador)")
print("\n  4. Usa la opcion 'Captura de pantalla completa' del navegador:")
print("     - Chrome: Ctrl+Shift+P > 'Capture full size screenshot'")
print("     - Firefox: Click derecho > 'Take a Screenshot' > 'Save full page'")
print("\n  5. Guarda las imagenes como:")
print("     - mapa_01_heatmap_siniestros.png")
print("     - mapa_02_siniestros_pois.png")
print("     - mapa_03_nocturnos_bares.png")
print("     En: data/processed/eda_visualizaciones/")

print("\n" + "-"*80)
print("\nOPCION 2 - AUTOMATICA (Requiere instalacion):")
print("  pip install selenium webdriver-manager")
print("  Luego ejecuta: python src/convert_maps_automated.py")

print("\n" + "-"*80)
print("\nOPCION 3 - PARA EL INFORME:")
print("  En lugar de incluir el mapa completo, puedes:")
print("  1. Incluir las capturas PNG en Word/PDF")
print("  2. Agregar un enlace o codigo QR al archivo HTML")
print("  3. Mencionar: 'Ver mapa interactivo en mapa_XX.html'")

print("\n" + "="*80)
print("ALTERNATIVA: Ya tienes 7 graficos estaticos PNG que SI se pueden incluir directamente")
print("="*80)
