# Análisis Espacial de Accidentalidad Vial en Bogotá (2015-2020)

## 📋 Descripción del Proyecto

Análisis espacial de 196,152 siniestros viales en Bogotá usando técnicas de clustering (DBSCAN) para identificar zonas críticas y priorizar intervenciones de seguridad vial.

**Autor:** Camilo Peñuela Espinosa
**Institución:** Pontificia Universidad Javeriana
**Período de análisis:** 2015-2020 (6 años)

## 🎯 Objetivos

- Identificar zonas geográficas críticas de accidentalidad grave
- Aplicar análisis espacial (Moran's I, KDE, DBSCAN)
- Calcular score de riesgo para priorización
- Generar dashboard interactivo para toma de decisiones

## 🔑 Hallazgos Principales

- **98.2% de siniestros graves** concentrados en Cluster 0 (macro-corredor occidental)
- **Top 20 intersecciones** representan 1.9% del total pero 18.3% de siniestros
- **ROI estimado:** 79× (intervención de $300K → beneficio $23.75M/año)
- **Av. Boyacá:** 9,367 siniestros (corredor más crítico)

## 📊 Metodología

1. **Preprocesamiento:** Muestreo estratificado de 30,014 registros (15.3%)
2. **Análisis Espacial:**
   - Moran's I (autocorrelación espacial, K=8)
   - KDE (identificación de hotspots)
   - DBSCAN (clustering, ε=1km, min_samples=20)
3. **Scoring:** Fórmula F1 = Densidad × Gravedad
4. **Visualización:** Dashboard interactivo con 3 capas

## 🛠️ Tecnologías

- Python 3.13
- Pandas, NumPy, GeoPandas
- Scikit-learn, PySAL (esda)
- Folium (mapas interactivos)
- Matplotlib, Seaborn

## 📁 Estructura del Proyecto

```
├── data/
│   ├── processed/          # Datos procesados
│   └── raw/                # Datos originales (no incluidos por tamaño)
├── seccion_8_implementacion/
│   └── codigo/             # Scripts de análisis espacial
├── seccion_9_resultados/
│   └── visualizaciones/    # Mapas y gráficos
├── seccion_11_dashboard/
│   ├── dashboard_interactivo.html
│   └── reporte_top20.csv
├── PRESENTACION_PROYECTO_COMPLETO.md
├── CORRECCIONES_COORDENADAS.md
└── README.md
```

## 🚀 Uso

### Instalar dependencias

```bash
pip install pandas numpy geopandas scikit-learn folium matplotlib seaborn libpysal esda scipy
```

### Ejecutar análisis

```bash
# Test de representatividad de muestra
python test_chi_cuadrado.py

# Generar visualizaciones de Moran's I
python generar_moran_scatterplot.py

# Crear mapa KDE
python crear_kde_mejorado.py
```

### Ver resultados

Abrir en navegador:
- `seccion_11_dashboard/dashboard_interactivo.html` (Dashboard completo)
- `seccion_9_resultados/visualizaciones/9.4_kde_hotspots.html` (Mapa KDE)
- `seccion_9_resultados/visualizaciones/9.6_top20_intersecciones.html` (Top 20)

## 📈 Resultados Clave

### Clusters Identificados (DBSCAN)

| Cluster | Siniestros | Área (km²) | Densidad | Score | Nivel |
|---------|------------|-----------|----------|-------|-------|
| 0       | 29,405     | 47.2      | 623.0    | 100.0 | 🔴 CRÍTICO |
| 1-4     | 720        | 17.1      | 42.1     | 20.4  | 🟢 BAJO |

### Top 5 Corredores

1. **Av. Boyacá:** 9,367 siniestros (31.7%)
2. **Calle 26:** 2,864 siniestros (9.7%)
3. **Av. Las Américas:** 2,672 siniestros (9.0%)
4. **Calle 80:** 1,971 siniestros (6.7%)
5. **Autopista Norte:** 1,937 siniestros (6.6%)

## 📝 Documentación

- [Presentación completa](PRESENTACION_PROYECTO_COMPLETO.md)
- [Correcciones de coordenadas](CORRECCIONES_COORDENADAS.md)
- PDFs de entregas: `Entrega 1.pdf`, `SEGUNDA ENTREGA SECCIONES 5,6,7.pdf`, `TERCERA ENTREGA.pdf`

## 📊 Validaciones

- **Chi-cuadrado:** p=1.000 (muestra estratificada representativa)
- **Moran's I:** 0.058 (p<0.001, z=22.5) - autocorrelación significativa
- **Silhouette:** -0.07 (esperado para corredores continuos)
- **Calinski-Harabasz:** 403.85 (alta cohesión interna)

## 🤝 Contribuciones

Este es un proyecto académico. Para sugerencias o consultas, contactar a través de GitHub Issues.

## 📄 Licencia

Proyecto académico - Pontificia Universidad Javeriana

## 🙏 Agradecimientos

- Secretaría de Movilidad de Bogotá (datos públicos)
- ANSV - Agencia Nacional de Seguridad Vial

---

**Fecha de última actualización:** Noviembre 2025
