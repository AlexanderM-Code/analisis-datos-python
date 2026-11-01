# Análisis de datos con Python

Laboratorios y prácticas del curso **Analítica de Datos** (Ingeniería de Software, séptimo semestre, Universidad Cooperativa de Colombia, 2026-2), resueltos en Python con pandas, scikit-learn y TensorFlow.

El recorrido va del dato crudo al modelo: inspección y limpieza, análisis exploratorio, estadística inferencial, regresión, clasificación, agrupamiento, series de tiempo, selección de variables y redes neuronales. Cada notebook está ejecutado, con sus gráficas, resultados y la interpretación escrita de cada paso.

**Autor:** Alexander Mejía Ospina · [Portafolio](https://alexanderm-code.github.io/) · [LinkedIn](https://www.linkedin.com/in/alexander-mejia-ospina)

## Resultados destacados

| Problema | Técnica | Resultado |
|---|---|---|
| Reconocer dígitos escritos a mano (MNIST) | Red neuronal densa con Keras | **97,64 %** de exactitud en 10 000 imágenes de prueba |
| Clasificar especies de Iris | Árbol de decisión elegido con validación cruzada | **96,67 %** en prueba (29 de 30) |
| Pronosticar ventas de 12 meses | Ingenuo estacional frente a métodos constantes | MAE de **9,42** unidades, menos de la mitad que los otros métodos |
| Predecir la progresión de la diabetes | Random Forest con selección de variables | Con 3 de 10 variables, el mismo error que con todas (RMSE 52,8) |
| Detectar tumores malignos (Práctica 2) | Regresión logística | 84,2 % de exactitud con una sola variable, con análisis de los falsos negativos |

## Contenido

Cada enlace abre el notebook en Google Colab.

### 1. Preparación de datos y análisis exploratorio

| Lab | Tema | Qué se trabaja |
|---|---|---|
| 01 | [Inspección del dataset](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/01-preparacion-y-eda/lab01_inspeccion_dataset.ipynb) | Estructura, tipos de datos y valores únicos |
| 02 | [Calidad de los datos](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/01-preparacion-y-eda/lab02_calidad_datos.ipynb) | Faltantes, duplicados y categorías inconsistentes |
| 03 | [Preparación de los datos](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/01-preparacion-y-eda/lab03_preparacion_datos.ipynb) | Limpieza, imputación y estandarización de categorías |
| 04 | [Análisis exploratorio (EDA)](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/01-preparacion-y-eda/lab04_eda.ipynb) | `groupby`, histogramas, barras y boxplots |
| 05 | [Estadística descriptiva](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/01-preparacion-y-eda/lab05_descripcion_estadistica.ipynb) | Media, mediana, dispersión, cuartiles e IQR |
| 06 | [Valores atípicos](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/01-preparacion-y-eda/lab06_valores_atipicos.ipynb) | Criterio del IQR y análisis por categoría |

### 2. Correlación y estadística inferencial

| Lab | Tema | Qué se trabaja |
|---|---|---|
| 07 | [Relaciones y patrones](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/02-correlacion-y-estadistica/lab07_relaciones_y_patrones.ipynb) | Dispersión, Pearson y efecto de un valor extremo |
| 07.1 | [Correlaciones visuales y cuantitativas](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/02-correlacion-y-estadistica/lab07_1_correlaciones_visual_cuantitativo.ipynb) | Matriz de correlaciones y mapa de calor (Auto MPG) |
| 08 | [Pearson frente a Spearman](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/02-correlacion-y-estadistica/lab08_pearson_vs_spearman.ipynb) | Relaciones lineales y monótonas |
| 09 | [Significancia de una correlación](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/02-correlacion-y-estadistica/lab09_significancia_correlacion.ipynb) | Pruebas de hipótesis y p-value |
| 10 | [Distribuciones de probabilidad](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/02-correlacion-y-estadistica/lab10_distribuciones_probabilidad.ipynb) | Binomial, Poisson, exponencial y normal |
| 11 | [Distribución normal y puntuación Z](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/02-correlacion-y-estadistica/lab11_distribucion_normal_z_score.ipynb) | Regla 68-95-99,7 y Z-score |
| 12 | [Intervalos de confianza](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/02-correlacion-y-estadistica/lab12_intervalos_confianza.ipynb) | Error estándar, margen de error y tamaño de muestra |

### 3. Regresión y clasificación

| Lab | Tema | Qué se trabaja |
|---|---|---|
| 13 | [Regresión lineal simple](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/03-regresion-y-clasificacion/lab13_regresion_lineal_simple.ipynb) | Ajuste, residuos, R², MAE y RMSE |
| 14 | [Regresión lineal múltiple](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/03-regresion-y-clasificacion/lab14_regresion_lineal_multiple.ipynb) | Significancia de coeficientes y modelo reducido |
| 15 | [ANOVA de un factor](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/03-regresion-y-clasificacion/lab15_anova_iris.ipynb) | Comparación de medias entre especies |
| 16 | [Regresión logística](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/03-regresion-y-clasificacion/lab16_regresion_logistica_iris.ipynb) | Matriz de confusión, exactitud y recall |
| 17 | [Árbol de regresión](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/03-regresion-y-clasificacion/lab17_arbol_regresion_iris.ipynb) | Profundidad elegida con KFold y diagnóstico del modelo |
| 18 | [Validación cruzada](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/03-regresion-y-clasificacion/lab18_validacion_cruzada_iris.ipynb) | Comparación de árboles con pliegues estratificados |

### 4. Aprendizaje no supervisado y series de tiempo

| Lab | Tema | Qué se trabaja |
|---|---|---|
| 19 | [k-means](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/04-no-supervisado-y-series/lab19_kmeans_iris.ipynb) | Agrupamiento, centroides y método del codo |
| 20 | [Reglas de asociación](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/04-no-supervisado-y-series/lab20_reglas_asociacion.ipynb) | Apriori: soporte, confianza y lift |
| 21 | [Series de tiempo](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/04-no-supervisado-y-series/lab21_series_de_tiempo.ipynb) | Tendencia, estacionalidad, promedio móvil y pronósticos |

### 5. Selección de variables y redes neuronales

| Lab | Tema | Qué se trabaja |
|---|---|---|
| 22 | [Selección de variables con Random Forest](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/05-seleccion-de-variables-y-redes/lab22_seleccion_variables_random_forest.ipynb) | Filtrado, wrapper e importancias del modelo |
| 23 | [Redes neuronales con MNIST](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/05-seleccion-de-variables-y-redes/lab23_mnist_redes_neuronales.ipynb) | Keras, subajuste, sobreajuste, Dropout y parada temprana |

### Prácticas evaluativas

| Práctica | Tema | Qué se trabaja |
|---|---|---|
| 1 | [Análisis exploratorio y estadístico](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/practicas-evaluativas/practica1_evaluativa.ipynb) | Análisis completo de un dataset de ventas minoristas (Kaggle), de la calidad de los datos a la correlación |
| 2 | [Auto MPG](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/practicas-evaluativas/practica2_autompg.ipynb) | Regresión simple y múltiple, VIF y ANOVA |
| 2 | [Regresión logística](https://colab.research.google.com/github/AlexanderM-Code/analisis-datos-python/blob/main/practicas-evaluativas/practica2_logistica.ipynb) | Clasificación de tumores (Breast Cancer Wisconsin) |

## Herramientas

Python · pandas · NumPy · Matplotlib · seaborn · SciPy · statsmodels · scikit-learn · mlxtend · TensorFlow/Keras · Google Colab

## Cómo ejecutarlos

La forma más sencilla es abrir cualquier notebook en Colab con los enlaces de arriba.

- Los laboratorios con **Iris, Diabetes, MNIST** y los datos simulados (19 a 23) cargan sus datos solos y se ejecutan sin archivos extra.
- Los demás leen un CSV desde `/content/` (`ventas.csv`, `auto-mpg.csv`, `datos_distribuciones.csv`, etc.). Son los datasets entregados en el curso; súbelos a Colab antes de ejecutar. `auto-mpg.csv` es público en [UCI / Kaggle](https://www.kaggle.com/datasets/uciml/autompg-dataset).

Para ejecutarlos en local:

```bash
pip install -r requirements.txt
jupyter notebook
```

## Créditos

- Las guías de los laboratorios fueron diseñadas por el docente **Luis Miguel Gómez Meneses** para el curso. El código, la ejecución y las respuestas de cada notebook son míos.
- La **Práctica Evaluativa 2** la hice en pareja con **Santiago Rodríguez Jiménez**.
