# PI01-Individual-MLOPS
# Proyecto de Análisis de Datos de STEAM
![image](https://github.com/vay0/PI01-Individual-MLOPS/assets/105746281/1cfd2cf3-88fd-4af0-868f-763b39fbc671)


En este proyecto de MLOPS, se me ha solicitado desempeñar el papel de un Data Scientist en Steam, una plataforma multinacional de videojuegos. 
Dado que el conjunto de datos presenta datos poco estructurados (anidados, con columnas que contienen datos nulos, entre otros problemas), se 
requiere realizar rápidamente el trabajo de un Data Engineer y crear un MVP (Producto Mínimo Viable) para cerrar el proyecto. El proyecto implica
la realización del proceso de ETL, desarrollo de una API, deployment, análisis exploratorio de datos y construcción de un modelo de aprendizaje
automático con el objetivo de crear un sistema de recomendación.

## Estructura de Archivos

### Datasets: 
En el repositorio se encuentran los siguientes datasets necesarios para el desarrollo de la API y el modelo de recomendación:
  - Recomendacion.parquet
  - Reviews.parquet
  - games.parquet
  - general.xlsx
  - generos.parquet
  - items.parquet

- **Diccionario de Datos STEAM.xlsx:** Este archivo de Excel contiene la descripción de cada columna de cada uno de los datasets.

### Archivos de Python

- **EDA y ML:** Este archivo contiene el proceso de Análisis Exploratorio de Datos (EDA). Aquí se exploran los datos, se realizan visualizaciones
  y se extrae información para comprender mejor el conjunto de datos. Además, también se encuentra el desarrollo del modelo de recomendación solicitado.

- **Etl y funciones:** Este archivo detalla el proceso de Extracción, Transformación y Carga de datos (ETL), donde se obtienen los datos y se transforman
  de acuerdo con las necesidades del proyecto para su posterior análisis. Además, este archivo también contiene las funciones necesarias para la API.

### Main.py - API

El archivo `main.py` contiene el código de la API web desarrollada con FastAPI. Esta API ofrece endpoints para consultar y analizar datos relacionados con
juegos y usuarios.
Este solo contiene los endpoints de las funciones: 
  - UsersRecommend
  - UsersNotRecommend
  - sentiment_analysis
