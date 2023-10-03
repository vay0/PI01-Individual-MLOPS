from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI(title= 'STEAM',
              description= 'El objetivo de esta API es mostrar los resultados para las siguientes funciones a partir de la bases de datos de STEAM')

df_games = pd.read_parquet('games.parquet')
df_items = pd.read_parquet('items.parquet')
df_reviews = pd.read_parquet('Reviews.parquet')
df_generos = pd.read_parquet('generos.parquet')
df = pd.read_parquet('Recomendacion.parquet')




@app.get('/UsersRecommend')
def UsersRecommend(year:int):
    """
    Según el año de publicacion, se devuelve una lista con el top 3 de los juegos mas recomendados
    """
    juego_usuario = pd.merge(df_reviews, df_games, left_on='item_id', right_on='id', how='inner')
    juego_usuario['posted'] = juego_usuario['posted'].astype(str)
    juego_usuario['year'] = juego_usuario['posted'].str.split('-').str[0].astype(int)
    
    if year not in juego_usuario['year'].values:
        no_encontrado = f'El año {year} no fue encontrado en la base de datos'
        return no_encontrado
    
    else:
        juego_usuario = juego_usuario[juego_usuario['year'] == year]
        juego_usuario_filtrado = juego_usuario[juego_usuario['sentiment'].isin([1, 2])]
        resultados = juego_usuario_filtrado.groupby(['year','title']).agg({'recommend': 'sum', 'sentiment': 'count'})
        resultados.rename(columns={'recommend': 'total_recomendaciones', 'sentiment': 'total_sentimientos'}, inplace=True)
        resultados['total'] = resultados['total_recomendaciones'] + resultados['total_sentimientos']
        resultado_sorted = resultados.sort_values(by=['total'], ascending=[False])
        recomendados = resultado_sorted.head(3)
        recomendados_list = [
            f'Puesto {i + 1}: {juego[1]}'
            for i, juego in enumerate(recomendados.index)
        ]

        return f'Para el año {year}, los juegos recomendados son: {", ".join(recomendados_list)}'
  
@app.get('/UsersNotRecommend')
def UsersNotRecommend(year: int):
    """
    Según el año de publicacion, se devuelve una lista con el top 3 de los juegos menos recomendados
    """
    juego_usuario = pd.merge(df_reviews, df_games, left_on='item_id', right_on='id', how='inner')
    juego_usuario['posted'] = juego_usuario['posted'].astype(str)
    juego_usuario['year'] = juego_usuario['posted'].str.split('-').str[0].astype(int)
    juego_usuario_filtrado = juego_usuario[(juego_usuario['sentiment'] == 0) & (juego_usuario['recommend'] == False)]
    
    if year not in juego_usuario_filtrado['year'].values:
        no_encontrado = f'El año {year} no fue encontrado en la base de datos'
        return no_encontrado
    
    else:
        juego_usuario_filtrado = juego_usuario_filtrado[juego_usuario_filtrado['year'] == year]
        resultados = juego_usuario_filtrado.groupby(['year','title']).agg({'recommend': 'count', 'sentiment': 'count'})
        resultados.rename(columns={'recommend': 'total_norecomendados', 'sentiment': 'total_sentimientos'}, inplace=True)
        resultados['total'] = resultados['total_norecomendados'] + resultados['total_sentimientos']
        resultado_sorted = resultados.sort_values(by=['year','total'], ascending=[True, False])
        no_recomendados = resultado_sorted.head(3)
        no_recomendados_list = [
            f'Puesto {i + 1}: {juego[1]}'
            for i, juego in enumerate(no_recomendados.index)
        ]

        return f'Para el año {year}, los juegos no recomendados son: {", ".join(no_recomendados_list)}'
  
@app.get('/sentiment_analysis')
def sentiment_analysis(year: int):
  """
  Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios
  que se encuentren categorizados con un análisis de sentimiento.
  """
  games_reviews = pd.merge(df_games, df_reviews, left_on='id', right_on='item_id')
  sentimiento = games_reviews[['release_date', 'sentiment']]
  sentimiento = sentimiento.dropna(subset=['release_date'])  # Eliminar filas con release_date nulos
  
  # Convertir release_date a cadena y manejar NaN
  sentimiento['release_date'] = sentimiento['release_date'].astype(str)
  sentimiento['year'] = sentimiento['release_date'].str.split('-').str[0].astype(int)
  filtro_year = sentimiento[sentimiento['year'] == year]
  
  if year in filtro_year['year'].values:
      sentiment_mapping = {2: 'positivo', 1: 'neutro', 0: 'negativo'}
      filtro_year['sentiment'] = filtro_year['sentiment'].map(sentiment_mapping)
      sentiment_counts = filtro_year['sentiment'].value_counts().to_dict()
      
      return sentiment_counts
  
  else:
      no_encontrado = f'El año {year} no fue encontrado en la base de datos'
      return no_encontrado
  
  
  # http://127.0.0.1:8000
