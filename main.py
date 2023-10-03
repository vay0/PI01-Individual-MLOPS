from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI(title= 'STEAM',
              description= 'El objetivo de esta API es mostrar los resultados para las siguientes funciones a partir de la bases de datos de STEAM')

df_games = pd.read_parquet('Datasets\games.parquet')
df_items = pd.read_parquet('Datasets\items.parquet')
df_reviews = pd.read_parquet('Datasets\Reviews.parquet')
df_generos = pd.read_parquet('Datasets\generos.parquet')
df = pd.read_parquet('Datasets\Recomendacion.parquet')


@app.get('/PlayTimeGenre')
def PlayTimeGenre(genero: str):
    genero = genero.lower()
    # Filtrar el DataFrame df_generos por género
    genero_data = df_generos[df_generos['genres'] == genero]
    
    if genero_data.empty:
        return f'El género {genero} no fue encontrado en la base de datos'
    
    # Obtener los ID de los juegos que pertenecen al género
    juego_ids = genero_data['id'].tolist()
    
    # Filtrar df_items por los ID de juego y combinar con df_games
    generos_juegos = df_items[df_items['item_id'].isin(juego_ids)].merge(df_games, left_on='item_id', right_on='id')
    
    # Calcular el año con la máxima cantidad de horas de juego
    generos_juegos['release_date'] = generos_juegos['release_date'].astype(str)
    generos_juegos['year'] = generos_juegos['release_date'].str.split('-').str[0].astype(int)
    
    max_playtime_year = generos_juegos.groupby('year')['playtime_forever'].sum().idxmax()
    
    año_playtime = f'El año de lanzamiento con mas horas de juego segun el genero {genero} es {max_playtime_year}'
    return año_playtime

@app.get('/userforgenre')
def userforgenre(genero: str):
    """
    """
    genero = genero.lower()
    # Combinar DataFrames
    generos_usuario = pd.merge(df_generos, df_items, left_on='id', right_on='item_id', how='inner')
    generos_usuario = pd.merge(generos_usuario, df_games, left_on='id', right_on='id', how='inner')

    # Filtrar por género
    generos_usuario = generos_usuario[generos_usuario['genres'] == genero]

    if generos_usuario.empty:
        return f'El género {genero} no fue encontrado en la base de datos'

    # Usuario con mayor playtime
    generos_usuario['playtime_forever'] = generos_usuario['playtime_forever'] / 60
    generos_usuario['release_date'] = generos_usuario['release_date'].astype(str)
    generos_usuario['year'] = generos_usuario['release_date'].str.split('-').str[0].astype(int)
    horas_playtime = generos_usuario.groupby(['genres', 'user_id'])[['playtime_forever']].sum()
    indices_max_playtime = horas_playtime.groupby('genres')['playtime_forever'].idxmax()
    usuarios_por_genero = horas_playtime.loc[indices_max_playtime]
    generos_filtrados = generos_usuario[generos_usuario['user_id'].isin(usuarios_por_genero.index.get_level_values('user_id'))]
    playtime_por_genero_usuario_y_anio = generos_filtrados.groupby(['genres', 'user_id', 'year'])[['playtime_forever']].sum()
    
    for (genre, user_id), data in playtime_por_genero_usuario_y_anio.groupby(level=[0, 1]):
        horas_por_anio = data['playtime_forever'].reset_index()
        lista_horas = [{"Año": int(anio), "Horas": horas} for anio, horas in zip(horas_por_anio['year'], horas_por_anio['playtime_forever'])]
        lista_horas_str = str(lista_horas).replace("'", "")
        mensaje = f'Para el género {genre} el usuario {user_id} tiene más horas de juego donde "Horas jugadas":{lista_horas_str}'
  
    return mensaje

@app.get('/UsersRecommend')
def UsersRecommend(year:int):
    """
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
  
@app.get('/recomendacion_juego')
def recomendacion_juego(id):
    if id not in df['id'].values:
        no_encontrado = f'El id {id} no fue encontrado en la base de datos'
        return no_encontrado
    
    else:
        # Crea una matriz TF-IDF para los géneros de los juegos
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Cambia 500 al número deseado de características
        tfidf_matrix = tfidf_vectorizer.fit_transform(grouped_genres_title['genres_text'])
        
        # Calcula la similitud del coseno entre los juegos bajo demanda
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Encuentra el índice del juego con el ID dado
        idx = df[df['id'] == id].index[0]

        # Obtén las puntuaciones de similitud del coseno para todos los juegos
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Ordena los juegos en función de las puntuaciones de similitud
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Obtén los índices de los juegos más similares (excluyendo el juego de entrada)
        sim_scores = sim_scores[1:6]  # Cambia esto si quieres más o menos recomendaciones

        # Obtiene los títulos de los juegos recomendados
        game_indices = [i[0] for i in sim_scores]
        recommended_game_titles = df.iloc[game_indices]['title']

        # Convierte la lista de títulos en una lista plana y única
        recommended_game_titles_flat = list(set([title for sublist in recommended_game_titles for title in sublist]))

        return recommended_game_titles_flat
  
  # http://127.0.0.1:8000
