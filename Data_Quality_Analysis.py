import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv')

# diccionario con tipos de variables y sus rangos
data_type =  {
                'disc_number': int, # min 1 (cantidad de discos)
                'duration_ms': int, # >0
                'explicit': bool, # True or False
                'track_number':int,
                'track_popularity':int, #0 - 100
                'track_id':str,
                'track_name':str,
                'audio_features.danceability':float, # 0 - 1
                'audio_features.energy':float, # 0 - 1
                'audio_features.key': int, # -1 - 11
                'audio_features.loudness':float, # usualmente entre -60 y 0  db
                'audio_features.mode':int, # 1 (major) o 0 (minor)
                'audio_features.speechiness':float, # max 1 (¿0 podria ser ausencia de palabras?)
                'audio_features.acousticness':float, # 0 - 1
                'audio_features.instrumentalness':float, # max 1 (¿0 podria ser contenido vocal en la cancion?)
                'audio_features.liveness':float, # max 1, ¿min 0?
                'audio_features.valence':float, # 0 - 1
                'audio_features.tempo':float, # positivos
                'audio_features.id':str,
                'audio_features.time_signature':int, # 3 - 7
                'artist_id':str,
                'artist_name':str,
                'artist_popularity':int, # 0 - 100
                'album_id':str,
                'album_name':str, # si se ha eliminado el album, no tiene nombre
                'album_release_date':str, # ejemplo: "1981-12"
                'album_total_tracks':int  # >0
}


print(data.shape) # 539 registros y 27 columnas

## Comparacion de tipo de variable
type_comparison = pd.concat([pd.DataFrame(data.dtypes),pd.DataFrame.from_dict(data_type,orient='index')],axis=1)
type_comparison.columns = ['Dataset types', 'Correct data type']
print(type_comparison)

## Columnas con valores nulos y su conteo
print(data.isnull().sum()[data.isnull().sum() > 0])

## Verificacion de duplicados
data_unique = data.drop_duplicates()
print(len(data)-len(data_unique)) # 18 duplicados



### Validacion de rangos y tipo de variable (exhaustivo) ###

## 'disc_number'
data.value_counts('disc_number') # variable int, sin anomalias

## 'duration_ms'
data['duration_ms'].describe() # variable int
print(data[~data['duration_ms']>0])  # 2 canciones con duracion negativa
print(data[data['duration_ms'] <= 83253]) # canciones con poca duracion


## 'explicit'
print(data[~data['explicit'].isin(['True','False'])]) # variable deberia ser bool, pero hay 5 canciones con valor no booleano en 'explicit'

## 'track_number'
data['track_number'].describe() # variable int
sns.histplot(data=data, x='track_number') ; plt.show() # sin anomalias

## 'track_popularity'
print(data[~np.logical_and(data['track_popularity']>=0, data['track_popularity']<=100)]) # variable int, 7 track_popularity fuera de rango (6 negativas, 1 arriba de 100)

## 'track_id'
print(data.value_counts('track_id')[data.value_counts('track_id')>1]) # variable str, 19 track_id duplicados
print(data[data['track_id'].isin(list(data.value_counts('track_id')[data.value_counts('track_id')>1].index))])
print(data[data['track_id']=='2YWtcWi3a83pdEg3Gif4Pd']) # el segundo en la lista es uno de los que tiene error en 'explicit'

## 'track_name'
non_unique_track_names = list(data.value_counts('track_name')[data.value_counts('track_name')>1].index) # variable str, algunas canciones pertenecen a diferentes albums, aunque hay duplicados
copy_non_unique_track = data[data['track_name'].isin(non_unique_track_names)] # 352 canciones que se registran mas de una vez
copy_non_unique_track.drop_duplicates() #334 al remover 18 duplicados

## 'audio_features.danceability'
print(data[~np.logical_and(data['audio_features.danceability']>=0, data['audio_features.danceability']<=1)]) # variable float, todos entre 0 y 1, solo 2 NaN

## 'audio_features.energy'
print(data[~np.logical_and(data['audio_features.energy']>=0, data['audio_features.energy']<=1)]) # variable float, todos entre 0 y 1, solo 2 NaN

## 'audio_features.key'
print(data[~np.logical_and(data['audio_features.key']>=-1, data['audio_features.key']<=11)]) # solo se evidencia una cancion con audio_features.key NaN
print(data.value_counts('audio_features.key')) # variable deberia ser int, pero toma el tipo float porque los valores tiene punto decimal y un cero

## 'audio_features.loudness'
out_range_audio_features_loudness = data[~np.logical_and(data['audio_features.loudness']>=-60, data['audio_features.loudness']<=0)]
print(out_range_audio_features_loudness['audio_features.loudness'])  # variable float, dentro del rango tipico y solo 2 NaN

## 'audio_features.mode'
print(data[~data['audio_features.mode'].isin([1,0])]) # variable int, sin anomalias
print(data.value_counts('audio_features.mode'))

## 'audio_features.speechiness'
sns.boxplot(data=data, x= 'audio_features.speechiness') ; plt.show() # variable float, valores atipicos pero dentro del rango
print(data['audio_features.speechiness'].describe()) # los valores no superan 1 y tampoco son negativos

## 'audio_features.acousticness'
af_acousticness_anomalies_index = ~np.logical_and(data['audio_features.acousticness']>=0, data['audio_features.acousticness']<=1)
print(data[af_acousticness_anomalies_index][['track_id','track_name','audio_features.acousticness']]) # variable float, 2 negativos, 3 arriba de 1 y 1 NaN

## 'audio_features.instrumentalness'
print(data['audio_features.instrumentalness']) # parece ser notacion cientifica, pero el tipo de variable es str y no float
#data['audio_features.instrumentalness'].astype(float) #ERROR: hay un string '7.28x-06', x en lugar de e (index 524)
#data.iloc[524] # registro con anomalia
data_af_instrumentalness_copy = pd.DataFrame(data['audio_features.instrumentalness']) # copia para hacer analisis
data_af_instrumentalness_copy.iloc[524] = 7.28e-06  # se corrige en la copia para poder verificar rangos
data_af_instrumentalness_copy = data_af_instrumentalness_copy.astype(float) # sin anomalias, los valores no superan 1
print(data_af_instrumentalness_copy.describe())

## 'audio_features.liveness'
print(data['audio_features.liveness'].describe()) # variable float, sin anomalias

## 'audio_features.valence'
print(data['audio_features.valence'].describe()) # variable float, sin anomalias

## 'audio_features.tempo'
print(data['audio_features.tempo'].describe()) # variable float positiva, sin anomalias

## 'audio_features.id'
audio_features_id_duplicates = list((data.value_counts('audio_features.id')[data.value_counts('audio_features.id')>1]).index) #lista de 20 audio_features.id duplicados
audio_features_id_duplicates_complete_data = data[data['audio_features.id'].isin(audio_features_id_duplicates)] # lista de canciones con audio_features.id duplicados
audio_features_id_duplicates_complete_data.drop_duplicates() # Cruel Summer es duplicado, la diferencia es un error en explicit
                                                             #Gorgeous es duplicado, la diferencia es un error en track_id
print(data[~(data['track_id'] == data['audio_features.id'])]) # Verificacion de igualdad entre 'track_id' y 'audio_features.id'

## 'audio_features.time_signature'
print(data['audio_features.time_signature']) # variable float, deberia ser int
print(data.value_counts('audio_features.time_signature')) # sin anomalias

## 'artist_id'
print(data.value_counts('artist_id')) # sin anomalias

## 'artist_name'
print(data.value_counts('artist_name')) # sin anomalias

## 'artist_popularity'
print(data.value_counts('artist_popularity')) # todos los artist_popularity estan arriba de 100 y el maximo debe ser 100, anomalia

## 'album_id'
album_id_and_total_tracks_count = (data[['album_id','album_total_tracks']].value_counts()).reset_index() # conteo de album_id y el numero total de canciones estipulado
album_id_and_total_tracks_count.columns = ['album_id','album_total_tracks','album_id_count']
album_id_and_total_tracks_count.iloc[21,1] = 13 # correccion en copia con 'album_total_tracks' solo para analizar (Thirteen a 13)
inconsistencies_album_id_and_total_tracks_count = ~(album_id_and_total_tracks_count.iloc[:,1].astype(int) == album_id_and_total_tracks_count.iloc[:,2])
print(album_id_and_total_tracks_count[inconsistencies_album_id_and_total_tracks_count]) # 5 albums con inconsitencias en el numero de canciones

# print(data[data['album_id']=='1NAmidJlEaVgA3MpcPFYGq']) # album_total_tracks = 18, album_id_count = 36 -> album duplicado
# print(data[data['album_id']=='6kZ42qRrzov54LcAk4onW9']) #album_total_tracks = 34, album_id_count = 30 -> faltan canciones o album_total_tracks tiene una anomalia
# print(data[data['album_id']=='6DEjYFkNZh67HP7R9PSZvv']) #album_total_tracks = 15, album_id_count = 16 -> Gorgeous es duplicado, uno de ellos tiene NaN en track_id
# print(data[data['album_id']=='2Xoteh7uEpea4TohMxjtaq']) #album_total_tracks = 10, album_id_count = 15 -> podria haber un error en album_total_tracks, no hay duplicados
# print(data[data['album_id']=='5eyZZoQEFQWRHkV2xgAeBw']) #album_total_tracks = 13, album_id_count = 15 -> podria haber un error en album_total_tracks, no hay duplicados

## 'album_name'
print(data[data['album_name'].isnull()]['album_id'].value_counts()) # 'album_id' de los album sin nombre
# print(data[data['album_id']=='1MPAXuTVL2Ej5x0JHiSPq8']) # sin canciones duplicadas en el album sin nombre (error o el album fue eliminado)
# print(data[data['album_id']=='6fyR4wBPwLHKcRtxgd4sGh']) # sin canciones duplicadas en el album sin nombre (error o el album fue eliminado)

## 'album_release_date'
print(data['album_release_date'].value_counts())
# print(data[data['album_release_date']== '1989-10-24']) # Un album tiene fecha de lanzamiento el 1989-10-24 y es antes del nacimiento del artista (1989-12-13)
# print(data[data['album_release_date']== '2027-05-26']) # Un album tiene fecha de lanzamiento el 2027-05-26, indicando un error debido a que no hemos llegado al año 2027


## 'album_total_tracks'
print(data['album_total_tracks']) # el tipo de variable no es int, debido a que escribieron en letras el numero de canciones del album 'Taylor Swift'
print(data[data['album_total_tracks']=='Thirteen']) #dicho album coincide con el album que tiene fecha de lanzamiento errada


### Graficas

## Histogramas de las variables numericas
#for i in data.select_dtypes(include=np.number).columns:
#    plt.figure()
#    sns.histplot(data[i])


## Presencia de contenido vocal vs Presencia de contenido instrumental
plt.figure()
plt.scatter(x=data['audio_features.speechiness'],y=data_af_instrumentalness_copy)
plt.xlabel('audio_features.speechiness');plt.ylabel('audio_features.instrumentalness')
plt.show() # Parece normal, tiene sentido
