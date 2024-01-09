import json
import pandas as pd

with open("taylor_swift_spotify.json") as file:
    data = json.load(file)

# Generar csv
album_i = []
for album in data['albums']:

    # track info
    no_audiofeatures = pd.DataFrame(album['tracks']).drop('audio_features',axis = 1)
    audiofeatures_empty = []
    for track in album['tracks']:
        audiofeatures_empty.append(track['audio_features'].values())
        audiofeatures = pd.DataFrame(audiofeatures_empty)
        audiofeatures.columns = 'audio_features.'+ pd.DataFrame(track['audio_features'].keys())[0]

    # artist info
    artist_values_df = pd.concat([pd.Series(data['artist_id']) ,pd.Series(data['artist_name']),pd.Series(data['artist_popularity'])] , axis = 1)
    artist_values_sequence = artist_values_df.loc[artist_values_df.index.repeat(len(audiofeatures))].reset_index(drop=True)
    artist_values_sequence.columns = ['artist_id','artist_name','artist_popularity']

    # album info
    album_values_df = pd.DataFrame([album['album_id'], album['album_name'], album['album_release_date'], album['album_total_tracks']]).transpose()
    album_values_sequence = album_values_df.loc[album_values_df.index.repeat(len(audiofeatures))].reset_index(drop=True)
    album_values_sequence.columns = ['album_id','album_name','album_release_date','album_total_tracks']


    album_i.append(pd.concat([no_audiofeatures, audiofeatures, artist_values_sequence,album_values_sequence], axis=1))


dataset = pd.concat(album_i)

# Export csv
dataset.to_csv('dataset.csv',index=False)