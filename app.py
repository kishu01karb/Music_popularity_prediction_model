import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open('music_predictor.pkl','rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler']

# Load model and scaler at the start
model, scaler = load_model()  # ← Added: need to load before using

# Define features list
features = [
    'Danceability', 'Energy', 'Loudness', 'Speechiness', 
    'Acousticness', 'Instrumentalness', 'Liveness', 
    'Valence', 'Tempo', 'Intensity', 'dance_floor_score', 
    'mood_score', 'organic_sound'
]  

#create a function to predict any song
def predict_song_popularity(song_features):
    """
    Predict if a song will be a HIT OR NOT HIT
    song_features:dict with feature values
    """
    # create Dataframe with features
    song_df = pd.DataFrame([song_features])
    
    #add engineered features
    song_df['Intensity'] = song_df['Energy']*(song_df['Loudness']+60)/60
    song_df['dance_floor_score'] = song_df['Danceability'] * (song_df['Tempo']/120)
    song_df['mood_score']= song_df['Valence'] * (song_df['Mode']+1)
    song_df['organic_sound'] = (song_df['Acousticness'] + song_df['Instrumentalness'])/2
    
    #select and scale features
    song_features_scaled = scaler.transform(song_df[features])
    
    #predict
    prediction = model.predict(song_features_scaled)[0]
    probabilities = model.predict_proba(song_features_scaled)[0]
    
    return prediction , probabilities

#main App
st.title("🎵 Music Hit Predictor")
col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, 0.7)
    energy = st.slider("Energy", 0.0, 1.0, 0.7)
    loudness = st.slider("Loudness", -60.0, 0.0, -5.0)
    tempo = st.slider("Tempo", 60.0, 200.0, 120.0)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)

with col2:
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15)
    mode = st.selectbox("Mode", [0, 1])

song = {'Danceability': danceability, 'Energy': energy, 'Loudness': loudness,
        'Speechiness': speechiness, 'Acousticness': acousticness, 
        'Instrumentalness': instrumentalness, 'Liveness': liveness,
        'Valence': valence, 'Tempo': tempo, 'Mode': mode}

if st.button("🎯 Predict"):
    prediction, probs = predict_song_popularity(song)
    
    if prediction == "HIT":
        st.success(f"🎉 {prediction}!")
        st.balloons()
    else:
        st.error(f"📉 {prediction}")
    
    st.write(f"**Confidence:** {probs[0]:.1%}")
    st.progress(probs[0])
