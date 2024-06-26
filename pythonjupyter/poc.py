import os
import openai
import speech_recognition as sr
from textblob import TextBlob
import pandas as pd
from transformers import pipeline

# Initialize recognizer
recognizer = sr.Recognizer()

# Function to transcribe audio to text
def transcribe_audio(file_path):
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Audio not clear"
    except sr.RequestError as e:
        return "Could not request results; {0}".format(e)

# Initialize emotion classifier (using a pre-trained model)
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Function to analyze emotion using pre-trained model
def analyze_emotion(text):
    emotions = emotion_classifier(text)
    return emotions[0]['label']

# Function to detect concerns using OpenAI GPT-3
def detect_concerns(text):
    openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your OpenAI API key
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Identify any concerns in the following text: {text}",
        max_tokens=50
    )
    concerns = response.choices[0].text.strip().split(', ')
    return concerns

# Example audio files and corresponding texts
audio_files = ['/home/anirudh/Downloads/Television.wav', '/home/anirudh/Downloads/Adver.wav', '/home/anirudh/Downloads/Television.wav', '/home/anirudh/Downloads/Adver.wav']

# Creating the POC table
poc_data = {
    'File': [],
    'Text': [],
    'Sentiment': [],
    'Emotion': [],
    'Concern 1': [],
    'Concern 2': [],
    'Concern 3': []
}

for file in audio_files:
    # Transcribe audio
    text = transcribe_audio(file)
    
    # Analyze sentiment
    polarity, subjectivity = analyze_sentiment(text)
    sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
    
    # Analyze emotion
    emotion = analyze_emotion(text)
    
    # Detect concerns using GPT-3
    concerns = detect_concerns(text)
    
    # Add to POC data
    poc_data['File'].append(file)
    poc_data['Text'].append(text)
    poc_data['Sentiment'].append(sentiment)
    poc_data['Emotion'].append(emotion)
    poc_data['Concern 1'].append(concerns[0] if len(concerns) > 0 else '')
    poc_data['Concern 2'].append(concerns[1] if len(concerns) > 1 else '')
    poc_data['Concern 3'].append(concerns[2] if len(concerns) > 2 else '')

# Create DataFrame
df = pd.DataFrame(poc_data)

# Display DataFrame
print(df)

# Optionally, save DataFrame to a CSV file
df.to_csv('voice_sentiment_analysis_poc.csv', index=False)
