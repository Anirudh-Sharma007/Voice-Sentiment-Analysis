import speech_recognition as sr
from textblob import TextBlob

# Load an audio file
audio_file = '/home/anirudh/Downloads/Television.wav'

# Initialize the recognizer
recognizer = sr.Recognizer()

# Load the audio file
with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)

# Recognize speech using Google Web Speech API
try:
    text = recognizer.recognize_google(audio_data)
    print("Recognized Text:", text)
    
    # Perform sentiment analysis on the recognized text
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    # Output the sentiment
    print("Sentiment:", sentiment)
    print("Polarity:", sentiment.polarity)
    print("Subjectivity:", sentiment.subjectivity)
    
except sr.UnknownValueError:
    print("Google Web Speech API could not understand the audio")
except sr.RequestError as e:
    print("Could not request results from Google Web Speech API; {0}".format(e))
