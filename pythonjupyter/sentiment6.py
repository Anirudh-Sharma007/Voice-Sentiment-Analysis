import whisper
from transformers import pipeline
import os
import pandas as pd

def transcribe_audio(file_path):
    # Load the Whisper model
    model = whisper.load_model("small")  # small, medium, base, etc.

    # Transcribe the audio file
    result = model.transcribe(file_path)

    # Return the transcribed text
    return result["text"]

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result

# Main function to run the program
def main():
    folder_path = "/home/anirudh/Documents/C++_practice/pythonjupyter/audioFiles"  # Replace with your folder path
    list_of_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    results = []

    for file_name in list_of_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            transcription = transcribe_audio(file_path)
            print(f"Transcription for {file_name}: {transcription}")
            sentiment = analyze_sentiment(transcription)
            print(f"Sentiment Analysis for {file_name}: {sentiment}")
            
            result_dict = {
                'file_name': file_name,
                'text_infile': transcription,
                'sentiment_infile': sentiment
            }
            results.append(result_dict)
        except Exception as e:
            print(f"An error occurred with file {file_name}: {e}")

    # Optionally, convert results to a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv("transcriptions_and_sentiments.csv", index=False)
    print("Results have been saved to transcriptions_and_sentiments.csv")
    print(df)

if __name__ == "__main__":
    main()
