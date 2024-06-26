import whisper

def transcribe_audio(file_path):
    # Load the Whisper model
    model = whisper.load_model("small") # small, medium, base etc.

    # Transcribe the audio file
    result = model.transcribe(file_path)

    # Print the transcribed text
    # print("Transcribed text:", result["text"])
    return result["text"]

from transformers import pipeline
# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

#--------------------------------------------------------

# Analyze sentiment of the transcribed text
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result
###########
# list = []
# dictrr = {'col1':'file_name1','col2':'text_infile1','col3':'sentiment_infile1'}
# list.append(dictrr)
# df = pd.DataFrame(list)

# import os
# list_of_files = os.listdir('/home/anirudh/Documents/C++_practice/pythonjupyter')
############
# Main function to run the program
def main():
    print("Please upload a WAV file")
    file_path = "/home/anirudh/Documents/C++_practice/pythonjupyter/Adver.wav"
    if file_path:
        try:
            transcription = transcribe_audio(file_path)
            print("Transcription: ", transcription)
            sentiment = analyze_sentiment(transcription)
            print("Sentiment Analysis: ", sentiment)
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No file uploaded.")

if __name__ == "__main__":
    main()