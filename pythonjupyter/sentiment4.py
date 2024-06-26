from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import tkinter as tk
from tkinter import filedialog
# from google.colab import files
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load pre-trained model and processor from Hugging Face
# model_name = "facebook/wav2vec2-large-960h"
# processor = Wav2Vec2Processor.from_pretrained(model_name)
# model = Wav2Vec2ForCTC.from_pretrained(model_name)

# # Function to upload audio file
# # def upload_audio_file():
# #     uploaded = files.upload()
# #     file_path = next(iter(uploaded))
# #     return file_path

# # Load and process audio file
# def load_audio(file_path):
#     waveform, sample_rate = torchaudio.load(file_path)

#     # Convert to mono if stereo
#     if waveform.shape[0] > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)

#     return waveform, sample_rate

# # Preprocess the audio and transcribe
# def transcribe_audio(file_path):
#     waveform, sample_rate = load_audio(file_path)

#     # If the sample rate is not 16kHz, resample it
#     if sample_rate != 16000:
#         resampler = torchaudio.transforms.Resample(sample_rate, 16000)
#         waveform = resampler(waveform)

#     input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
#     print("Shape of input_values before model input:", input_values.shape)
#     with torch.no_grad():
#         logits = model(input_values).logits

#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.decode(predicted_ids[0])

#     return transcription

# # Transcribing the provided audio file
# print("Please upload a WAV file")
# file_path = "/home/anirudh/Documents/C++_practice/pythonjupyter/Television.wav"
# transcription = transcribe_audio(file_path)
# print("Transcription: ", transcription)

#--------------------------------------------------------
import whisper

def transcribe_audio(file_path):
    # Load the Whisper model
    model = whisper.load_model("small") # small, medium, base etc.

    # Transcribe the audio file
    result = model.transcribe(file_path)

    # Print the transcribed text
    # print("Transcribed text:", result["text"])
    return result["text"]

# Example usage
# file_path = "Television.wav"  # Replace with your actual file path
# transcribed_text = transcribe_audio(file_path)





from transformers import pipeline
# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

#--------------------------------------------------------

# Analyze sentiment of the transcribed text
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result


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