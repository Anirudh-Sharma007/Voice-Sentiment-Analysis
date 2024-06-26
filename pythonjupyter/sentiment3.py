import whisper

def transcribe_audio(file_path):
    # Load the Whisper model
    model = whisper.load_model("small") # small, medium, base etc.

    # Transcribe the audio file
    result = model.transcribe(file_path)

    # Print the transcribed text
    print("Transcribed text:", result["text"])
    return result["text"]

# Example usage
file_path = "Television.wav"  # Replace with your actual file path
transcribed_text = transcribe_audio(file_path)
  
  # In the code above, we first import the  whisper  module and define a function
  # transcribe_audio  that takes the file path of an audio file as an argument. 
  # We then load the Whisper model using the  load_model  function and transcribe the audio

