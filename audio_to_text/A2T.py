import whisper

model = whisper.load_model("base")
result = model.transcribe("C:/Users/yuhan/PycharmProjects/emo_V22/audio_to_text/raw.MP3")
print(result["text"])
