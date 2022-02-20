import soundfile as sf

def save_audio(filepath, audio):
    sf.write(filepath, audio, 22050, "PCM_16")