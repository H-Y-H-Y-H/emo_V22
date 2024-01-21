import matplotlib.pyplot as plt
import numpy as np
import wave
import os
import contextlib
from pydub import AudioSegment

def plot_audio_waveform(file_path, frame_rate=30):
    """
    Plots the waveform of an audio file with time on the x-axis and normalized amplitude.
    Supports both WAV and MP3 formats.
    frame_rate: Frame rate of the audio file in FPS (default is 30fps).
    """
    # Convert mp3 file to wav if necessary
    if file_path.endswith('.mp3'):
        # Convert mp3 to wav
        sound = AudioSegment.from_mp3(file_path)
        file_path = file_path.replace('.mp3', '.wav')
        sound.export(file_path, format="wav")

    # Open the audio file as a waveform
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.readframes(-1)
        sound_info = np.frombuffer(frames, dtype=np.int16)
        nframes = f.getnframes()
        framerate = f.getframerate()
        nchannels = f.getnchannels()

    # If the audio is stereo, convert it to mono
    if nchannels == 2:
        sound_info = np.mean(sound_info.reshape(-1, 2), axis=1)

    # Calculate window size and mean values for each window
    sound_info = np.abs(sound_info)
    window_size = int(framerate / frame_rate)
    mean_values = [np.mean(sound_info[i:i + window_size]) for i in range(0, len(sound_info), window_size)]

    duration = len(mean_values) / frame_rate
    # Normalize the sound data
    mean_values = (mean_values - np.min(mean_values)) / (np.max(mean_values) - np.min(mean_values))


    # Create a time array in seconds
    time = np.linspace(0, duration, num=len(mean_values))
    mean_values = np.clip(mean_values,0,0.8)
    mean_values = (mean_values - np.min(mean_values)) / (np.max(mean_values) - np.min(mean_values))

    cmds = 1-mean_values
    # Plot the waveform
    # plt.figure(figsize=(12, 4))
    # plt.plot(time, cmds)
    # plt.title('Audio Waveform from MP4 with Mean Amplitude in Windows')
    # plt.ylabel('Mean Normalized Amplitude')
    # plt.xlabel('Time (seconds)')
    # plt.show()

    np.savetxt(f'wav_bl_cmds/cmds_{demo_id}.csv', cmds)
# Example usage (you would replace 'audio_file.wav' with your file path)
for demo_id in range(9,10):
    audio_path =f'../../EMO_GPTDEMO/audio/emo/emo{demo_id}.wav'
    plot_audio_waveform(audio_path)

# Note: This script expects the audio file to be present in the same directory.
# For MP3 files, it will create a temporary WAV file in the same directory.
