# import pygame
# import sys
# import os

# def play_audio(file_path):

#     pygame.mixer.init()
#     pygame.display.set_mode((200,100))
#     pygame.mixer.music.load(file_path)
#     pygame.mixer.music.set_volume(10)
#     pygame.mixer.music.play(0)

#     while pygame.mixer.music.get_busy():
#         pygame.event.poll()
#         pygame.time.Clock().tick(10)
#     pygame.mixer.quit()

# if __name__ == "__main__":
#     audio_file = '../dataset/en1-ava.wav'
#     play_audio(audio_file)
#     # if len(sys.argv) > 1:
#     #     audio_file = sys.argv[1]
#     #     play_audio(audio_file)
#     # else:
#     #     print("Usage: python play_audio.py <path_to_audio_file>")


# importing vlc module
import vlc
audio_file = '../dataset/en1-ava.wav'
# creating vlc media player object
# media = vlc.MediaPlayer(audio_file)
 
# # start playing video
# media.play()
import subprocess
import os

p = subprocess.Popen([os.path.join("C:/", "Program Files(x86)", "VideoLAN", "VLC", "vlc.exe"),audio_file])