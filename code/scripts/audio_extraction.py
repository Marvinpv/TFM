import csv
import yt_dlp as youtube_dl

# Función para descargar el audio del video
def descargar_audio(youtube_id, solo_start_sec, solo_end_sec):
    # Configuración de youtube-dl
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audios/%(title)s.%(ext)s',
    }

    # URL del video de YouTube
    video_url = f'https://www.youtube.com/watch?v={youtube_id}'

    # Rango de tiempo para extraer el audio
    time_range = f'{solo_start_sec}-{solo_end_sec}'

    # Agregar el rango de tiempo al argumento de youtube-dl
    ydl_opts['postprocessors'] = [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192'
    }]

    # Descargar el audio
    print(video_url)
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
        except Exception as e:
            print('Error:',e)

# Leer el archivo CSV y descargar los audios
with open('csv_youtube.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        youtube_id = row['youtube_id']
        solo_start_sec = float(row['solo_start_sec'])
        solo_end_sec = float(row['solo_end_sec'])
        descargar_audio(youtube_id, solo_start_sec, solo_end_sec)