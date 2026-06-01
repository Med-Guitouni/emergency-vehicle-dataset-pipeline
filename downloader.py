import yt_dlp
import os


class VideoDownloader:

    def __init__(self, output_dir="videos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_channel(self, channel_url, max_videos=32):
        options = {
            'outtmpl': f'{self.output_dir}/%(upload_date)s_%(title)s.%(ext)s',
            'format': 'mp4',
            'playlistend': max_videos,
            'quiet': False
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([channel_url])

    def download_single(self, url):
        options = {
            'outtmpl': f'{self.output_dir}/%(upload_date)s_%(title)s.%(ext)s',
            'format': 'mp4',
            'quiet': False
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])

    def list_downloaded(self):
        videos = [f for f in os.listdir(self.output_dir) if f.endswith('.mp4')]
        print(f"Found {len(videos)} videos:")
        for v in videos:
            print(f"  {v}")
        return videos

