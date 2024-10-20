url=$(yt-dlp -g "https://www.youtube.com/watch?v=89KPOnjHEC4" -S res:480)
ffmpeg -i "$url" -vf fps=1/60 out%d.jpg