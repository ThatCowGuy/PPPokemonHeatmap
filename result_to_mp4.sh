ffmpeg -y -framerate 3 -pattern_type glob -i 'detections/*.png' -c:v libx264 out.mp4
