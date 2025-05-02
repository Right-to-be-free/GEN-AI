import kagglehub

# Download latest version
path = kagglehub.dataset_download("samithsachidanandan/most-popular-1000-youtube-videos")

print("Path to dataset files:", path)