import urllib.request
import time
import os
from tqdm import tqdm

urls = list()
with open("urls_with_name.txt", "r") as file:
    urls = [i.strip().split(",") for i in file.readlines()]

for filename, url in tqdm(urls):
    urllib.request.urlretrieve(url, os.path.join("../raw", filename))
    time.sleep(2)
