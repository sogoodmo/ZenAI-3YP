import requests
from bs4 import BeautifulSoup
import os

URL = "https://unsplash.com/s/photos/person-standing"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")
images = soup.find_all("img")

if not os.path.exists("images"):
    os.makedirs("images")

for i, image in enumerate(images):
    img_url = image["src"]
    if not img_url.startswith("http"):
        # sometimes an image source can be relative 
        # if it is provide the base url which also happens 
        # to be the site variable atm. 
        img_url = f"{URL}{img_url}"
    response = requests.get(img_url)
    with open(f"images/img_{i}.jpg", "wb") as f:
        f.write(response.content)