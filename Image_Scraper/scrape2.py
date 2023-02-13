from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests
import os

URL = "https://unsplash.com/s/photos/person-standing"

# Start a Selenium webdriver instance
driver = webdriver.Chrome()
driver.get(URL)


# Scroll down the page
print("Go")
time.sleep(5)
print("Stop")
for i in range(10):
    driver.execute_script("window.scrollTo(0, 0.8 * document.body.scrollHeight);")

    time.sleep(2)
    # # Get the height of the page
    # page_height = driver.execute_script("return document.body.scrollHeight")

    # # Scroll down to 4/5ths of the page height
    # scroll_to = page_height * 4 / 5
    # driver.execute_script(f"window.scrollTo(0, {scroll_to});")
    # time.sleep(5)
    # try:
    #     # Find the button with class "CwMIr DQBsa p1cWU jpBZ0 AYOsT Olora I0aPD dEcXu"
    #     button = driver.find_element_by_class_name("CwMIr DQBsa p1cWU jpBZ0 AYOsT Olora I0aPD dEcXu")
    #     button.click()
    #     time.sleep(5)
    # except:
    #     # If the button is not found, continue scrolling
    #     continue

# Get the page source after scrolling
html = driver.page_source

# Close the webdriver
driver.quit()

# Use BeautifulSoup to parse the page source
soup = BeautifulSoup(html, "html.parser")
images = soup.find_all("img")

# Create a directory to save the images
if not os.path.exists("images_2"):
    os.makedirs("images_2")

# Download each image and save it
for i, image in enumerate(images):
    img_url = image["src"]
    if not img_url.startswith("http"):
        # sometimes an image source can be relative 
        # if it is provide the base url which also happens 
        # to be the site variable atm. 
        img_url = f"{URL}{img_url}"
    response = requests.get(img_url)
    with open(f"images_2/img_{i}.jpg", "wb") as f:
        f.write(response.content)
