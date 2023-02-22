import cv2
import numpy as np
import base64

# Read an image file from disk
image = cv2.imread('/Users/mohamed/ZenAI-3YP/AppSite/zenai-app/src/bg.jpg')

# Encode the image as a JPEG string
_, image_jpeg = cv2.imencode('.jpg', image)
image_b64 = base64.b64encode(image_jpeg).decode('utf-8')

# Decode the JPEG string and convert it to a NumPy array
image_bytes = base64.b64decode(image_b64)
nparr = np.frombuffer(image_bytes, np.uint8)

# Decode the NumPy array as a JPEG image
image_decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

print(image_decoded)