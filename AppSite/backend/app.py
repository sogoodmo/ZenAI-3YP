import cv2
from flask import Flask, request, jsonify
from base64 import b64encode
import base64
from PIL import Image 
import numpy as np
from flask_cors import CORS
import io
import os 


app = Flask(__name__)
app.debug = True
cors=CORS(app, supports_credentials=True) 


@app.route('/webcam-frame', methods=['POST', 'GET'])
def process_webcam_frame():
    
    try:
        data = request.get_json()
        result = data['data']

        ''' Some how this is the only way this works? The conventional method of npbuffer and cv2.imdecode results in a None image...'''
        # Get the image from the request and decode it
        b = bytes(result, 'utf-8')
        byte_image = b[b.find(b'/9'):]

        npimg = np.array(Image.open(io.BytesIO(base64.b64decode(byte_image))))
        
        # Recolor from PIL to OpenCV
        image = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR) 

        # Draw a red square in the middle of the image
        height, width, _ = image.shape
        x1 = int(width * 0.25)
        y1 = int(height * 0.25)
        x2 = int(width * 0.75)
        y2 = int(height * 0.75)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Convert the image to JPEG format and return it as a response
        ret, image_jpeg = cv2.imencode('.jpg', image)
        image_bytes = image_jpeg.tobytes()
        image_b64 = b64encode(image_bytes).decode('utf-8')
        response = {'image': image_b64}
        return jsonify(response)
        
    except Exception as e:
        print(e)
        return 'error'

if __name__ == '__main__':
    app.run()