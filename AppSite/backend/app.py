import cv2
from flask import Flask, request, jsonify
from base64 import b64decode, b64encode
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
app.debug = True
cors=CORS(app, supports_credentials=True) 


@app.route('/webcam-frame', methods=['POST'])
def process_webcam_frame():
    try:
        # Get the image from the request and decode it
        image_b64 = request.data
        image_bytes = b64decode(image_b64)

        with open('test.jpg', 'wb') as f:

            f.write(image_bytes)

        # Convert the image bytes to a NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(image is None)
        
        if image:
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
            print('here')
            return jsonify(response)
        
        else:
            return 'error'
    
    except Exception as e:
        print(e)
        return 'error'

if __name__ == '__main__':
    app.run()