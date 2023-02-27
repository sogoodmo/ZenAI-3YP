import json
import cv2
from cv2 import trace
from flask import Flask, request, jsonify
from base64 import b64encode
import base64
from PIL import Image 
import numpy as np
from flask_cors import CORS
import io
import os

''' How we get all the models'''
from ZenAI import *

app = Flask(__name__)
app.debug = True
cors=CORS(app, supports_credentials=True) 

def img_2_byte(img):
    ret, image_jpeg = cv2.imencode('.jpg', img)
    image_bytes = image_jpeg.tobytes()
    image_b64 = b64encode(image_bytes).decode('utf-8')

    return image_b64

os.chdir(os.path.dirname(os.path.abspath(__file__)))
path = os.getcwd()

@app.route('/webcam-frame', methods=['POST', 'GET'])
def process_webcam_frame():
    
    try:
        data = request.get_json()
        result = data['data']
        diff = int(data['diff'])
        # timeSpent = int(data['timeSpent'])

        ''' Some how this is the only way this works? The conventional method of npbuffer and cv2.imdecode results in a None image...'''
        # Get the image from the request and decode it
        b = bytes(result, 'utf-8')
        byte_image = b[b.find(b'/9'):]

        npimg = np.array(Image.open(io.BytesIO(base64.b64decode(byte_image))))
        
        # Recolor from PIL to OpenCV
        image = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR) 


        # num_imgs = len([f for f in os.listdir('stupid') if os.path.isfile(os.path.join('stupid', f))])
        # cv2.imwrite(f'stupid/tmp_{num_imgs+1}.jpg', image) debugging save 

        response = process_image(image=image, last_pose=last_pose, model=SvmModel, SCORE_DIFFICULTY=diff)


        if 'error' in response:
            return_window = response['error_image']
            response['error_image'] = "FAILED"
            image_b64 = img_2_byte(return_window)

        else:
            return_window = response['combined_videos']
            response['combined_videos'] = "SUCCESS"
            image_b64 = img_2_byte(return_window)



        # cv2.imwrite(f'stupid/tmp_{num_imgs+1}_returned.jpg', return_window) debugging 


        response['image'] = image_b64
        # print("\n\n\n\n\n\nPRINTING RESPONSE DICTIONARY")
        # for k,v in response.items():
        #     if k != 'image':
        #         print(k, v)
        #     else:
        #         print(f'v too long here is k: {k}')
        # print("FINISHED PRINTING RESPONSE DICTIONARY \n\n\n\n\n\n")
        return jsonify(response)  
        
    except Exception as e:
        print('ERROR IN APP.PY')
        traceback.print_exc()
        return 'error'

if __name__ == '__main__':
    app.run()