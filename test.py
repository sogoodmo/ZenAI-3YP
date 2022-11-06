import mediapipe as mp
import cv2 as cv 
import numpy as np 


def face_detection() -> None:
    print("Starting test..")
    
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv.VideoCapture(0)
    
    with mp_face_detection.FaceDetection( model_selection=0, min_detection_confidence=0.5 ) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue 
            
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = face_detection.process(image)
            
            
            img_no_ml = image
            
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

            cv.imshow('Test Video', cv.flip(image, 1))
            cv.imshow(f'No Recognition', cv.flip(img_no_ml,1))

            if cv.waitKey(5) & 0xFF == ord('x'):
                break
    cap.release() 

def test_vid_cap():
    
    print("Starting test..")
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    
    frame_num = 0
    while True:
        frame_num += 1
        ret, frame = cap.read()   
        if not ret:
            print("Cannot receieve frame img. Exiting...")
            break 
        
        IMG = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv.imshow(f'Frame #1', cv.flip(IMG,1))
        # cv.imshow(f'Frame #2', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    print("Finishing..")
       
def main() -> None:
    face_detection() 
    
    
    
    # test_vid_cap()
          
      
if __name__ == '__main__':
    main()
