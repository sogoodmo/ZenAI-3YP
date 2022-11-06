import mediapipe as mp
import cv2


def main() -> None:
    print("Starting test..")
    
    
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    with mp_face_detection.FaceDetection( model_selection=0, min_detection_confidence=0.5 ) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue 
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

            cv2.imshow('Test Video', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == ord('x'):
                break
    cap.release() 
            
if __name__ == '__main__':
    main()
