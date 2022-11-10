import mediapipe as mp
import cv2 
import numpy as np 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 


 
''' Depricated Methods - Simply for testing and understanding the CV2 + Media Pipe Code'''
# def pose_detection():
#     print("Starting pose detection...")
#     mp_drawing = mp.solutions.drawing_utils
#     mp_drawing_styles = mp.solutions.drawing_styles
#     mp_pose = mp.solutions.pose
    
    
#     cap = cv.VideoCapture(0)
    
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while cap.isOpened():
#             suc, img = cap.read()
            
#             if not suc:
#                 print("Empty frame..")
#                 continue 
            
#             img.flags.writeable = False
#             img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#             img_pose = pose.process(img)

#             # Drawing the image             
#             img.flags.writeable = True
#             img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            
#             mp_drawing.draw_landmarks(img, img_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
#             cv.imshow('Test Video', cv.flip(img, 1))

#             if cv.waitKey(5) & 0xFF == ord('x'):
#                 break
#     cap.release()
#     cv.destroyAllWindows()
# def face_detection() -> None:
#     print("Starting test..")
    
#     mp_face_detection = mp.solutions.face_detection
#     mp_drawing = mp.solutions.drawing_utils
    
#     cap = cv.VideoCapture(0)
    
#     with mp_face_detection.FaceDetection( model_selection=0, min_detection_confidence=0.5 ) as face_detection:
#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 print("Ignoring empty camera frame")
#                 continue 
            
#             image.flags.writeable = False
#             image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#             results = face_detection.process(image)
            
            
#             img_no_ml = image
            
#             image.flags.writeable = True
#             image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
#             if results.detections:
#                 for detection in results.detections:
#                     mp_drawing.draw_detection(image, detection)

#             cv.imshow('Test Video', cv.flip(image, 1))
#             cv.imshow(f'No Recognition', cv.flip(img_no_ml,1))

#             if cv.waitKey(5) & 0xFF == ord('x'):
#                 break
#     cap.release() 
# def test_vid_cap():

    
#     print("Starting test..")
    
#     cap = cv.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open camera")
#         return
    
    
#     frame_num = 0
#     while True:
#         frame_num += 1
#         ret, frame = cap.read()   
#         if not ret:
#             print("Cannot receieve frame img. Exiting...")
#             break 
        
#         IMG = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#         cv.imshow(f'Frame #1', cv.flip(IMG,1))
#         # cv.imshow(f'Frame #2', frame)
        
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
        
#     cap.release()
#     print("Finishing..")
def estimate_pose():

    ''' Using the media pose model'''
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        '''
            Capturing webcam footage 
        '''
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            suc, frame = cap.read() 
            if not suc:
                print("Frame empty..")
                continue 
            
            #Recolor image 
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detections 
            results = pose.process(image)
            
            #Revert image color 
            image.flags.writeable = True 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
            #Render detections 
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            
            
            # Displaying the frame 
            cv2.imshow("Video", cv2.flip(image, 1))
            
            # Closing the video capture  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release() 
    cv2.destroyAllWindows()

def main() -> None:
    estimate_pose()
          
      
if __name__ == '__main__':
    main()
