import cv2
import mediapipe as mp
import numpy as np
import os
import csv

MAX_IMG_STORE = 0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# All landmark except for hand and face specific
RelevantLandmarks = list(mp_pose.PoseLandmark)[11:17] + list(mp_pose.PoseLandmark)[23:29]

path = 'C:\\Users\\moham\\Desktop\\Third Year Project\\LauranceMoneyDataset\\'

class_map = {
    # "warrior" : "WarriorIII", #Warrior
    "tree" : "Tree", #Tree 
    "cobra": "Cobra", #Cobra
    "chair": "Chair", #Plank 
    "dog": "DownDog" #Downward Dog
}

skip_landmark = {
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_WRIST
}

l_hip_landmark_angle_idx = (11,23,25)
r_hip_landmark_angle_idx = (12,24,26)

l_shoulder_landmark_angle_idx = (13,11,23)
r_shoulder_landmark_angle_idx = (14,12,24)

l_arm_landmark_angle_idx = (15,13,11)
r_arm_landmark_angle_idx = (16,14,12)

l_knee_landmark_angle_idx = (23,25,27)
r_knee_landmark_angle_idx = (24,26,28)

#Match idx of RelevantLandmarks 
angle_idxs_required = [
    l_shoulder_landmark_angle_idx,
    r_shoulder_landmark_angle_idx,
    
    l_arm_landmark_angle_idx,
    r_arm_landmark_angle_idx,
    
    l_hip_landmark_angle_idx,
    r_hip_landmark_angle_idx,
    
    l_knee_landmark_angle_idx,
    r_knee_landmark_angle_idx
]


def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)    
    c = np.array(c)   
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle 
    
    return angle 

def generate_csv_train(train=True):
    '''
    Open CSV -> For all classes -> Open each file in that class folder -> Generate keypoints and write to CSV
    
    '''
    total_lines_added = 0
    with open(os.path.join(path, "training.csv" if train else "testing.csv"), 'w') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for classFolder, className in class_map.items():
            imgFolder = os.path.join(path + 'train\\' + classFolder) if train else os.path.join(path + 'test\\' + classFolder)
            
            for filename in os.listdir(imgFolder):
                
                image = cv2.imread(os.path.join(imgFolder, filename))
                    
                # Initialize fresh pose tracker and run it.
                with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose_tracker:
                    result = pose_tracker.process(image=image)
                    pose_landmarks = result.pose_landmarks


                if pose_landmarks is not None:
            
                    pose_relevant_landmark_angles = []
                    # Going through all relevant landmarks, extracting their key angles
                    # Calculating the angle then adding to array 
                    for i1, i2, i3 in angle_idxs_required:
                        
                        fst = (pose_landmarks.landmark[i1].x, pose_landmarks.landmark[i1].y)
                        snd = (pose_landmarks.landmark[i2].x, pose_landmarks.landmark[i2].y)
                        thrd = (pose_landmarks.landmark[i3].x, pose_landmarks.landmark[i3].y)
                        
                        
                        pose_relevant_landmark_angles.append(calc_angle(fst, snd, thrd))

                        
                    # Write pose sample to CSV.
                    pose_relevant_landmark_angles_data = np.around(pose_relevant_landmark_angles, 5).astype(str).tolist()
                        
                    csv_out_writer.writerow([filename] + [className] + pose_relevant_landmark_angles_data)
                    total_lines_added += 1
                

            print(f"{'Training Total: ' if train else 'Testing Total: '}{total_lines_added} (Finished: {className})\n\n")
                    
generate_csv_train(True)
generate_csv_train(False)
print("!!! DONE !!!")
