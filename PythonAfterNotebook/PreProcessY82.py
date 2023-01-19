import cv2
import mediapipe as mp
import numpy as np
import os
import requests
import csv

MAX_IMG_STORE = 0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

min_detection_confidence = 0.5

min_tracking_confidence = 0.5

# All landmark except for hand and face specific
RelevantLandmarks = list(mp_pose.PoseLandmark)[11:17] + list(mp_pose.PoseLandmark)[23:29]

path = 'C:\\Users\\moham\\Desktop\\Third Year Project\\yoga82code\\Yoga-82\\'
img_links_path = os.path.join(path, 'yoga_dataset_links')
train_file = os.path.join(path, 'yoga_train.txt')
test_file = os.path.join(path, 'yoga_test.txt')


class_map = {
    (0,3,73) : "WarriorIII", #WarriorIII
    (0,0,68) : "Tree", #Tree 
    (4,14,10): "Cobra", #Cobra
    (0,0,8): "Chair", #Plank- Changing to chair 
    (0,1,17): "DownDog" #Downward Dog
}

def generate_link_map(train=True):
    train_links = dict()
    with open(train_file if train else test_file, 'r') as file:
        for line in file:
            img_path, l1, l2, l3 = line.split(',')

            #Ending early if img not classified as what we want 
            img_class = (int(l1), int(l2), int(l3))

            if img_class not in class_map:
                continue 

            img_path, img_num = img_path.replace('/', ' ').split(' ') 
            img_path = img_path + '.txt'
            
            img_to_find = img_path.replace('.txt', '/'+img_num)
            
            #Add more links to dictionary 
            train_links |= {inner_line.split('\t')[0] : inner_line.split('\t')[1].strip() 
                            for inner_line in open(os.path.join(img_links_path, img_path))}
    return train_links 

training_img_map = generate_link_map(train=True)
testing_img_map = generate_link_map(train=False)

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
skip_landmark = {
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_WRIST
}

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)    
    c = np.array(c)   
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 380-angle 
    
    return angle 

def generate_csv_train(train=True):
    with open(train_file if train else test_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)
    img_map = training_img_map if train else testing_img_map 
    img_count = 0
    tot = 0
    with open(os.path.join(path, "training.csv" if train else "testing.csv"), 'w') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        for line_idx, line in enumerate(lines):
            
            img_path, l1, l2, l3 = line.split(',')
            
            #Ending early if img not classified as what we want 
            img_class = (int(l1), int(l2), int(l3))
            
            if img_class not in class_map:
                continue
            
            # Ending early if for some reason I havn't included this image 
            # In our previous search 
            if img_path not in img_map:
                # print("IMAGE NOT IN MAP.. ERROR?")
                return 
                
            img_url = img_map[img_path] 
            tmp_img = os.path.join(path,"tmp.jpg")
            
            try:
                img_data = requests.get(img_url).content
            except Exception as e:
                # print(f'Error in downloading image... {img_url} | Trying next image...\n')
                # print(e)
                continue 
            
            with open(tmp_img, 'wb') as handler:
                handler.write(img_data)
            
            try:
                image = cv2.imread(tmp_img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                os.remove(tmp_img)
            except FileNotFoundError as e:
                # print("Couldn't find temp file.. Trying next img\n")
                continue 
            except Exception as e:
                # print("COULDN'T CONVERT IMAGE TO CV2 ARRAY.. Trying next img\n")
                continue
            
            
            # print(f"Succesfully download and read image from URL")       



            # Initialize fresh pose tracker and run it.
            with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose_tracker:
                result = pose_tracker.process(image=image)
                pose_landmarks = result.pose_landmarks
                


            #If a one of the valid pose' was detected, write this  
            output_image = image.copy()

            if pose_landmarks is not None:
                # print(f"Succesfully generated pose landmarks from url image {img_url}")
                
                pose_relevant_landmark_angles = []
                # Going through all relevant landmarks, extracting their key angles
                # Calculating the angle then adding to array 
                for i1, i2, i3 in angle_idxs_required:
                    
                    fst = (pose_landmarks.landmark[i1].x, pose_landmarks.landmark[i1].y)
                    snd = (pose_landmarks.landmark[i2].x, pose_landmarks.landmark[i2].y)
                    thrd = (pose_landmarks.landmark[i3].x, pose_landmarks.landmark[i3].y)
                    
                    
                    pose_relevant_landmark_angles.append(calc_angle(fst, snd, thrd))

                
                
                #Getting cords of the landmarks FOR ANGLES WE CALC'D CORDS FOR
                pose_relevant_landmark_cords = [[pose_landmarks.landmark[idx].x, pose_landmarks.landmark[idx].y]
                                               for _, idx, _ in angle_idxs_required if idx not in skip_landmark]
                
                # Write pose sample to CSV.
                pose_relevant_landmark_angles_data = np.around(pose_relevant_landmark_angles, 5).astype(str).tolist()
                pose_relevant_landmark_angles_visual = np.around(pose_relevant_landmark_angles, 2).astype(str).tolist()
                
                csv_out_writer.writerow([class_map[img_class]] + pose_relevant_landmark_angles_data)
                tot += 1
                # print("!!! Successfully added example row to CSV !!!")
                
                # Only storing certain number of images, don't want to clutter my disk 
                if (img_count < MAX_IMG_STORE):
                
                     # Map pose landmarks from [0, 1] range to absolute coordinates to get
                    # correct aspect ratio.
                    frame_height, frame_width = output_image.shape[:2]
                    pose_relevant_landmark_cords *= np.array([frame_width, frame_height])
                    real_cords = tuple(pose_relevant_landmark_cords.astype(int))


                    for idx, kp_cords in enumerate(real_cords):
                        cv2.putText(output_image, f'{pose_relevant_landmark_angles_visual[idx]}', kp_cords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA) 

                    #Save image and recolour
                    cv2.imwrite(os.path.join(path, 'tmp/tmp_img_' + str(img_count) + '.png'), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
                    img_count+=1
                    # print("!!! Successfully saved annotated image tmp folder !!!")
                
                print(f"COMPLETED IMAGE {line_idx}/{num_lines}...Total: {'Training Total: ' if train else 'Testing Total: '}{tot}\n\n")
                
            # else:
                # print(f"Could not extract pose from image: {img_url} Trying next image...\n")
generate_csv_train(False)
generate_csv_train(True)
print("!!! DONE !!!")
