import matplotlib.pyplot as plt
import csv 

'''


 SO MANY FUNCTIONS NOT DEFINED, MOVE THIS TO JUPYTER NOTEBOOK IF YOU WANNA USE


'''

''' Extracting the angles for each of the relevant joints in the frame'''
def extract_joint_angles(image, pose):
    # Make detections 
    results = pose.process(image=image)
    pose_landmarks = results.pose_landmarks

    ''' Converting landmarks into angles'''
    pose_relevant_landmark_angles = []
    # Going through all relevant landmarks, extracting their key angles
    # Calculating the angle then adding to array 
    for i1, i2, i3 in angle_idxs_required:
        
        fst = (pose_landmarks.landmark[i1].x, pose_landmarks.landmark[i1].y)
        snd = (pose_landmarks.landmark[i2].x, pose_landmarks.landmark[i2].y)
        thrd = (pose_landmarks.landmark[i3].x, pose_landmarks.landmark[i3].y)
        
        pose_relevant_landmark_angles.append(calc_angle(fst, snd, thrd))

    pose_relevant_landmark_angles_visual = np.around(pose_relevant_landmark_angles, 2).astype(str).tolist()

    return pose_relevant_landmark_angles, pose_relevant_landmark_angles_visual

DEMO_IMAGE_DIR = '../DemoImages'
CSV_FILE_NAME = '../DemoImages/joint_angles.csv'

# Set confidence thresholds
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Create pose detection object
with mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE, static_image_mode=True) as pose:
    
    # Open CSV file
    with open(CSV_FILE_NAME, mode='w') as csv_file:
        fieldnames = ['image_name', 'joint_angles']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop through images in directory
        for filename in os.listdir(DEMO_IMAGE_DIR):
            if filename.endswith(".jpeg"):
                print('here')
                image_path = os.path.join(DEMO_IMAGE_DIR, filename)
                image = cv2.imread(image_path)
                
                # Recolor image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                try:
                    # Detect pose and extract joint angles
                    pose_landmarks = pose.process(image).pose_landmarks 
                    joint_angles, joint_angles_rounded = extract_joint_angles(image, pose)
                    
                    # Write joint angles and image name to CSV file
                    writer.writerow({'image_name': filename, 'joint_angles': joint_angles})
                
                except Exception as e:
                    print(f"Error processing image {filename}: {e}")
    
cv2.destroyAllWindows()