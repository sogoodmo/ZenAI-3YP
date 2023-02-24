import mediapipe as mp 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# All landmark except for hand and face specific
# RelevantLandmarks = list(mp_pose.PoseLandmark)[11:17] + list(mp_pose.PoseLandmark)[23:29]


#Match idx of RelevantLandmarks 
angle_idxs_required = [
    (11,23,25),    # l_shoulder_landmark_angle_idx
    (12,24,26),    # r_shoulder_landmark_angle_idx
    
    (13,11,23),    # l_arm_landmark_angle_idx
    (14,12,24),    # r_arm_landmark_angle_idx
    
    (15,13,11),    # l_hip_landmark_angle_idx
    (16,14,12),    # r_hip_landmark_angle_idx
    
    (23,25,27),    # l_knee_landmark_angle_idx
    (24,26,28)    # r_knee_landmark_angle_idx
]
skip_landmark = {
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_WRIST
}