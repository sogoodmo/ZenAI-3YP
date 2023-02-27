import mediapipe as mp
from pyparsing import col
from sklearn import svm

from ZenAI_mp_globals import mp_drawing, mp_pose, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, angle_idxs_required, skip_landmark
from joint_pose_vocab import vocab_dict
from ZenAI_helper import calc_angle, extract_joint_angles, extract_joint_cords, is_left_pose, is_down_tree, cosine_similarity, create_skeleton_video_display, create_error_screen
from ZenAI_data_transformation import generate_data
from ZenAI_ideal_angles import calculate_ideal_angles
from ZenAI_models import ZenRandomForest, ZenKNN, ZenNN, ZenSVM, Model

import os 
import numpy as np 
import cv2 
import traceback
import time 

os.chdir(os.path.dirname(os.path.abspath(__file__)))
path = os.getcwd()

data = generate_data()
joint_idx_map = data['joint_idx_map']
classes = data['classes']
columns = data['columns']
feedback_classes = data['feedback_classes']
combined_test = data['combined_test']
combined_train = data['combined_train']


ideal_angles_map = calculate_ideal_angles(classes=classes, combined_train=combined_train, columns=columns, feedback_classes=feedback_classes)
last_pose = 'Neutral'
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720 

SvmModel = ZenSVM(columns=columns, classes=classes)
NNModel = ZenNN(columns=columns, classes=classes)
KNNModel = ZenKNN(columns=columns, classes=classes)
RandomForestModel = ZenRandomForest(columns=columns, classes=classes)

EXAMPLE_POSE_IMAGE = {i.replace('.jpeg', '') : cv2.imread(f"flask_assets/{i}", 1) for i in os.listdir("flask_assets")}
# DEMO_POSE_VIDEOS = {f.replace('.mp4', '') : '../demo_videos/' + f  for f in os.listdir('../demo_videos')}


def process_image(image, last_pose, model: Model, SCORE_DIFFICULTY, start_time=0):
    with mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE, static_image_mode=False) as pose:
        # cv2.imwrite('tmp.jpg', image) debugging 
        try: 
            ''' Detect pose and extract angles / cords ''' 
            result = pose.process(image=image) 

            ''' If media pipe libray can't detect any poses, just display the normal image back.'''
            if not result.pose_landmarks:
                error_display = create_error_screen(FRAME_WIDTH, FRAME_HEIGHT)


                return {'error': 'cannot detect pose',
                        'error_image' : error_display
                        }

            pose_landmarks = result.pose_landmarks

            joint_angles, joint_angles_rounded = extract_joint_angles(pose_landmarks)
            joint_cords = extract_joint_cords(pose_landmarks)
            
            ''' Classification of Pose''' 
            classified_pose, classified_pose_confidence, prediction_probabilites = model.predict(joint_angles)

            # print(f'\n\n\n\nCLASSIFIED POSE {columns} {classified_pose} {classified_pose_confidence}\n{joint_angles}\n{prediction_probabilites}\n\n\n\n')
            # with open('shit.txt', 'a+') as f:
            #     f.write(str(joint_angles))
            #     f.write(str(joint_cords))
            #     f.write(',\n')

            ''' We take the pose the user is doing to be neutral if either it's classified as a neutral pose or the confidence of the classified pose is < 70%'''
            classified_pose = classified_pose if classified_pose_confidence >= 0.70 else 'Neutral'

            # Get the current time and if the pose has changed, update the start time to be the current time
            current_time = time.time()
            if classified_pose != last_pose:
                start_time = current_time
                last_pose = classified_pose

            # If the pose did just change, then this value will be 0 
            elapsed_time = current_time - start_time

            feedback_pose = classified_pose
        
            ''' 
                It was a waste of computation to calculate the angle score for a neutral pose

                and if the pose is Warrior or Tree we need to add another layer of detail when scoring angles 
                as the ideal angles of someone doing a Left Tree or Right Tree are different. 
                This makes the feedback dynamic and even allows for variation in the tree pose specifically
            '''
            ideal_angles = [] 
            angles_score = []
            if classified_pose != 'Neutral':
                if classified_pose in {'WarriorII', 'Tree'}:
                    left = is_left_pose(classified_pose, joint_angles)
                    right = not left

                    if classified_pose == 'WarriorII':
                        if left:
                            feedback_pose = 'WarriorII_L'
                        else:
                            feedback_pose = 'WarriorII_R'
                        

                    elif classified_pose == 'Tree':
                        down = is_down_tree(joint_angles) 
                        up = not down 

                        if left and down:
                            feedback_pose = 'Tree_L_D'
                        elif left and up:
                            feedback_pose = 'Tree_L_U'
                        elif right and down:
                            feedback_pose =  'Tree_R_D'
                        elif right and up:
                            feedback_pose = 'Tree_R_U'     


                ideal_angles = ideal_angles_map[feedback_pose]


                ''' Calculate score for current frame given the expected exercise'''
                angles_score = [cosine_similarity(cur_angle, ideal_angle, SCORE_DIFFICULTY) for cur_angle, ideal_angle in zip(joint_angles, ideal_angles)]
            print(SCORE_DIFFICULTY)
            # print(f'\n\n\n\n\n{classified_pose}\n {angles_score}\n\n\n\n\n')
            ''' Second Window'''
            feedback_window, black_image, ret_feedback = create_skeleton_video_display(pose_landmarks, 
                                                          classified_pose,
                                                          classified_pose_confidence,
                                                          joint_angles_rounded,
                                                          joint_cords,
                                                          angles_score,
                                                          elapsed_time,
                                                          EXAMPLE_POSE_IMAGE[classified_pose],
                                                          ideal_angles,
                                                          feedback_pose, 
                                                          joint_idx_map,
                                                          FRAME_WIDTH, FRAME_HEIGHT)

            combined_videos = np.concatenate((black_image, feedback_window), axis=1)
            combined_videos = black_image

            pose_error = '' if classified_pose == 'Neutral' else ret_feedback[3]
            pose_fix = '' if classified_pose == 'Neutral' else ret_feedback[4]

            return {
                'classified_pose' : classified_pose,
                'classified_pose_confidence' : classified_pose_confidence,
                'joint_angles_rounded' : list(joint_angles_rounded),
                'joint_cords' : list(joint_cords),
                # 'angles_score' : list(angles_score),
                'elapsed_time' : elapsed_time,
                'ideal_angles' : list(ideal_angles),
                'feedback_pose' : feedback_pose,
                'combined_videos' : combined_videos,
                'feedback' : ret_feedback[:3],
                'pose_error' : pose_error,
                'pose_fix' : pose_fix
            }
            
        except Exception as e:
            print('ERROR IN ZENAI.PY')
            traceback.print_exc()
            error_display = create_error_screen(FRAME_WIDTH, FRAME_HEIGHT, str(e)) 

            return {'error' : 'exception caught',
                    'error_image' : error_display}
