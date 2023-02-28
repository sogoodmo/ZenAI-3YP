import numpy as np 
from ZenAI_mp_globals import angle_idxs_required, skip_landmark, mp_pose, mp_drawing
from joint_pose_vocab import vocab_dict
import cv2 

''' CV2 PutText globals'''
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
WHITE_TEXT = (255, 255, 255)

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)    
    c = np.array(c)   
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    
    return angle 

''' Extracting the angles for each of the relevant joints in the frame'''
def extract_joint_angles(pose_landmarks):

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

''' Extracting the x, y cords of each joint'''
def extract_joint_cords(pose_landmarks):
    # Getting cords of the landmarks FOR ANGLES WE CALC'D CORDS FOR. 
    # If any any of this landamrks have a visbility < MIN_DETECTION_CONFIDENCE. 
    # Display an error / throw an error saying you need the whole body in the frame
    pose_relevant_landmark_cords = [] 
    for _, idx, _ in angle_idxs_required:
        if idx in skip_landmark:
            continue
        
        current_landmark = pose_landmarks.landmark[idx]
        
        pose_relevant_landmark_cords.append([current_landmark.x, current_landmark.y])

    return pose_relevant_landmark_cords


''' Suprisngly using the same angle of 120 as a rough mid point between allowing 30degrees away from a 90degree is reasonable for both W2 and Tree
But I will leave the explicit code in to add to the readabilty.'''
def is_left_pose(pose, angles):

    ''' The standard index of left knee in the angles'''
    l_knee = 6 
  
    if pose == 'WarriorII':
        return angles[l_knee] <= 170 
    elif pose == 'Tree':
        return angles[l_knee] <= 90
    else:
        raise Exception("Only poses supported are 'WarriorII' and 'Tree'")
    
def is_down_tree(angles):
    l_shoulder = 0
    return angles[l_shoulder] <= 90

def cosine_similarity(angle1, angle2, difficulty):
    """
    Calculates the cosine similarity between two angles in degrees.
    """
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)
    cos_sim = np.cos(angle1_rad - angle2_rad)

    ''' Returing the score raised to the n'th power, where n determines how sensitive the score is.'''
    sim_rescaled = abs(cos_sim) ** difficulty
    
    ''' Returning weather the original cosine similarity was positive or negative to indicate which direction to turn'''
    return (cos_sim < 0, sim_rescaled)


def unnormalize_cords(x, y, fw, fh):
    return tuple(np.multiply([x, y], [fw, fh]).astype(int))

def calculate_center_of_gravity(pose_landmarks, fw, fh):
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    center_hip = (left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2

    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    center_shoulder = (left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2

    center_of_gravity = ((center_hip[0] + center_shoulder[0]) / 2, (center_hip[1] + center_shoulder[1]) / 2)


    arrow_start = (int(center_of_gravity[0] * fw), int(center_of_gravity[1] * fh))
    arrow_end = (arrow_start[0], int(fh * 0.8))


    return (arrow_start, arrow_end)


def gradient(score):

    """Converts a score between 0 and 1 to an RGB value that is a gradient between red and green."""
    r = int(max(0, min(255, (1 - score*3) * 255)))
    g = int(max(0, min(255, score * 255)))
    b = 0
    return (b, g, r)



def generate_user_pose_feedback(joint, user_angle, ideal_angle, score, is_greater_than_90, joint_idx, pose):
    """
    Generates feedback for the user's pose based on the joint, score, and angle difference.
    """
    if score >= 0.95:
        return f"Great job! Your {joint} is in the ideal position."
    else:
        rough_ideal_angle = min([0, 20, 45, 60, 90, 120, 135, 160, 180], key=lambda x: abs(x-int(ideal_angle)))
        rough_angle = round(float(user_angle), -1)


        if score <= 0.3:
            severity = 'a lot'
        elif 0.3 < score <= 0.6:
            severity = 'a fair bit'
        else:
            severity = 'a little'


        action = vocab_dict[pose][joint_idx][0] if is_greater_than_90 else vocab_dict[pose][joint_idx][1]
        common_error, common_fix = vocab_dict[pose][joint_idx][2].split(',')

        feedback_suggestion = {
            'raw' : f"{joint} | U: {rough_angle} / {user_angle} | G: {rough_ideal_angle} / {ideal_angle:.2f} | S: {score:.2f} | P: {pose}",
            'formatted' : [f'Your +{joint}+ is out of place.', 
                           f'You should aim to get your +{joint}+ to roughly +{rough_ideal_angle}+ degrees and it\'s currently around +{int(rough_angle)}+ degrees.',
                           f'You fix this by +{action}+ your +{joint}+ +{severity}+',
                           f'{common_error}',
                           f'{common_fix}'
                           ]
        }

        return feedback_suggestion


def calculate_center_of_gravity(pose_landmarks, fw, fh):
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    center_hip = (left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2

    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    center_shoulder = (left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2

    center_of_gravity = ((center_hip[0] + center_shoulder[0]) / 2, (center_hip[1] + center_shoulder[1]) / 2)


    arrow_start = (int(center_of_gravity[0] * fw), int(center_of_gravity[1] * fh))
    arrow_end = (arrow_start[0], int(fh * 0.8))


    return (arrow_start, arrow_end)


''' Render pose estimated from mediapipe'''
def render_skeleton(image, landmarks):
    mp_drawing.draw_landmarks(
        image, 
        landmarks,
        mp_pose.POSE_CONNECTIONS,                
        mp_drawing.DrawingSpec(color=(50, 145, 168), circle_radius=2, thickness=2),
        mp_drawing.DrawingSpec(color=(209, 192, 42), circle_radius=2, thickness=2)
    )

def create_live_video_display(image, pose_landmarks, classified_pose, classified_pose_confidence, frame_width, frame_height):
    #Revert image color 
    image.flags.writeable = True 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #Render detections 
    mp_drawing.draw_landmarks(
        image, 
        pose_landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(50, 145, 168), circle_radius=2, thickness=2),
        mp_drawing.DrawingSpec(color=(209, 192, 42), circle_radius=2, thickness=2)
    )
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    display_text = 'Neutral' if classified_pose == 'Neutral' else f'Confidence: {classified_pose} {classified_pose_confidence*100:.2f}%'
    
    cv2.putText(image, display_text, unnormalize_cords(0.3, 0.1, frame_width, frame_height), cv2.FONT_HERSHEY_SIMPLEX, 2, (125, 0, 0), 2, cv2.LINE_AA) 

    return image

def create_error_screen(frame_width, frame_height, errMessage = ""):
    black_image = np.zeros((frame_height, frame_width, 3), np.uint8)

    if errMessage == "":
        cv2.putText(black_image, "User not detected.", unnormalize_cords(0.1, 0.5, frame_width, frame_height), FONT, 3, WHITE_TEXT, 4, LINE)
        cv2.putText(black_image, "Please get in frame.", unnormalize_cords(0.1, 0.6, frame_width, frame_height), FONT, 3, WHITE_TEXT, 4, LINE)
    else:
        cv2.putText(black_image, errMessage, unnormalize_cords(0.1, 0.5, frame_width, frame_height), FONT, 3, WHITE_TEXT, 4, LINE)


    return black_image

def create_skeleton_video_display(pose_landmarks, classified_pose, classified_pose_confidence, joint_angles_rounded, joint_cords, joint_scores, elapsed_time, reference_image, ideal_angles, feedback_pose, joint_idx_map, frame_width, frame_height, dl=False):
    DISPLAY_ANGLE_SCORE = True

    ''' Creating the window that displays the feedback'''
    feedback_window = np.zeros((frame_height, frame_width, 3), np.uint8)

    ''' Create a black image to use as the background'''
    black_image = np.zeros((frame_height, frame_width, 3), np.uint8)

    ''' Render the skeleton on the black image ''' 
    render_skeleton(black_image, pose_landmarks)
    
    ''' Calculate center of gravity
        and draw an arrow from the center of gravity to the floor '''
    arrow_start, arrow_end = calculate_center_of_gravity(pose_landmarks, frame_width, frame_height)
    cv2.arrowedLine(black_image, arrow_start, arrow_end, (0, 255, 0), 5)
    
    cv2.putText(black_image, classified_pose, unnormalize_cords(0.1, 0.1, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)
    
    ''' Only display the score information of a pose if it's not a Neutral pose'''
    if classified_pose != 'Neutral':
        ''' 
        Calculating the best, worst and average scores
    
        Calculation of the overall score may be changed to include a weighted mean
        where there will be a pre-defined weight vector of joints for each pose.
        '''
        scores_list = [score[1] for score in joint_scores]
        total_pose_score = np.average(scores_list)
        best_joint_idx = np.argmax(scores_list)
        worst_joint_idx = np.argmin(scores_list)

        improvement_suggestions = dict()
        ''' Draw red circles on the joints with size proportional to the joint score ''' 
        for i, (x, y) in enumerate(joint_cords):
            diff_over_90, score = joint_scores[i]
            joint = joint_idx_map[i]
            
            radius = int((100 * (1 - score)) * 0.5)

            black_image = cv2.circle(black_image, unnormalize_cords(x, y, frame_width, frame_height), radius, (0, 0, 255), -1)

            ''' Highlighting the worst and best joint in the users pose
                & Display the joint score for each joint on the right side of the frame'''

            ''' Get rid of this for performance'''
            if not dl:
                if DISPLAY_ANGLE_SCORE:
                    if i == best_joint_idx:
                        cv2.putText(black_image, f'Best: {joint}: {score*100:.2f}%', unnormalize_cords(0.6, 0.2 + (1 / (1.25 * len(joint_cords))), frame_width, frame_height), FONT, 1, (0, 255, 0), 2, LINE)
                    elif i == worst_joint_idx:
                        cv2.putText(black_image, f'Worst: {joint}: {score*100:.2f}%', unnormalize_cords(0.6, 0.2 + (2 / (1.25 * len(joint_cords))), frame_width, frame_height), FONT, 1, (0, 0, 255), 2, LINE)
                        improvement_suggestions[worst_joint_idx] = generate_user_pose_feedback(joint, joint_angles_rounded[i], ideal_angles[i], score, diff_over_90, i, feedback_pose)
            else:
                if DISPLAY_ANGLE_SCORE:
                    if i == best_joint_idx:
                        cv2.putText(black_image, f'Best: {joint}: {score*100:.2f}%', unnormalize_cords(0.6, 0.2 + (i / (1.25 * len(joint_cords))), frame_width, frame_height), FONT, 1, (0, 255, 0), 2, LINE)
                    elif i == worst_joint_idx:
                        cv2.putText(black_image, f'Worst: {joint}: {score*100:.2f}%', unnormalize_cords(0.6, 0.2 + (i / (1.25 * len(joint_cords))), frame_width, frame_height), FONT, 1, (0, 0, 255), 2, LINE)
                        improvement_suggestions[worst_joint_idx] = generate_user_pose_feedback(joint, joint_angles_rounded[i], ideal_angles[i], score, diff_over_90, i, feedback_pose)
                    else:
                        cv2.putText(black_image, f'{joint}: {score*100:.2f}%', unnormalize_cords(0.6, 0.2 + (i / (1.25 * len(joint_cords))), frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)


            # with open('feedback.txt', 'a') as f:
            #     f.write(f'{feedback_pose}: {str(generate_user_pose_feedback(joint, joint_angles_rounded[i], ideal_angles[i], score, diff_over_90, i, feedback_pose))}\n')

        ''' Draw the rectangle on the left side of the screen '''
        rect_x, rect_y = unnormalize_cords(0.05, 1, frame_width, frame_height)
        rect_width, rect_height = int(frame_width * 0.05), int(total_pose_score * frame_height)
        cv2.rectangle(black_image, (rect_x, rect_y - rect_height), (rect_x + rect_width, rect_y), gradient(total_pose_score), -1)

        ''' Displaying the average of all the users scores individual angles'''
        cv2.putText(black_image, f'Pose Score: {total_pose_score*100:.2f}%', unnormalize_cords(0.42, 0.1, frame_width, frame_height), FONT, 2, WHITE_TEXT, 2, LINE)
        
        ''' Displaying how long the user has held the pose'''
        # cv2.putText(black_image, f'Time held: {elapsed_time:.2f} seconds', unnormalize_cords(0.1, 0.1, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)

        ''' 
        Generate Imrpovements
        '''
        # for i, suggestion in enumerate(improvement_suggestions.values()):
        #     y_placement = 0.2 + (i / (1.25 * len(improvement_suggestions)))
        #     cv2.putText(feedback_window, suggestion, unnormalize_cords(0.05, y_placement, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)
        formatted_suggestion = improvement_suggestions[worst_joint_idx]['formatted']
        # print(f'\n\n\n\n {formatted_suggestion} \n\n\n\n')
        ''' Generate the improvement suggestion for the worst joint'''
        if dl:
            cv2.putText(feedback_window, improvement_suggestions[worst_joint_idx]['raw'], unnormalize_cords(0.05, 0.1, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)

            cv2.putText(feedback_window, formatted_suggestion[0], unnormalize_cords(0.05, 0.2, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)

            cv2.putText(feedback_window, formatted_suggestion[1], unnormalize_cords(0.05, 0.3, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)
            cv2.putText(feedback_window, formatted_suggestion[2], unnormalize_cords(0.05, 0.35, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)
            
            cv2.putText(feedback_window, formatted_suggestion[3], unnormalize_cords(0.05, 0.45, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)

            cv2.putText(feedback_window, formatted_suggestion[4], unnormalize_cords(0.05, 0.55, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)

        ''' Pie Chart indiciating how well the joint is'''
        if not dl:
            score_angle = int(joint_scores[worst_joint_idx][1] * 360)
            pie_center = unnormalize_cords(0.62, 0.9, frame_width, frame_height)
            cv2.ellipse(black_image, pie_center, (40, 40), 0, 0, score_angle, (0, 255, 0), -1)
            cv2.ellipse(black_image, pie_center, (40, 40), 0, score_angle, 360, (0, 0, 255), -1)
            cv2.putText(black_image, f'{joint_idx_map[worst_joint_idx]} Score: {joint_scores[worst_joint_idx][1]*100:.0f}%', unnormalize_cords(0.62+0.05, 0.9+0.025, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)
        else:
            score_angle = int(joint_scores[worst_joint_idx][1] * 360)
            pie_center = unnormalize_cords(0.1, 0.8, frame_width, frame_height)
            cv2.ellipse(feedback_window, pie_center, (50, 50), 0, 0, score_angle, (0, 255, 0), -1)
            cv2.ellipse(feedback_window, pie_center, (50, 50), 0, score_angle, 360, (0, 0, 255), -1)
            cv2.putText(feedback_window, f'{joint_idx_map[worst_joint_idx]} Score: {joint_scores[worst_joint_idx][1]*100:.0f}%', unnormalize_cords(0.1+0.05, 0.8+0.05, frame_width, frame_height), FONT, 1, WHITE_TEXT, 2, LINE)

        ''' uncommoent '''
        # with open('test_output.txt', 'a') as f:
        #     f.write(formatted[0])
        #     f.write(formatted[1])
        #     f.write(formatted[2])
        #     f.write(formatted[3])
        ''' '''
        # diff_over_90, score = joint_scores[worst_joint_idx]
        # improvement_suggestion = generate_user_pose_feedback(joint_idx_map[worst_joint_idx], score, diff_over_90)
        
    ''' Displaying the reference image in low opacity'''
    ''' Get rid of this for performance'''
    reference_image = cv2.resize(reference_image, (frame_width, frame_height))
    blended_image = cv2.addWeighted(black_image, 0.8, reference_image, 0.2, 0)



    
    ''' Get rid of this for performance'''
    # ''' Write the joint angles on the black image ''' 
    for i, angle in enumerate(joint_angles_rounded):
        x, y = joint_cords[i]
        cv2.putText(blended_image, angle, unnormalize_cords(x, y, frame_width, frame_height), FONT, 0.5, WHITE_TEXT, 2, LINE)



    if feedback_pose != 'Neutral':
        ret_feedback = formatted_suggestion
    else:
        ret_feedback = ['' for _ in range(5)] 

    return feedback_window, blended_image, ret_feedback