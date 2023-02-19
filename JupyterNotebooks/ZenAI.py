#!/usr/bin/env python
# coding: utf-8
# In[1]:
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import cv2
import mediapipe as mp
import time 
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
import traceback


TRAIN_SVM = True
TRAIN_FOREST = False
TRAIN_NN = False
TRAIN_KNN = False

os.chdir('../Dataset')
path = os.getcwd()

columns = ['class','l_shoulder','r_shoulder','l_arm','r_arm','l_hip','r_hip','l_knee','r_knee']

Y82_test = pd.read_csv(os.path.join(path,'Y82_testing_new.csv'), header=None)
Y82_train = pd.read_csv(os.path.join(path,'Y82_training_new.csv'), header=None)

L_test = pd.read_csv(os.path.join(path,'L_testing_new.csv'), header=None)
L_train = pd.read_csv(os.path.join(path,'L_training_new.csv'), header=None)

W2_test = pd.read_csv(os.path.join(path,'W2_testing_new.csv'), header=None)
W2_train = pd.read_csv(os.path.join(path,'W2_training_new.csv'), header=None)

Neutral_test = pd.read_csv(os.path.join(path,'Neutral_testing.csv'), header=None)
Neutral_train = pd.read_csv(os.path.join(path,'Neutral_training.csv'), header=None) 

combined_test = pd.concat([L_test, Y82_test, W2_test, Neutral_test])
combined_train = pd.concat([L_train, Y82_train, W2_train, Neutral_train])

combined_test.columns = columns 
combined_train.columns = columns 

combined_test['class'], classes = pd.factorize(combined_test['class'])
combined_train['class'], _ = pd.factorize(combined_train['class'])

''' Filtering out all the extra examples for the cobra class to make a more balanced data set '''
extra_test_cobra_rows = combined_test[combined_test['class'] == 1].sample(130)
extra_train_cobra_rows = combined_train[combined_train['class'] == 1].sample(275)

combined_test = combined_test.drop(extra_test_cobra_rows.index)
combined_train = combined_train.drop(extra_train_cobra_rows.index)

classes = list(classes)

''' These will be the list of classes when giving feedback due to the fact Tree and Warrior have directional variations'''
feedback_classes = classes + ['WarriorII_L', 'WarriorII_R', 'Tree_L_D', 'Tree_R_D', 'Tree_L_U', 'Tree_R_U']
feedback_classes.remove('Neutral')
feedback_classes.remove('Tree')
feedback_classes.remove('WarriorII')


# In[2]:


all_combined_df = pd.concat([Y82_train, Y82_test, L_test, L_train, W2_train, W2_test])
all_combined = pd.concat([combined_test, combined_train])

def split_features_labels(df):
    return df.drop('class', axis=1), df['class']
    
for i, c in enumerate(classes):
    print(f"Train: {c} - {len(combined_train[combined_train['class'] == i])}")
    print(f"Test: {c} - {len(combined_test[combined_test['class'] == i])}")
    print()


# # Random Forest Classifer
# # 88-89% Maybe 90?
# ## IDK anymore.. even this is at like 95% lol

# In[3]:


if TRAIN_FOREST:
  RANDOM_ORDER_DATA = True 
  MAX_ESTIMATORS = 100
  MAX_DEPTH = 8 

  if RANDOM_ORDER_DATA:
    mutated_train = combined_train.sample(frac=1)
  else:
    mutated_train = combined_train

  X_train, y_train = split_features_labels(mutated_train)
  X_test, y_test = split_features_labels(combined_test)


  forest_classifier = RandomForestClassifier()
  param_grid = {'n_estimators' : np.arange(1, MAX_ESTIMATORS),
                'max_depth' : np.arange(1, MAX_DEPTH),
              }

  forest_classifier_gscv = RandomizedSearchCV(forest_classifier, param_distributions=param_grid, cv=5, n_jobs=-1)

  #fit model to data
  forest_classifier_gscv.fit(X_train, y_train)

  MAX_DEPTH = forest_classifier_gscv.best_params_['max_depth']
  N_ESTIMATORS = forest_classifier_gscv.best_params_['n_estimators']

  best_forest = RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS)
  best_forest.fit(X_train, y_train)

  pred = best_forest.predict(X_test)
  cm = confusion_matrix(y_test, pred)
  display_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
  display_confusion_matrix.plot()
  print(classification_report(y_test, pred))


# # KNN Classifer 
# # Eh Around 88%
# 

# In[4]:


if TRAIN_KNN:
    ''' Don't need to do this anymore, cause the dataset is already split (Didn't realise this) '''
    # from sklearn.model_selection import train_test_split
    # 80/20 Split of data, Doesn't randomize, Randomsplit ensures the proportion of classes is the same. 
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    RANDOM_ORDER_DATA = True 

    '''Shuffle data for better resuliting'''
    if RANDOM_ORDER_DATA:
        mutated_train = combined_train.sample(frac=1)
    else:
        mutated_train = combined_train

    X_train, y_train = split_features_labels(mutated_train)
    X_test, y_test = split_features_labels(combined_test)

    max_neighbours = int(np.sqrt(len(X_train)))
    knn_algorithms = ['kd_tree', 'brute', 'ball_tree']


    knn = KNN()

    param_grid = {'n_neighbors' : np.arange(1, max_neighbours),
                'algorithm' : knn_algorithms}

    # Using grid search cross validation to find the best value of K 
    knn_gscv = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)

    knn_gscv.fit(X_train, y_train)

    ALGORITHM = knn_gscv.best_params_['algorithm']
    N_NEIGHBORS = knn_gscv.best_params_['n_neighbors']

    best_KNN = KNN(algorithm=ALGORITHM, n_neighbors=N_NEIGHBORS)
    best_KNN.fit(X_train, y_train)

    print(f"Fitted KNN Classifer with {ALGORITHM=} and {N_NEIGHBORS=}")


    pred = best_KNN.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    display_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    display_confusion_matrix.plot()
    print(classification_report(y_test, pred))


# # NN Classifier
# ## About 87-89% Acc - okay
# ## WOW WITH JUST A BIT OF DATACLEANING, GETTING SOME MORE EXAMPLES AND FIXING A BUG THE ACCURACY SHOT UP - 94% 

# In[5]:


if TRAIN_NN:
    GRID_SEARCH_PARAMS = False

    '''Shuffle data for better resuliting'''
    RANDOM_ORDER_DATA = True 
    if RANDOM_ORDER_DATA:
        mutated_train = combined_train.sample(frac=1)
    else:
        mutated_train = combined_train

    X_train, y_train = split_features_labels(mutated_train)
    X_test, y_test = split_features_labels(combined_test)

    # Define the parameter distributions to sample from
    param_dist = {
        'hidden_layer_sizes' : [(i, j) for i in range(1, 15) for j in range(1, 15)],
        'solver': ['adam', 'lbfgs'],
        'activation': ['relu', 'logistic'],
        'alpha' : [0.0001, 0.0001, 0.00005]
    }

    # Initialize MLPClassifier with default values
    mlp = MLPClassifier(max_iter=1000)

    if GRID_SEARCH_PARAMS:
        random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1)

        # Train the classifier on your data
        random_search.fit(X_train, y_train)

        # Get the best hyperparameters from the search
        best_params = random_search.best_params_
        print("Best solver: ", best_params['solver'])
        print("Best activation: ", best_params['activation'])
        print("Layers: ", best_params['hidden_layer_sizes'] )
        print("Alpha: ", best_params['alpha'])

        # Use the best hyperparameters to initialize the MLPClassifier
        best_mlp = MLPClassifier(solver=best_params['solver'], activation=best_params['activation'], alpha=best_params['alpha'], hidden_layer_sizes=best_params['hidden_layer_sizes'], max_iter=1000)

        # Train the MLPClassifier on the training data
        best_mlp.fit(X_train, y_train)
    else:
        ''' If you want to skip searching for params. Train NN with values:
        solver: adam
        activation: relu
        layers: 11, 14
        alpha: 0.0001
        '''
        best_mlp = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(11, 14), max_iter=1000)
        best_mlp.fit(X_train, y_train)
        
    pred = best_mlp.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    display_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    display_confusion_matrix.plot()
    print(classification_report(y_test, pred))


# In[6]:


# print("Best solver: ", best_params['solver'])
# print("Best activation: ", best_params['activation'])
# print("Layers: ", best_params['hidden_layer_sizes'] )
# print("Alpha: ", best_params['alpha'])


# In[7]:


# # Use the best hyperparameters to initialize the MLPClassifier
# best_mlp = MLPClassifier(solver=best_params['solver'], activation=best_params['activation'], alpha=best_params['alpha'], hidden_layer_sizes=best_params['hidden_layer_sizes'], max_iter=1000)

# # Train the MLPClassifier on the training data
# best_mlp.fit(X_train, y_train)


# # SVC Grid Search Classifer
# ## Consistent 90% Sometimes 91%
# ## After replacing Warrior3 for Warrior2 (Is a hard pose to do, and W3 had too much overlap with the other classes + fixing a 20degree error in the pre-processing the accuracy gained went up by 5% !!)

# In[8]:


if TRAIN_SVM:
    RANDOM_ORDER_DATA = True 
    RANDOM_CV = False

    '''Shuffle data for better resuliting'''
    if RANDOM_ORDER_DATA:
        mutated_train = combined_train.sample(frac=1)
    else:
        mutated_train = combined_train

    X_train, y_train = split_features_labels(mutated_train)
    X_test, y_test = split_features_labels(combined_test)

    svm = SVC(kernel='rbf') 


    '''Tried search for gamma manually but it appears using scale is just better''' 
    n_features = X_train.shape[1]
    gamma_start = 1 / (n_features * max(X_train.var()))
    gamma_step = 0.005
    gamma_end = gamma_start + (10 * gamma_step)

    ### Doing Grid Search Now ###

    gamma_range = np.arange(gamma_start, gamma_end, gamma_step)
    C_range = [0.1, 1, 5, 10]
    # set the parameter grid for grid search
    param_grid = {
            'C': C_range,
            'gamma' : ['scale']
        }

    # perform grid search
    if RANDOM_CV:
        grid = RandomizedSearchCV(svm, cv=5, param_distributions=param_grid, verbose=1)
    else:
        grid = GridSearchCV(svm, cv=5, param_grid=param_grid, verbose=1)
    grid.fit(X_train, y_train)

    # best parameters and score
    print("Best parameters:", grid.best_params_)
    print("Best Score: ", grid.best_score_)

    best_svc = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], probability=True)

    best_svc.fit(X_train, y_train)

    ''' Plotting '''

    pred = best_svc.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    display_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    display_confusion_matrix.plot()
    print(classification_report(y_test, pred))


# # Video & Classifer Integration

# In[9]:


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# All landmark except for hand and face specific
RelevantLandmarks = list(mp_pose.PoseLandmark)[11:17] + list(mp_pose.PoseLandmark)[23:29]

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

landmarkStr = {
    mp_pose.PoseLandmark.NOSE : "NOSE",
    mp_pose.PoseLandmark.LEFT_EYE_INNER : "LEFT_EYE_INNER",
    mp_pose.PoseLandmark.LEFT_EYE : "LEFT_EYE",
    mp_pose.PoseLandmark.LEFT_EYE_OUTER : "LEFT_EYE_OUTER",
    mp_pose.PoseLandmark.RIGHT_EYE_INNER : "RIGHT_EYE_INNER",
    mp_pose.PoseLandmark.RIGHT_EYE : "RIGHT_EYE",
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER : "RIGHT_EYE_OUTER",
    mp_pose.PoseLandmark.LEFT_EAR : "LEFT_EAR",
    mp_pose.PoseLandmark.RIGHT_EAR : "RIGHT_EAR",
    mp_pose.PoseLandmark.MOUTH_LEFT : "MOUTH_LEFT",
    mp_pose.PoseLandmark.MOUTH_RIGHT : "MOUTH_RIGHT",
    mp_pose.PoseLandmark.LEFT_SHOULDER : "LEFT_SHOULDER",
    mp_pose.PoseLandmark.RIGHT_SHOULDER : "RIGHT_SHOULDER",
    mp_pose.PoseLandmark.LEFT_ELBOW : "LEFT_ELBOW",
    mp_pose.PoseLandmark.RIGHT_ELBOW : "RIGHT_ELBOW",
    mp_pose.PoseLandmark.LEFT_WRIST : "LEFT_WRIST",
    mp_pose.PoseLandmark.RIGHT_WRIST : "RIGHT_WRIST",
    mp_pose.PoseLandmark.LEFT_PINKY : "LEFT_PINKY",
    mp_pose.PoseLandmark.RIGHT_PINKY : "RIGHT_PINKY",
    mp_pose.PoseLandmark.LEFT_INDEX : "LEFT_INDEX",
    mp_pose.PoseLandmark.RIGHT_INDEX : "RIGHT_INDEX",
    mp_pose.PoseLandmark.LEFT_THUMB : "LEFT_THUMB",
    mp_pose.PoseLandmark.RIGHT_THUMB : "RIGHT_THUMB",
    mp_pose.PoseLandmark.LEFT_HIP : "LEFT_HIP",
    mp_pose.PoseLandmark.RIGHT_HIP : "RIGHT_HIP",
    mp_pose.PoseLandmark.LEFT_KNEE : "LEFT_KNEE",
    mp_pose.PoseLandmark.RIGHT_KNEE : "RIGHT_KNEE",
    mp_pose.PoseLandmark.LEFT_ANKLE : "LEFT_ANKLE",
    mp_pose.PoseLandmark.RIGHT_ANKLE : "RIGHT_ANKLE",
    mp_pose.PoseLandmark.LEFT_HEEL : "LEFT_HEEL",
    mp_pose.PoseLandmark.RIGHT_HEEL : "RIGHT_HEEL",
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX : "LEFT_FOOT_INDEX",
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX : "RIGHT_FOOT_INDEX"
}

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)    
    c = np.array(c)   
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    
    return angle 


# In[10]:

def classify_pose(example, classifer):
    example = pd.DataFrame(np.array(example).reshape(1, -1), columns=columns[1:])

    if classifer == 'KNN':
        probabilty_classes = best_KNN.predict_proba(example)
    elif classifer == 'Forest':
        probabilty_classes = best_forest.predict_proba(example)
    elif classifer == 'SVM':
        probabilty_classes = best_svc.predict_proba(example)
    elif classifer == 'NN':
        probabilty_classes = best_mlp.predict_proba(example)
    else:
        raise Exception("Please enter valid classifer. Currently only ('KNN' | 'Forest' | 'NN' | 'SVM')")

    
    prob_predicted_classes = [] 
    for class_idx, prob in enumerate(probabilty_classes[0]):
        prob_predicted_classes.append((classes[class_idx], prob))
        
    prob_predicted_classes.sort(key = lambda x: x[1], reverse=True)
    
    # Get the highest predicted class 
    predicted_class = prob_predicted_classes[0]
    
    return (predicted_class[0], predicted_class[1], sorted(prob_predicted_classes, key = lambda x: x[1], reverse=True))

# Sanity test classification 
classify_pose([8.390697637127042, 13.364568331384618, 16.49759248897499, 153.50000646379374, 173.20291493738577, 199.52935190007366, 179.00845878279233, 198.25172013734928], classifer='SVM')


# In[11]:


pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)

def extract_pca_2d(example):
    example = pd.DataFrame(np.array(example).reshape(1, -1), columns=columns[1:])
    return pca.transform(example)

def unnormalize_cords(x, y, fw, fh):
    return tuple(np.multiply([x, y], [fw, fh]).astype(int))


# # Getting the ideal angles into a dataframe
# ## Currently the method just gets the ideal images and finds the angles in those images and uses that as the ideal example.
# ## A more sophisiticated method may be getting a large amount of ideal examples and averaging over all of their angles to get a mean ideal angle for each key point. Ideal
# 
# # Edits: 
# ## There are now two ways of calculating the ideal angle - one is more sophisticated and splits up WarriorII and Tree into 2 different 'subclasses' one for facing left and one for facing right. This was done as the score someone would recieve for doign these exercises wouldn't be accurate as we previously didn't take into account which variation they did it
# 
# ## The old way of getting the average angles still exist and we simply just need to the toggle the `OLD_IDEAL_ANGLES` variable.

# In[12]:


OLD_IDEAL_ANGLES = False 

if OLD_IDEAL_ANGLES:
    # Depricated since we're now differentiate left and right warrior 
    ''' Might end up going back to this'''
    ideal_angles = pd.read_csv('../DemoImages/Ideal_Angles.csv', header=None)
    ideal_angles.columns = columns 

    ideal_angles_map_single = {pose : ideal_angles[ideal_angles['class'] == pose].values.tolist()[0][1:] for pose in classes}

    ''' Finding the average for all poses'''
    ideal_angles_map_average = {pose : combined_train[combined_train['class'] == pose_idx].mean(axis=0).tolist()[1:] for pose_idx, pose in enumerate(classes) }

else:
    ideal_angles = pd.read_csv('../DemoImages/joint_angles.csv', header=None)
    ideal_angles.columns = columns
    ideal_angles_map_single = {pose : ideal_angles[ideal_angles['class'] == pose].values.tolist()[0][1:] for pose in feedback_classes}

    ''' Finding the average for all poses except WarriorII and Tree. These need to be dealt in a special case explained in the next section'''
    ideal_angles_map_average = {pose : combined_train[combined_train['class'] == pose_idx].mean(axis=0).tolist()[1:] for pose_idx, pose in enumerate(classes) if pose not in {'WarriorII', 'Tree'}}

    ''' Splitting Warrior into Warrior L and Warrior R -- Setting the threshold to 170 gives the most equal 5050 split (180:196) and makes sense'''
    w2index = classes.index('WarriorII')
    Left_WarriorII = combined_train[(combined_train['class'] == w2index) & (combined_train['l_knee'] <= 170)]
    Right_WarriorII = combined_train[(combined_train['class'] == w2index) & (combined_train['l_knee'] > 170)]
    

    ideal_angles_map_average['WarriorII_R'] = Left_WarriorII.mean(axis=0).tolist()[1:]
    ideal_angles_map_average['WarriorII_L'] = Right_WarriorII.mean(axis=0).tolist()[1:]

    ''' Splitting Tree into Tree L and Tree R -> The split of is 100:250 (L:R) even while changing the degree threshold this value doesn't change much.'''
    ''' Potentially gonna have to split into Arms-Up Tree and Arms-Down Tree since the training data may include both poses with arms up and poses with arms down'''
    tree_idx = classes.index('Tree')

    Right_Tree_Down = combined_train[(combined_train['class'] == tree_idx) & (combined_train['l_knee'] <= 90) & (combined_train['l_shoulder'] <= 90) ]
    Left_Tree_Down = combined_train[(combined_train['class'] == tree_idx) & (combined_train['l_knee'] > 90) & (combined_train['l_shoulder'] <= 90) ]
    Right_Tree_Up = combined_train[(combined_train['class'] == tree_idx) & (combined_train['l_knee'] <= 90) & (combined_train['l_shoulder'] > 90) ]
    Left_Tree_Up = combined_train[(combined_train['class'] == tree_idx) & (combined_train['l_knee'] > 90) & (combined_train['l_shoulder'] > 90) ]


    # min_shoulder = combined_train[(combined_train['class'] == tree_idx) & (combined_train['l_knee'] > 120)]['l_shoulder'].min()
    # max_shoulder = combined_train[(combined_train['class'] == tree_idx) & (combined_train['l_knee'] > 120)]['l_shoulder'].max()

    # print(min_shoulder, max_shoulder)
    
    ideal_angles_map_average['Tree_L_D'] = Left_Tree_Down.mean(axis=0).tolist()[1:]
    ideal_angles_map_average['Tree_R_D'] = Right_Tree_Down.mean(axis=0).tolist()[1:]
    ideal_angles_map_average['Tree_L_U'] = Left_Tree_Up.mean(axis=0).tolist()[1:]
    ideal_angles_map_average['Tree_R_U'] = Right_Tree_Up.mean(axis=0).tolist()[1:]
    
ideal_angles_map = ideal_angles_map_average


# In[13]:


def cosine_similarity(angle1, angle2, difficulty):
    """
    Calculates the cosine similarity between two angles in degrees.
    """
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)
    cos_sim = np.cos(angle1_rad - angle2_rad)

    # Rescale the cosine similarity to be between 0 and 1, with values closer to 0 indicating a worse match
    sim_rescaled = (cos_sim + 1) / 2
    
    ''' Returing the score raised to the n'th power, where n determines how sensitive the score is.'''
    return sim_rescaled**difficulty

joint_idx_map = {
        0 : 'L Shoulder',
        1 : 'R Shoulder',
        2 : 'L Arm',
        3 : 'R Arm',
        4 : 'L Hip',
        5 : 'R Hip',
        6 : 'L Knee',
        7 : 'R Knee'
    }

# Define the x-axis values (the angle indexes)
x = joint_idx_map.values()

# Set up the figure and axis for the plot
fig, ax = plt.subplots()

# Plot the cosine similarity values for each pose (excluding 'Neutral')
if OLD_IDEAL_ANGLES:
    poses = classes
else:
    poses = feedback_classes

for pose in poses:
    if pose == 'Neutral':
        continue
    
    cosine_similarities = [cosine_similarity(single, avg, 1) for single, avg in zip(ideal_angles_map_single[pose], ideal_angles_map_average[pose])]
    ax.plot(x, cosine_similarities, label=pose)

# Set the plot title and labels for the x- and y-axes
ax.set_title('Cosine Similarity between Single and Average Ideal Angles for Each Pose (W2 & Tree Split)')
ax.set_xlabel('Joint')
ax.set_ylabel('Cosine Similarity')

# Set the y-axis limits to be between 0 and 1
ax.set_ylim([0.4, 1])
ax.legend()
plt.show()


# In[14]:


if OLD_IDEAL_ANGLES:
    poses = classes
else:
    poses = feedback_classes
print(list(joint_idx_map.values()))
for pose in poses:
    
    cosine_similarities = [cosine_similarity(single, avg, 1) for single, avg in zip(ideal_angles_map_single[pose], ideal_angles_map_average[pose])]
    print(pose)
    print(f'Single Example Angles: {ideal_angles_map_single[pose]}')
    print(f'Average over training set: {ideal_angles_map_average[pose]}')
    print(f'Cosine Similarities: {cosine_similarities}')
    print()


# # Some extra data post-processing
# ## WarriorII and Tree are unique in the sense that the ideal angles are different if you're doing the left or right variation of the exercise
# ### My method of fixing this is, for each example we detect if it's the left variation or right variation and create a seperate entry in the map depending on that.

# In[15]:


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


# # Caluclate the total score of a users score. Currently using trivial MSE -- will get more sophisitcated as development continues 

# In[16]:


''' This didn't work haha'''
def calculate_pose_score(ideal_example, user_example):

    tot = 0 
    for x in zip(ideal_example, user_example):

        tmp_score = 100 * (1 - (np.abs((x[0] - x[1]))/360))
        tot += tmp_score
        # print(tmp_score)
    
    # print(tot//len(ideal_example))

''' Using Cosine Similarty Method --- Not exactly the best implementation

Convert the two lists of angles into numpy arrays, and then we calculate the dot product of the two arrays and divide it by the product of their norms. 
This gives us the cosine similarity. Finally, we return a score between 0 and 100 by adding 1 to the cosine similarity and multiplying it by 50. 
The result will be a number between 0 and 100, where 100 indicates that the vectors are identical and 0 indicates that the vectors are completely dissimilar.'''
def calculate_pose_score(current_ideal_angles, pose_relevant_landmark_angles):
    current_ideal_angles = np.array(current_ideal_angles)
    pose_relevant_landmark_angles = np.array(pose_relevant_landmark_angles)

    cos_sim = np.dot(current_ideal_angles, pose_relevant_landmark_angles) / (np.linalg.norm(current_ideal_angles) * np.linalg.norm(pose_relevant_landmark_angles))

    return ((cos_sim + 1) * 50)/100


def cosine_similarity(angle1, angle2, difficulty):
    """
    Calculates the cosine similarity between two angles in degrees.
    """
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)
    cos_sim = np.cos(angle1_rad - angle2_rad)

    # Rescale the cosine similarity to be between 0 and 1, with values closer to 0 indicating a worse match
    sim_rescaled = (cos_sim + 1) / 2
    
    ''' Returing the score raised to the n'th power, where n determines how sensitive the score is.'''
    return sim_rescaled**difficulty


# ### Helper Functions

# In[26]:


''' Doesn't work as intended'''
def display_error_image(frame_width, frame_height):
    # Create a black image to use as the background
    image = np.zeros((frame_height, frame_width, 3), np.uint8)

    # Add text to the image
    text = "Cannot detect user, please get in frame"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 5
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = int((frame_width - text_size[0]) / 2)
    text_y = int((frame_height + text_size[1]) / 2)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Display the image
    cv2.imshow("ZenAI", image)
    time.sleep(1)
    
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
    
    display_text = 'Neutral' if classified_pose == 'Neutral' else f'Confidence: {classified_pose} {classified_pose_confidence*100:.2f}%'
    
    cv2.putText(image, display_text, unnormalize_cords(0.3, 0.1, frame_width, frame_height), cv2.FONT_HERSHEY_SIMPLEX, 2, (125, 0, 0), 2, cv2.LINE_AA) 

    return image

    
def gradient(value):
    # Map the pose score to a value between 0 and 1
    normalized_value = (value - 0) / (1 - 0)

    # Create a gradient between red (low values) and green (high values)
    red = int(max(0, 255 * (1 - 2 * normalized_value)))
    green = int(max(0, 255 * (2 * normalized_value)))
    blue = 0

    return (blue, green, red)

def calculate_center_of_gravity(pose_landmarks):
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    center_hip = (left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2

    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    center_shoulder = (left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2

    center_of_gravity = ((center_hip[0] + center_shoulder[0]) / 2, (center_hip[1] + center_shoulder[1]) / 2)

    return center_of_gravity

def create_skeleton_video_display(pose_landmarks, classified_pose, classified_pose_confidence, joint_angles_rounded, joint_cords, joint_scores, elapsed_time, frame_width, frame_height):
    # Create a black image to use as the background
    black_image = np.zeros((frame_height, frame_width, 3), np.uint8)

    joint_idx_map = {
        0 : 'L Shoulder',
        1 : 'R Shoulder',
        2 : 'L Arm',
        3 : 'R Arm',
        4 : 'L Hip',
        5 : 'R Hip',
        6 : 'L Knee',
        7 : 'R Knee'
    }

    # Render the skeleton on the black image
    mp_drawing.draw_landmarks(
        black_image, 
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,                
        mp_drawing.DrawingSpec(color=(50, 145, 168), circle_radius=2, thickness=2),
        mp_drawing.DrawingSpec(color=(209, 192, 42), circle_radius=2, thickness=2)
    )
    ''' Center of gravity'''
    center_of_gravity = calculate_center_of_gravity(pose_landmarks)

    # Draw an arrow from the center of gravity to the floor
    arrow_start = (int(center_of_gravity[0] * frame_width), int(center_of_gravity[1] * frame_height))
    arrow_end = (arrow_start[0], int(frame_height * 0.8))
    cv2.arrowedLine(black_image, arrow_start, arrow_end, (0, 255, 0), 5)
    
    ''' Only display the score information of a pose if it's not a Neutral pose'''
    if classified_pose != 'Neutral':
        ''' 
        Calculating the best, worst and average scores
    
        Calculation of the overall score may be changed to include a weighted mean
        where there will be a pre-defined weight vector of joints for each pose.
        '''
        total_pose_score = np.average(joint_scores)
        best_joint_idx = np.argmax(joint_scores)
        worst_joint_idx = np.argmin(joint_scores)

        # Draw red circles on the joints with size proportional to the joint score
        for i, (x, y) in enumerate(joint_cords):
            score = 100 * (1 - joint_scores[i])
            radius = int(score * 0.5)

            x, y = unnormalize_cords(x, y, frame_width, frame_height)
            black_image = cv2.circle(black_image, (int(x), int(y)), radius, (0, 0, 255), -1)

            ''' Highlighting the worst and best joint in the users pose
                & Display the joint score for each joint on the right side of the frame'''
            if i == best_joint_idx:
                cv2.putText(black_image, f'Best: {joint_idx_map[i]}: {100 - score:.2f}%', unnormalize_cords(0.70, 0.2 + (i / (1.25 * len(joint_cords))), frame_width, frame_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif i == worst_joint_idx:
                cv2.putText(black_image, f'Worst: {joint_idx_map[i]}: {100 - score:.2f}%', unnormalize_cords(0.70, 0.2 + (i / (1.25 * len(joint_cords))), frame_width, frame_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(black_image, f'{joint_idx_map[i]}: {100 - score:.2f}%', unnormalize_cords(0.70, 0.2 + (i / (1.25 * len(joint_cords))), frame_width, frame_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw the rectangle on the left side of the screen
        rect_x, rect_y = unnormalize_cords(0.05, 1, frame_width, frame_height)
        rect_width, rect_height = int(frame_width * 0.05), int(total_pose_score * frame_height)
        rect_color = gradient(total_pose_score)
        cv2.rectangle(black_image, (rect_x, rect_y - rect_height), (rect_x + rect_width, rect_y), rect_color, -1)

        ''' Displaying the average of all the users scores individual angles'''
        cv2.putText(black_image, f'Pose Score: {total_pose_score:.2f}%', unnormalize_cords(0.4, 0.1, frame_width, frame_height), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        ''' Displaying how long the user has held the pose'''
        cv2.putText(black_image, f'Time held: {elapsed_time:.2f} seconds', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        if elapsed_time > 5:
            black_image = cv2.rectangle(black_image, (0, 0), (frame_width, frame_height), (0, 255, 255), 10)



    # Write the joint angles on the black image
    for i, angle in enumerate(joint_angles_rounded):
        x, y = joint_cords[i]
        cv2.putText(black_image, angle, unnormalize_cords(x, y, frame_width, frame_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return black_image


# In[18]:


''' Plotting 2D Projection'''
def plot_live_data(points, ax):
    cords = [(extract_pca_2d(x[0]), x[1]) for x in points]

    class_labels = [point[1] for point in cords]

    unique_classes = set(class_labels)
    colors = {class_label: index for index, class_label in enumerate(unique_classes)}
    mat_colors = ['b', 'r', 'g', 'c', 'm', 'y']

    ax.clear()

    for class_label in unique_classes:
        x_v = [point[0][0][0] for idx, point in enumerate(cords) if class_labels[idx] == class_label]
        y_v = [point[0][0][1] for idx, point in enumerate(cords) if class_labels[idx] == class_label]

        c_ = [mat_colors[colors[class_label]] for _ in range(len(x_v))]
        ax.scatter(x_v, y_v, c=c_, label=class_label, s=3)

    ax.legend()


# # Main video / classification loop

# In[28]:


last_pose = 'Neutral'
start_time = 0

PLAY_FEED = False

CLASSIFIER = 'SVM'

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

VERBOSE = False
PLOT_2D = True

EXAMPLE_POSE_IMAGE = {i.replace('.jpeg', '') : cv2.imread(f"../DemoImages/{i}", 1) for i in os.listdir("../DemoImages")}
DEMO_POSE_VIDEOS = {f.replace('.mp4', '') : '../demo_videos/' + f  for f in os.listdir('../demo_videos')}
DISPLAY_DEMO = False
SCORE_DIFFICULTY = 10

frame_width = 1280
frame_height = 720

# This is just storing each frame in a real life demonstration as a projected point in 2D using pca 
# Which then decision boundaries will be plotted of the SVM model used to determin how someone moves around in a real life example
PCA_2D_POINTS = [] 


# In[38]:

if PLAY_FEED:
    with mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE, static_image_mode=False) as pose:
    
        if DISPLAY_DEMO:
            cap = cv2.VideoCapture(DEMO_POSE_VIDEOS['Tree'])
        else:
            cap = cv2.VideoCapture(0) # Captures live video feed 

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        while cap.isOpened():
            suc, frame = cap.read() 
            if not suc:
                print("Frame empty..")
                continue 
            
            #Recolor image 
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            try: 
                ''' Detect pose and extract angles / cords ''' 
                result = pose.process(image=image) 

                ''' If media pipe libray can't detect any poses, just display the normal image back.'''
                if not result.pose_landmarks:
                    image.flags.writeable = True 
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    correct_example_image = cv2.resize(EXAMPLE_POSE_IMAGE[classified_pose], (frame_width, frame_height))   
                    black_image = np.zeros((frame_height, frame_width, 3), np.uint8)
                    cv2.putText(black_image, "User not detected.", unnormalize_cords(0.1, 0.5, frame_width, frame_height), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(black_image, "Please get in frame.", unnormalize_cords(0.1, 0.6, frame_width, frame_height), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4, cv2.LINE_AA)
                    
                else:                    

                    pose_landmarks = result.pose_landmarks

                    joint_angles, joint_angles_rounded = extract_joint_angles(image, pose)
                    joint_cords = extract_joint_cords(pose_landmarks)
                    
                    ''' Classification of Pose''' 
                    classified_pose, classified_pose_confidence, prediction_probabilites = classify_pose(joint_angles, CLASSIFIER)

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

                    if classified_pose == 'Neutral':
                        angles_score = []
                    else:
                        if classified_pose == 'WarriorII':

                            if is_left_pose(classified_pose, joint_angles):
                                feedback_pose = 'WarriorII_R'
                            else:
                                feedback_pose = 'WarriorII_L'

                        elif classified_pose == 'Tree':
                            left = is_left_pose(classified_pose, joint_angles)
                            down = is_down_tree(joint_angles) 
                            right = not left 
                            up = not down 

                            if left and down:
                                feedback_pose = 'Tree_R_D'
                            elif left and up:
                                feedback_pose = 'Tree_R_U'
                            elif right and down:
                                feedback_pose =  'Tree_L_D'
                            elif right and up:
                                feedback_pose = 'Tree_L_U'              

                        cv2.putText(image, feedback_pose, unnormalize_cords(0.2, 0.8, frame_width, frame_height), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4, cv2.LINE_AA)

                        ideal_angles = ideal_angles_map[feedback_pose]

                        ''' Calculate score for current frame given the expected exercise'''
                        angles_score = [cosine_similarity(cur_angle, ideal_angle, SCORE_DIFFICULTY) for cur_angle, ideal_angle in zip(joint_angles, ideal_angles)]


                    ''' First Window '''
                    image = create_live_video_display(image, pose_landmarks, classified_pose, classified_pose_confidence, frame_width, frame_height)

                    ''' Second Window'''
                    black_image = create_skeleton_video_display(pose_landmarks, classified_pose, classified_pose_confidence, joint_angles_rounded, joint_cords, angles_score, elapsed_time, frame_width, frame_height)
                    
                    ''' Third Window '''
                    correct_example_image = cv2.resize(EXAMPLE_POSE_IMAGE[classified_pose], (frame_width, frame_height))            
                                
                    
                    if PLOT_2D:
                        ''' Extracting 2D PCA Points -- and plotting a new point on each frame to show how the users pose change over time,'''
                        top_pc, second_pc = extract_pca_2d(joint_angles)[0]
                        PCA_2D_POINTS.append((top_pc, second_pc))
                        # Update the scatter plot with the new point
                
                # Display the three windows side by side
                combined_videos = np.concatenate((image, black_image, correct_example_image), axis=1)
                # cv2.imshow("ZenAI", image)
                # cv2.imshow("ZenAI", black_image)
                # cv2.imshow("ZenAI", correct_example_image)
                cv2.imshow("ZenAI", combined_videos)
            except Exception:
                # display_error_image(frame_width, frame_height) 
                traceback.print_exc()
                continue

            # Closing the video capture  
            if cv2.waitKey(1) & 0xFF == ord('w'):
                break
        
    cap.release() 
    cv2.destroyAllWindows()


# In[73]:


import threading
import queue


# Define functions for each thread
def capture_thread(cap):
    while cap.isOpened():
        suc, frame = cap.read()
        if not suc:
            print("Frame empty..")
            continue
        image_queue.put(frame)

def process_thread():
    pose = mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE, static_image_mode=False)
    while True:
        frame = image_queue.get()
        #Recolor image 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        try:
            ''' Detect pose and extract angles / cords '''
            pose_landmarks = pose.process(image=image).pose_landmarks
            joint_angles, joint_angles_rounded = extract_joint_angles(image, pose)
            joint_cords = extract_joint_cords(pose_landmarks)
            ''' Classification of Pose'''
            classified_pose, classified_pose_confidence, prediction_probabilites = classify_pose(joint_angles, CLASSIFIER)
            ideal_angles = ideal_angles_map[classified_pose]
            ''' Calculate score for current frame given the expected exercise'''
            angles_score = [cosine_similarity(cur_angle, ideal_angle, 5) for cur_angle, ideal_angle in zip(joint_angles, ideal_angles)]
            ''' Extracting 2D PCA Points -- and plotting a new point on each frame to show how the users pose change over time,'''
            PCA_2D_POINTS.append((joint_angles, classified_pose))
            ''' We take the pose the user is doing to be neutral if either it's classified as a neutral pose or the confidence of the classified pose is < 70%'''
            classified_pose = classified_pose if classified_pose_confidence >= 0.80 else 'Neutral'
            ''' First Window '''
            image = create_live_video_display(image, pose_landmarks, classified_pose, classified_pose_confidence, frame_width, frame_height)
            ''' Second Window'''
            black_image = create_skeleton_video_display(pose_landmarks, classified_pose, classified_pose_confidence, joint_angles_rounded, joint_cords, angles_score, frame_width, frame_height)
            ''' Third Window '''
            correct_example_image = cv2.resize(EXAMPLE_POSE_IMAGE[classified_pose], (frame_width, frame_height))
            display_queue.put((image, black_image, correct_example_image))
        except Exception:
            # display_error_image(frame_width, frame_height)
            traceback.print_exc()
            continue

def display_thread():
    while True:
        images = display_queue.get()
        # Display the three windows side by side
        combined_videos = np.concatenate(images, axis=1)
        cv2.imshow("ZenAI", combined_videos)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break

# Set up queues for passing data between threads
image_queue = queue.Queue()
display_queue = queue.Queue()

# Start threads for each task
cap = cv2.VideoCapture(DEMO_POSE_VIDEOS['Cobra']) # Captures live video feed
capture_thread = threading.Thread(target=capture_thread, args=(cap,))
process_thread = threading.Thread(target=process_thread)
display_thread = threading.Thread(target=display_thread)
capture_thread.start()
process_thread.start()
display_thread.start()

# Wait for all threads to finish
capture_thread.join()
process_thread.join()
display_thread.join()


# In[1]:


fig, ax = plt.subplots()
ax.set_xlabel("Top PC", size=14)
ax.set_ylabel("Second PC", size=14)
ax.set_title('Projecting real life example of application onto 2D', size=16)
plot_live_data(PCA_2D_POINTS, ax)


# In[ ]:

def plot_2d_projection(points, real):
    cords = [(extract_pca_2d(x[0]), x[1]) for x in points]

    class_labels = [point[1] for point in cords]

    unique_classes = set(class_labels)
    colors = {class_label: index for index, class_label in enumerate(unique_classes)}
    mat_colors = ['b', 'r', 'g', 'c', 'm', 'y']


    fig, ax = plt.subplots()

    for class_label in unique_classes:
        x_v = [point[0][0][0] for idx, point in enumerate(cords) if class_labels[idx] == class_label]
        y_v = [point[0][0][1] for idx, point in enumerate(cords) if class_labels[idx] == class_label]

        c_ = [mat_colors[colors[class_label]] for _ in range(len(x_v))]
        ax.scatter(x_v, y_v, c=c_, label=class_label, s=3)

    ax.legend()
    plt.xlabel("Top PC", size=14)
    plt.ylabel("Second PC", size=14)   

    ''' and plotting how the different classified pose of a human looks like '''
    if real:
        plt.title('Projecting real life example of application onto 2D', size=16)
    else:
        plt.title('Projecting training data onto 2d', size=16)
    plt.show()

training_data = [(X_train.iloc[i].to_list(), classes[y_train.iloc[i]]) for i in range(len(X_train))]

# plot_2d_projection(PCA_2D_POINTS, real=True)
# plot_2d_projection(training_data, real=False)


# In[ ]:

def plot_2d_projection(points, color, real):
    cords = [(extract_pca_2d(x[0]), x[1]) for x in points]

    class_labels = [point[1] for point in cords]

    unique_classes = set(class_labels)
    colors = {class_label: index for index, class_label in enumerate(unique_classes)}
    mat_colors = ['b', 'r', 'g', 'c', 'm', 'y']

    for class_label in unique_classes:
        x_v = [point[0][0][0] for idx, point in enumerate(cords) if class_labels[idx] == class_label]
        y_v = [point[0][0][1] for idx, point in enumerate(cords) if class_labels[idx] == class_label]

        c_ = [color for _ in range(len(x_v))]
        ax.scatter(x_v, y_v, c=c_, label=class_label, s=3)

fig, ax = plt.subplots()

plot_2d_projection(PCA_2D_POINTS, 'b', real=True)
combined_testing_data = [(X_test.iloc[i].to_list(), classes[y_test.iloc[i]]) for i in range(len(X_test))]
plot_2d_projection(combined_testing_data, 'r', real=False)

ax.legend()
plt.xlabel("Top PC", size=14)
plt.ylabel("Second PC", size=14)
plt.title('Projecting real life and test data onto 2D', size=16)
# plt.show()


# In[ ]:




