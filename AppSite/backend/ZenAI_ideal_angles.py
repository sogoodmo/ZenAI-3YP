import pandas as pd 

def calculate_ideal_angles(classes, combined_train):

    ''' Finding the average for all poses except WarriorII and Tree. These need to be dealt in a special case explained in the next section'''
    ideal_angles_map_average = {pose : combined_train[combined_train['class'] == pose_idx].mean(axis=0).tolist()[1:] for pose_idx, pose in enumerate(classes) if pose not in {'WarriorII', 'Tree'}}

    ''' Splitting Warrior into Warrior L and Warrior R -- Setting the threshold to 170 gives the most equal 5050 split (180:196) and makes sense'''
    w2index = classes.index('WarriorII')
    Right_WarriorII = combined_train[(combined_train['class'] == w2index) & (combined_train['l_knee'] > 170)]
    Left_WarriorII = combined_train[(combined_train['class'] == w2index) & (combined_train['l_knee'] <= 170)]


    ideal_angles_map_average['WarriorII_R'] = Right_WarriorII.mean(axis=0).tolist()[1:]
    ideal_angles_map_average['WarriorII_L'] = Left_WarriorII.mean(axis=0).tolist()[1:]

    ''' Splitting Tree into Tree L and Tree R -> The split of is 100:250 (L:R) even while changing the degree threshold this value doesn't change much.'''
    ''' Potentially gonna have to split into Arms-Up Tree and Arms-Down Tree since the training data may include both poses with arms up and poses with arms down'''
    tree_idx = classes.index('Tree')

    Left_Tree_Down = combined_train[(combined_train['class'] == tree_idx) & (combined_train['l_knee'] <= 90) & (combined_train['l_shoulder'] <= 90) ]
    Right_Tree_Down = combined_train[(combined_train['class'] == tree_idx) & (combined_train['l_knee'] > 90) & (combined_train['l_shoulder'] <= 90) ]
    Left_Tree_Up = combined_train[(combined_train['class'] == tree_idx) & (combined_train['l_knee'] <= 90) & (combined_train['l_shoulder'] > 90) ]
    Right_Tree_Up = combined_train[(combined_train['class'] == tree_idx) & (combined_train['l_knee'] > 90) & (combined_train['l_shoulder'] > 90) ]


    ideal_angles_map_average['Tree_R_D'] = Right_Tree_Down.mean(axis=0).tolist()[1:]
    ideal_angles_map_average['Tree_L_D'] = Left_Tree_Down.mean(axis=0).tolist()[1:]
    ideal_angles_map_average['Tree_R_U'] = Right_Tree_Up.mean(axis=0).tolist()[1:]
    ideal_angles_map_average['Tree_L_U'] = Left_Tree_Up.mean(axis=0).tolist()[1:]

    ideal_angles_map = ideal_angles_map_average

    return ideal_angles_map 