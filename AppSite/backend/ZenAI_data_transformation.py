import os
import pandas as pd 

os.chdir('../../Dataset')
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

feedback_classes = classes + ['WarriorII_L', 'WarriorII_R', 'Tree_L_D', 'Tree_R_D', 'Tree_L_U', 'Tree_R_U']
feedback_classes.remove('Neutral')
feedback_classes.remove('Tree')
feedback_classes.remove('WarriorII')

joint_idx_map = {
        0 : 'Left Shoulder',
        1 : 'Right Shoulder',
        2 : 'Left Arm',
        3 : 'Right Arm',
        4 : 'Left Hip',
        5 : 'Right Hip',
        6 : 'Left Knee',
        7 : 'Right Knee'
    }