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

'''
    Dictionary that provides more specific vocabulary for poses and their joints, this will make the feedback sound more natural.

    Also includes tips for common problems with certain body parts in certain poses 

    Format:(-1, 1, Common Tips)

    Also adding some redundacy to make the code easier to handle (Such as duplicating Tree_L_D and Tree_R_D) 
'''
vocab_dict = { 
    'WarriorII_R' : {
        0 : ('raising', 'lowering', 'Your arms may be straight, but they should be parallel to the ground.'),
        2 : ('straightening', 'bending', 'Your arms may be parallel to the ground, but they should be straight.'),
        4 : ('sinking', 'raising', 'Your hips may not be low enough. Try to make a right angle with your bent leg.'),
        6 : ('bending', 'straightening', 'Your stance may not be wide enough, you should make your knee perpendicular to the floor.'),
        1 : ('raising', 'lowering', 'Your arms may be straight, but they should be parallel to the ground.'),
        3 : ('straightening', 'bending', 'Your arms may be parallel to the ground, but they should be straight.'),
        5 : ('sinking', 'raising', 'Your hips may not be low enough. Try to make a right angle with your bent leg.'),
        7 : ('bending', 'straightening', 'Your stance may not be wide enough, you should make your knee perpendicular to the floor.')
    },
    'WarriorII_L' : {
        0 : ('raising', 'lowering', 'Your arms may be straight, but they should be parallel to the ground.'),
        2 : ('straightening', 'bending', 'Your arms may be parallel to the ground, but they should be straight.'),
        4 : ('sinking', 'raising', 'Your hips may not be low enough. Try to make a right angle with your bent leg.'),
        6 : ('bending', 'straightening', 'Your stance may not be wide enough, you should make your knee perpendicular to the floor.'),
        1 : ('raising', 'lowering', 'Your arms may be straight, but they should be parallel to the ground.'),
        3 : ('straightening', 'bending', 'Your arms may be parallel to the ground, but they should be straight.'),
        5 : ('sinking', 'raising', 'Your hips may not be low enough. Try to make a right angle with your bent leg.'),
        7 : ('bending', 'straightening', 'Your stance may not be wide enough, you should make your knee perpendicular to the floor.')
    },
    'Chair' : {
        0 : ('straightening', 'bending', 'You may not be extending your arms enough, you should try to straightening them as much as you can.'),
        2 : ('raising', 'lowering', 'You may not be raising your arms enough, try to think about keeping your biceps inline with your ears and make sure to relax your shoulders.'),
        4 : ('bending', 'straightening', 'You may be leaning too far foward, you want to form a right angle between your torso and your thighs.'),
        6 : ('bending', 'straightening', 'You knees may not be bent the correct amount, you should aim to bending your knees over your feet but not over your toes. '),
        1 : ('raising', 'lowering', 'You may not be raising your arms enough, try to think about keeping your biceps inline with your ears and make sure to relax your shoulders.'),
        3 : ('straightening', 'bending', 'You may not be extending your arms enough, you should try to straightening them as much as you can.'),
        5 : ('bending', 'straightening', 'You may be leaning too far foward, you want to form a right angle between your torso and your thighs.'),
        7 : ('bending', 'straightening', 'You knees may not be bent the correct amount, you should aim to bending your knees over your feet but not over your toes. ')
    },
    'DownDog' : {
        0 : ('walking', 'lowering', 'Your shoulders may not be straight, you can try to fix this by rasing your hips higher or tucking your torso and head towards yourself.'),
        2 : ('straightening', 'bending', 'Your arms may not be straight, you can try to fix this by walking up the floor with your fingers. '),
        4 : ('bending', 'straightening', 'Your hips may be bent too much, you can try to fix this by either moving your feet back or hands forward. '),
        6 : ('bending', 'straightening', 'Your knees may be bent too much, you can try to walk your feet up untill your heels can touch the floor and straightening your legs. '),
        1 : ('walking', 'lowering', 'Your shoulders may not be straight, you can try to fix this by rasing your hips higher or tucking your torso and head towards yourself.'),
        3 : ('straightening', 'bending', 'Your arms may not be straight, you can try to fix this by walking up the floor with your fingers. '),
        5 : ('bending', 'straightening', 'Your hips may be bent too much, you can try to fix this by either moving your feet back or hands forward. '),
        7 : ('bending', 'straightening', 'Your knees may be bent too much, you can try to walk your feet up untill your heels can touch the floor and straightening your legs. ')
    },
    'Cobra' : {
        0 : ('raising', 'lowering', 'Your shoulders may be too far out, you should try to keep your elbows tucked to your torso. '),
        2 : ('straightening', 'bending', 'Your arms may be bent too much, you should try to push off the floor and have a slight bent in your arms. '),
        4 : ('raising', 'lowering', 'Your chest may be too close to the ground, you should try raising your chest slightly and keep your pelvis on the ground. '),
        6 : ('straightening', 'bending', 'Your legs may be bent, for this exercise you should aim to keep your legs completely flat on the floor. '),
        1 : ('raising', 'lowering', 'Your shoulders may be too far out, you should try to keep your elbows tucked to your torso. '),
        3 : ('straightening', 'bending', 'Your arms may be bent too much, you should try to push off the floor and have a slight bent in your arms. '),
        5 : ('raising', 'lowering', 'Your chest may be too close to the ground, you should try raising your chest slightly and keep your pelvis on the ground. '),
        7 : ('straightening', 'bending', 'Your legs may be bent, for this exercise you should aim to keep your legs completely flat on the floor. ')
    },
    'Tree_R_D' : {
        0 : ('raising', 'lowering', 'Your elbows may be too flared, you can try to fix this by straightening your arms and bringing elbows shoulder width apart. '),
        2 : ('straightening', 'bending', 'Your arms may not be bent enough, you can try to fix this be putting your palms together and bringing your hands to your chest'),
        4 : ('raising', 'lowering', 'Your knee may be raisingd too high causing your hip to bending too much, you should try to lowering your knee while keeping it bent. '),
        6 : ('straightening', 'bending', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. '),
        1 : ('raising', 'lowering', 'Your elbows may be too flared, you can try to fix this by straightening your arms and bringing elbows shoulder width apart. '),
        3 : ('straightening', 'bending', 'Your arms may not be bent enough, you can try to fix this be putting your palms together and bringing your hands to your chest'),
        5 : ('raising', 'lowering', 'Your knee may be raisingd too high causing your hip to bending too much, you should try to lowering your knee while keeping it bent. '),
        7 : ('straightening', 'bending', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. ')
    },
    'Tree_R_U' : {
        0 : ('raising', 'lowering', 'Your elbows may be too flared, you should aim to have your elbows pointing down and to the side. '),
        2 : ('straightening', 'bending', 'Your arms may be bent too much, you can try to fix this be putting your palms together and raising your hands above your head'),
        4 : ('raising', 'lowering', 'Your knee may be raisingd too high causing your hip to bending too much, you should try to lowering your knee while keeping it bent. '),
        6 : ('straightening', 'bending', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. '),
        1 : ('raising', 'lowering', 'Your elbows may be too flared, you should aim to have your elbows pointing down and to the side. '),
        3 : ('straightening', 'bending', 'Your arms may be bent too much, you can try to fix this be putting your palms together and raising your hands above your head'),
        5 : ('raising', 'lowering', 'Your knee may be raisingd too high causing your hip to bending too much, you should try to lowering your knee while keeping it bent. '),
        7 : ('straightening', 'bending', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. ')
    },
    'Tree_L_D' : {
        0 : ('raising', 'lowering', 'Your elbows may be too flared, you can try to fix this by straightening your arms and bringing elbows shoulder width apart. '),
        2 : ('straightening', 'bending', 'Your arms may not be bent enough, you can try to fix this be putting your palms together and bringing your hands to your chest'),
        4 : ('raising', 'lowering', 'Your knee may be raisingd too high causing your hip to bending too much, you should try to lowering your knee while keeping it bent. '),
        6 : ('straightening', 'bending', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. '),
        1 : ('raising', 'lowering', 'Your elbows may be too flared, you can try to fix this by straightening your arms and bringing elbows shoulder width apart. '),
        3 : ('straightening', 'bending', 'Your arms may not be bent enough, you can try to fix this be putting your palms together and bringing your hands to your chest'),
        5 : ('raising', 'lowering', 'Your knee may be raisingd too high causing your hip to bending too much, you should try to lowering your knee while keeping it bent. '),
        7 : ('straightening', 'bending', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. ')
    },
    'Tree_L_U' : {
        0 : ('raising', 'lowering', 'Your elbows may be too flared, you should aim to have your elbows pointing down and to the side. '),
        2 : ('straightening', 'bending', 'Your arms may be bent too much, you can try to fix this be putting your palms together and raising your hands above your head'),
        4 : ('raising', 'lowering', 'Your knee may be raisingd too high causing your hip to bending too much, you should try to lowering your knee while keeping it bent. '),
        6 : ('straightening', 'bending', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. '),
        1 : ('raising', 'lowering', 'Your elbows may be too flared, you should aim to have your elbows pointing down and to the side. '),
        3 : ('straightening', 'bending', 'Your arms may be bent too much, you can try to fix this be putting your palms together and raising your hands above your head'),
        5 : ('raising', 'lowering', 'Your knee may be raisingd too high causing your hip to bending too much, you should try to lowering your knee while keeping it bent. '),
        7 : ('straightening', 'bending', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. ')
    }
}
