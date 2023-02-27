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
        0 : ('lowering', 'raising', 'Your arms may not be parallel to the ground, You can try to fix this by making them more parallel.'),
        2 : ('bending', 'straightening', 'Your arms may not be straight, You can try to fix this by extending your arms as far as you can.'),
        4 : ('raising', 'sinking', 'Your hips may not be bent the correct amount, You can try to fix this by adjusting your hips vertically untill your bent leg is at a right angle.'),
        6 : ('straightening', 'bending', 'Your knees may not be bent the correct amount, You can try to fix this by making sure your knee perpendicular to the floor.'),
        1 : ('lowering', 'raising', 'Your arms may not be parallel to the ground, You can try to fix this by making them more parallel.'),
        3 : ('bending', 'straightening', 'Your arms may not be straight, You can try to fix this by extending your arms as far as you can.'),
        5 : ('raising', 'sinking', 'Your hips may not be bent the correct amount, You can try to fix this by adjusting your hips vertically untill your bent leg is at a right angle.'),
        7 : ('straightening', 'bending', 'Your knees may not be bent the correct amount, You can try to fix this by making sure your knee perpendicular to the floor.'),
    },
    'WarriorII_L' : {
        0 : ('lowering', 'raising', 'Your arms may not be parallel to the ground, You can try to fix this by making them more parallel.'),
        2 : ('bending', 'straightening', 'Your arms may not be straight, You can try to fix this by extending your arms as far as you can.'),
        4 : ('raising', 'sinking', 'Your hips may not be bent the correct amount, You can try to fix this by adjusting your hips vertically untill your bent leg is at a right angle.'),
        6 : ('straightening', 'bending', 'Your knees may not be bent the correct amount, You can try to fix this by making sure your knee perpendicular to the floor.'),
        1 : ('lowering', 'raising', 'Your arms may not be parallel to the ground, You can try to fix this by making them more parallel.'),
        3 : ('bending', 'straightening', 'Your arms may not be straight, You can try to fix this by extending your arms as far as you can.'),
        5 : ('raising', 'sinking', 'Your hips may not be bent the correct amount, You can try to fix this by adjusting your hips vertically untill your bent leg is at a right angle.'),
        7 : ('straightening', 'bending', 'Your knees may not be bent the correct amount, You can try to fix this by making sure your knee perpendicular to the floor.'),
    },
    'Chair' : {
        0 : ('lowering', 'raising', 'You may not be raising your arms the correct amount, You can try to fix this by keeping your biceps inline with your ears and relaxing your shoulders.'),
        2 : ('bending', 'straightening', 'You may not be extending your arms enough, You can try to fix this by straightening them as much as you can and extending your fingers.'),
        4 : ('straightening', 'bending', 'You may be leaning too far foward, You can try to fix this by forming a right angle between your torso and your thighs.'),
        6 : ('straightening', 'bending', 'You knees may not be bent the correct amount, You can try to fix this by bending your knees over your feet but not over your toes.'),
        1 : ('lowering', 'raising', 'You may not be raising your arms the correct amount, You can try to fix this by keeping your biceps inline with your ears and relaxing your shoulders.'),
        3 : ('bending', 'straightening', 'You may not be extending your arms enough, You can try to fix this by straightening them as much as you can and extending your fingers.'),
        5 : ('straightening', 'bending', 'You may be leaning too far foward, You can try to fix this by forming a right angle between your torso and your thighs.'),
        7 : ('straightening', 'bending', 'You knees may not be bent the correct amount, You can try to fix this by bending your knees over your feet but not over your toes.'),
    },
    'DownDog' : {
        0 : ('walking down', 'walking up', 'Your shoulders may not be straight, You can try to fix this by rasing your hips higher or tucking your torso and head towards yourself.'),
        2 : ('bending', 'straightening', 'Your arms may not be straight, You can try to fix this by walking up the floor with your fingers.'),
        4 : ('straightening', 'bending', 'Your hips may not be bent the correct amount, You can try to fix this by either moving your feet back or hands forward.'),
        6 : ('bending', 'straightening', 'Your knees may be bent too much, You can try to fix this by walking your feet up untill your heels can touch the floor and straightening your legs.'),
        1 : ('walking down', 'walking up', 'Your shoulders may not be straight, You can try to fix this by rasing your hips higher or tucking your torso and head towards yourself.'),
        3 : ('bending', 'straightening', 'Your arms may not be straight, You can try to fix this by walking up the floor with your fingers.'),
        5 : ('straightening', 'bending', 'Your hips may not be bent the correct amount, You can try to fix this by either moving your feet back or hands forward.'),
        7 : ('bending', 'straightening', 'Your knees may be bent too much, You can try to fix this by walking your feet up untill your heels can touch the floor and straightening your legs.'),
    },
    'Cobra' : {
        0 : ('raising', 'lowering', 'Your shoulders may not be close enough to your torso, You can fix this by tucking your elbows into your torso.'),
        2 : ('straightening', 'bending', 'Your arms may not have a slight bend in them, You can fix this by pushing off the floor and having a slight bend in your arms.'),
        4 : ('raising', 'lowering', 'Your chest may be too close to the ground, You can try to fix this by raising your chest slightly and pushing your pelvis into the ground. '),
        6 : ('bending', 'straightening', 'Your legs may be bent too much, You can try to fix this activating your glutes to keep your legs flat on the floor.'),
        1 : ('raising', 'lowering', 'Your shoulders may not be close enough to your torso, You can fix this by tucking your elbows into your torso.'),
        3 : ('straightening', 'bending', 'Your arms may not have a slight bend in them, You can fix this by pushing off the floor and having a slight bend in your arms.'),
        5 : ('raising', 'lowering', 'Your chest may be too close to the ground, You can try to fix this by raising your chest slightly and pushing your pelvis into the ground. '),
        7 : ('bending', 'straightening', 'Your legs may be bent too much, You can try to fix this activating your glutes to keep your legs flat on the floor.'),
    },
    'Tree_R_D' : {
        0 : ('raising', 'lowering', 'Your elbows may be too flared, You can try to fix this by pointing your pointing down and to the side. '),
        2 : ('bending', 'straightening', 'Your arms may not be bent enough, You can try to fix this be putting your palms together and bringing your hands to your chest.'),
        4 : ('lowering', 'raising', 'Your knee may be raised too high causing your hip to bending too much, You can try to fix this by lowering your knee while keeping it bent.'),
        6 : ('straightening', 'bending', 'Your knee may not be bent enough, You can try to fix this by placing your foot as high up as you can on your standing leg.'),
        1 : ('raising', 'lowering', 'Your elbows may be too flared, You can try to fix this by straightening your arms and bringing elbows shoulder width apart. '),
        3 : ('bending', 'straightening', 'Your arms may not be bent enough, You can try to fix this be putting your palms together and bringing your hands to your chest.'),
        5 : ('lowering', 'raising', 'Your knee may be raised too high causing your hip to bending too much, You can try to fix this by lowering your knee while keeping it bent.'),
        7 : ('straightening', 'bending', 'Your knee may not be bent enough, You can try to fix this by placing your foot as high up as you can on your standing leg.'),
    },
    'Tree_L_D' : {
        0 : ('raising', 'lowering', 'Your elbows may be too flared, You can try to fix this by pointing your pointing down and to the side. '),
        2 : ('bending', 'straightening', 'Your arms may not be bent enough, You can try to fix this be putting your palms together and bringing your hands to your chest.'),
        4 : ('lowering', 'raising', 'Your knee may be raised too high causing your hip to bending too much, You can try to fix this by lowering your knee while keeping it bent.'),
        6 : ('straightening', 'bending', 'Your knee may not be bent enough, You can try to fix this by placing your foot as high up as you can on your standing leg.'),
        1 : ('raising', 'lowering', 'Your elbows may be too flared, You can try to fix this by straightening your arms and bringing elbows shoulder width apart. '),
        3 : ('bending', 'straightening', 'Your arms may not be bent enough, You can try to fix this be putting your palms together and bringing your hands to your chest.'),
        5 : ('lowering', 'raising', 'Your knee may be raised too high causing your hip to bending too much, You can try to fix this by lowering your knee while keeping it bent.'),
        7 : ('straightening', 'bending', 'Your knee may not be bent enough, You can try to fix this by placing your foot as high up as you can on your standing leg.'),
    },
    'Tree_R_U' : {
        0 : ('lowering', 'raising', 'Your elbows may be too flared, You can try to fix this by straightening your arms and bringing elbows shoulder width apart. '),
        2 : ('bending', 'straightening', 'Your arms may be bent too much, You can try to fix this be putting your palms together and raising your hands above your head.'),
        4 : ('lowering', 'raising', 'Your knee may be raised too high causing your hip to bending too much, You can try to fix this by lowering your knee while keeping it bent.'),
        6 : ('straightening', 'bending', 'Your knee may not be bent enough, You can try to fix this by placing your foot as high up as you can on your standing leg.'),
        1 : ('lowering', 'raising', 'Your elbows may be too flared, You can try to fix this by straightening your arms and bringing elbows shoulder width apart. '),
        3 : ('bending', 'straightening', 'Your arms may be bent too much, You can try to fix this be putting your palms together and raising your hands above your head.'),
        5 : ('lowering', 'raising', 'Your knee may be raised too high causing your hip to bending too much, You can try to fix this by lowering your knee while keeping it bent.'),
        7 : ('straightening', 'bending', 'Your knee may not be bent enough, You can try to fix this by placing your foot as high up as you can on your standing leg.'),
    },
    'Tree_L_U' : {
        0 : ('lowering', 'raising', 'Your elbows may be too flared, You can try to fix this by straightening your arms and bringing elbows shoulder width apart. '),
        2 : ('bending', 'straightening', 'Your arms may be bent too much, You can try to fix this be putting your palms together and raising your hands above your head.'),
        4 : ('lowering', 'raising', 'Your knee may be raised too high causing your hip to bending too much, You can try to fix this by lowering your knee while keeping it bent.'),
        6 : ('straightening', 'bending', 'Your knee may not be bent enough, You can try to fix this by placing your foot as high up as you can on your standing leg.'),
        1 : ('lowering', 'raising', 'Your elbows may be too flared, You can try to fix this by straightening your arms and bringing elbows shoulder width apart. '),
        3 : ('bending', 'straightening', 'Your arms may be bent too much, You can try to fix this be putting your palms together and raising your hands above your head.'),
        5 : ('lowering', 'raising', 'Your knee may be raised too high causing your hip to bending too much, You can try to fix this by lowering your knee while keeping it bent.'),
        7 : ('straightening', 'bending', 'Your knee may not be bent enough, You can try to fix this by placing your foot as high up as you can on your standing leg.'),
    }
}
