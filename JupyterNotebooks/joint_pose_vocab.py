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
'''
vocab_dict = { 
    'WarriorII' : {
        '0' : ('raise', 'lower', 'Your arms may be staight, but they should be parallel to the ground.'),
        '1' : ('straighten', 'bend', 'Your arms may be parallel to the ground, but they should be straight.'),
        '2' : ('sink', 'raise', 'Your hips may not be low enough. Try to make a right angle with your bent leg.'),
        '3' : ('bend', 'straighten', 'Your stance may not be wide enough, you should make your knee perpendicular to the floor.')
        '4' : ('raise', 'lower', 'Your arms may be staight, but they should be parallel to the ground.'),
        '5' : ('straighten', 'bend', 'Your arms may be parallel to the ground, but they should be straight.'),
        '6' : ('sink', 'raise', 'Your hips may not be low enough. Try to make a right angle with your bent leg.'),
        '7' : ('bend', 'straighten', 'Your stance may not be wide enough, you should make your knee perpendicular to the floor.')
    },
    'Chair' : {
        '0' : ('raise', 'lower', 'You may not be raising your arms enough, try to think about keeping your biceps inline with your ears and make sure to relax your shoulders.'),
        '1' : ('straighten', 'bend', 'You may not be extending your arms enough, you should try to straighten them as much as you can.'),
        '2' : ('bend', 'straighten', 'You may be leaning too far foward, you want to form a right angle between your torso and your thighs.'),
        '3' : ('bend', 'straighten', 'You knees may not be bent the correct amount, you should aim to bend your knees over your feet but not over your toes. ')
        '4' : ('raise', 'lower', 'You may not be raising your arms enough, try to think about keeping your biceps inline with your ears and make sure to relax your shoulders.'),
        '5' : ('straighten', 'bend', 'You may not be extending your arms enough, you should try to straighten them as much as you can.'),
        '6' : ('bend', 'straighten', 'You may be leaning too far foward, you want to form a right angle between your torso and your thighs.'),
        '7' : ('bend', 'straighten', 'You knees may not be bent the correct amount, you should aim to bend your knees over your feet but not over your toes. ')
    },
    'DownDog' : {
        '0' : ('walk', 'lower', 'Your shoulders may not be straight, you can try to fix this by rasing your hips higher or tucking your torso and head towards yourself.'),
        '1' : ('straighten', 'bend', 'Your arms may not be straight, you can try to fix this by walking up the floor with your fingers. '),
        '2' : ('bend', 'straighten', 'Your hips may be bent too much, you can try to fix this by either moving your feet back or hands forward. '),
        '3' : ('bend', 'straighten', 'Your knees may be bent too much, you can try to walk your feet up untill your heels can touch the floor and straighten your legs. ')
        '4' : ('walk', 'lower', 'Your shoulders may not be straight, you can try to fix this by rasing your hips higher or tucking your torso and head towards yourself.'),
        '5' : ('straighten', 'bend', 'Your arms may not be straight, you can try to fix this by walking up the floor with your fingers. '),
        '6' : ('bend', 'straighten', 'Your hips may be bent too much, you can try to fix this by either moving your feet back or hands forward. '),
        '7' : ('bend', 'straighten', 'Your knees may be bent too much, you can try to walk your feet up untill your heels can touch the floor and straighten your legs. ')
    },
    'Cobra' : {
        '0' : ('raise', 'lower', 'Your shoulders may be too far out, you should try to keep your elbows tucked to your torso. '),
        '1' : ('straighten', 'bend', 'Your arms may be bent too much, you should try to push off the floor and have a slight bent in your arms. '),
        '2' : ('raise', 'lower', 'Your chest may be too close to the ground, you should try raise your chest slightly and keep your pelvis on the ground. '),
        '3' : ('straighten', 'bend', 'Your legs may be bent, for this exercise you should aim to keep your legs completely flat on the floor. ')
        '4' : ('raise', 'lower', 'Your shoulders may be too far out, you should try to keep your elbows tucked to your torso. '),
        '5' : ('straighten', 'bend', 'Your arms may be bent too much, you should try to push off the floor and have a slight bent in your arms. '),
        '6' : ('raise', 'lower', 'Your chest may be too close to the ground, you should try raise your chest slightly and keep your pelvis on the ground. '),
        '7' : ('straighten', 'bend', 'Your legs may be bent, for this exercise you should aim to keep your legs completely flat on the floor. ')
    },
    'Tree_D' : {
        '0' : ('raise', 'lower', 'Your elbows may be too flared, you can try to fix this by straighten your arms and bringing elbows shoulder width apart. '),
        '1' : ('straighten', 'bend', 'Your arms may not be bent enough, you can try to fix this be putting your palms together and bringing your hands to your chest'),
        '2' : ('raise', 'lower', 'Your knee may be raised too high causing your hip to bend too much, you should try to lower your knee while keeping it bent. '),
        '3' : ('straighten', 'bend', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. ')
        '4' : ('raise', 'lower', 'Your elbows may be too flared, you can try to fix this by straighten your arms and bringing elbows shoulder width apart. '),
        '5' : ('straighten', 'bend', 'Your arms may not be bent enough, you can try to fix this be putting your palms together and bringing your hands to your chest'),
        '6' : ('raise', 'lower', 'Your knee may be raised too high causing your hip to bend too much, you should try to lower your knee while keeping it bent. '),
        '7' : ('straighten', 'bend', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. ')
    },
    'Tree_D' : {
        '0' : ('raise', 'lower', 'Your elbows may be too flared, you should aim to have your elbows pointing down and to the side. '),
        '1' : ('straighten', 'bend', 'Your arms may be bent too much, you can try to fix this be putting your palms together and raising your hands above your head'),
        '2' : ('raise', 'lower', 'Your knee may be raised too high causing your hip to bend too much, you should try to lower your knee while keeping it bent. '),
        '3' : ('straighten', 'bend', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. ')
        '4' : ('raise', 'lower', 'Your elbows may be too flared, you should aim to have your elbows pointing down and to the side. '),
        '5' : ('straighten', 'bend', 'Your arms may be bent too much, you can try to fix this be putting your palms together and raising your hands above your head'),
        '6' : ('raise', 'lower', 'Your knee may be raised too high causing your hip to bend too much, you should try to lower your knee while keeping it bent. '),
        '7' : ('straighten', 'bend', 'Your knee may not be bent enough, you can try to fix this by placing your foot as high up as you can on your standing leg. ')
    }
}
