import os 

for i in range(120, 150):
    os.system(rf"python3 -m irl.mack.run_mack_aairl \
                 --env simple_path_finding_jaciii \
                 --seed {i} \
                 --expert_path data/goal-done-test/checkpoint01100-1000tra.pkl\
                 --logdir ./data/goal-done-test")
                 #--expert_path checkpoint00100-1000dtra.pkl\
                 #--expert_path data/aamas/aamas_expert-1000tra.pkl \
