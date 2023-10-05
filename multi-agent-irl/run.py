import os 

for i in range(20, 40):
    os.system(rf"python3 -m irl.mack.run_mack_airl \
                 --env simple_path_finding_jaciii \
                 --expert_path data/aamas/aamas_expert-1000tra.pkl \
                 --seed {i} \
                 --logdir ./data/aamas")
