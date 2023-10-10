import os 

for i in range(30, 50):
    os.system(rf"python3 -m irl.mack.run_mack_gail \
                 --env simple_path_finding_jaciii \
                 --expert_path data/aamas/aamas_expert-1000tra.pkl \
                 --seed {i} \
                 --logdir ./data/aamas")
