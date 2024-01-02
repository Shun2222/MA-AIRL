import os 

for i in range(30, 31):
    os.system(rf"python3 -m irl.mack.run_mack_estrew \
                 --env simple_path_finding_jaciii \
                 --expert_path data/icarart-tiny2/checkpoint01100-1000tra.pkl\
                 --seed {i} \
                 --logdir ./data/icarrt5/estrew")
                 #--expert_path data/aamas/aamas_expert-1000tra.pkl \
