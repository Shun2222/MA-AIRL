import os 

for i in range(20, 40):
    os.system(rf"python3 -m irl.mack.run_mack_airl \
                 --env simple_path_finding_jaciii \
                 --expert_path data/aamas/exps/mack/simple_path_finding_single_jaciii/l-0.1-b-1000/seed-1/checkpoint01100-100tra.pkl \
                 --seed {i} \
                 --logdir ./data/aamas")
