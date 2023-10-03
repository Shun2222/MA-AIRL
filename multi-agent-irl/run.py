import os 

for i in range(20, 40):
    os.system(f"python -m irl.mack.run_mack_airl --env simple_path_finding_jaciii --expert_path data/path2-disc/checkpoint01100-1000tra.pkl --seed {i} --logdir ./data/airl10000000")
