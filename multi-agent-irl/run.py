import os 

for i in range(10, 40):
    os.system(f"python -m irl.mack.run_mack_aairl --env simple_path_finding_jaciii --expert_path ./data/path2/checkpoint01100-1000tra.pkl --seed {i} --logdir ./data/path2")
