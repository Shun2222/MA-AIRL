import os 

for i in range(10, 40):
    os.system(f"python -m irl.mack.run_mack_aairl --env simple_path_finding_jaciii --expert_path data/path2-disc/checkpoint01100-1000tra.pkl --seed {i} --discrete --grid_size 5 5 --logdir ./data/path2-disc")
