import os 

for i in range(10, 11):
    os.system(f"python -m irl.mack.run_mack_gail --env simple_path_finding_jaciii --expert_path ./data/my-path/data-10tra.pkl --seed {i} --logdir ./data --discrete --grid_size 15 15")
