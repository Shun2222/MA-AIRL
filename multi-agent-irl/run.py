import os 

for i in range(10, 100):
    os.system(f"python -m irl.mack.run_mack_airl --env simple_tag --expert_path ./data/tag-dist-rew/checkpoint01100-10tra.pkl --seed {i}")
