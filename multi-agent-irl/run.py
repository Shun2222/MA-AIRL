import os

for i in range(10, 50):
        os.system(f"python -m irl.mack.run_mack_aairl --env simple_tag --expert_path ./data/tag-dist-rew/checkpoint01100-10tra.pkl --seed {i}")
