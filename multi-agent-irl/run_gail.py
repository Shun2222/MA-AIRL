import os

for i in range(10, 40):
    os.system(f"python -m irl.mack.run_mack_gail --env simple_tag --expert_path ./data/tag-dist-rew/checkpoint01100-10tra.pkl --logdir /mnt/shunsuke/share --seed {i}")    
