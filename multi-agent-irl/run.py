import os 

<<<<<<< HEAD
for i in range(10, 11):
    os.system(f"python -m irl.mack.run_mack_aairl --env simple_path_finding_jaciii --expert_path ./data/path2/checkpoint01100-1000tra.pkl --seed {i} --logdir ./data/path2")
=======
for i in range(10, 40):
    os.system(f"python -m irl.mack.run_mack_gail --env simple_path_finding_jaciii --expert_path ./data/path2/checkpoint01100-1000tra.pkl --seed {i} --logdir ./data/path2")
>>>>>>> 76a6a4cd48163aaacbed81f1e3969150eb44fcd0
