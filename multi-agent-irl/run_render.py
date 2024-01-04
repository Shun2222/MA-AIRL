import os

path = "data/stopenv2"
for i in range(1, 2):
    os.system("python -m irl.render_info --env simple_path_finding_jaciii --path /atlas/u/lantaoyu/exps/mack/simple_path_finding_single_jaciii/l-0.1-b-1000/seed-1/checkpoint01100 --num_trajs 1000".format(path))
