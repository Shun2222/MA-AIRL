import os

path = "data/stopenv2"
for i in range(1, 2):
    os.system("python -m irl.render --env simple_path_finding_jaciii --path data/icaart5/aairl/simple_path_finding_jaciii/decentralized/s-200/l-0.1-b-500-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-31/m_50000 --num_trajs 1".format(path))
