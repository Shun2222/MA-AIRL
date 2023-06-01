import os

path = r"C:\Users\shuns\research\simple_tag\decentralized\s-200\l-0.1-b-500-d-0.1-c-500-l2-0.1-iter-1-r-0.0"
path = r"C:\atlas\u\lantaoyu\exps\airl\simple_tag\decentralized\s-200\l-0.1-b-500-d-0.1-c-500-l2-0.1-iter-1-r-0.0"

for i in range(10, 30):
    os.system("python -m irl.render --env simple_tag --path {}/seed-{}/m_01000".format(path, i))
