import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys

def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def get_average_reward(path,n_seeds):
    first_path = path + "/0/metrics.json"
    reward = np.array(load_json(first_path)["test_rewards"],dtype=np.float64)[0:120]
    for i in range(1,n_seeds):
        i_path = path + "/" + str(i) + "/metrics.json"
        rewards = np.array(load_json(i_path)["test_rewards"])[0:120]
        #print(type(rewards), type(reward))
        #print(rewards)
        reward += rewards
    return reward / n_seeds

if len(sys.argv)>=2:
    p = str(sys.argv[1])
    basepath = p + "/"+p + "/" + p
else:
    basepath = "RIG_reacher_experiments/RIG_reacher_experiments/RIG_reacher_experiments"
if len(sys.argv) >=3:
    title = str(sys.argv[2])
else:
    title = None

dirs = os.listdir(basepath)
rewards_list = []
idxs = []
for i in range(len(dirs)):
    p = basepath + "/" + str(dirs[i])
    print("P: ",p)
    #if "exploration" in p: #and "reward" in p:
    print(p)
    rewards = get_average_reward(p, 5)
    rewards_list.append(rewards)
    idxs.append(i)

fig = plt.figure(figsize = (10,8))
for (rlist, idx) in zip(rewards_list,idxs):
    plt.plot(rlist, label=str(dirs[idx]))
plt.xlabel("Episode")
plt.ylabel("Reward Obtained")
if title:
    plt.title(title)
plt.legend()
fig.savefig(title + "_rewards.jpg")
plt.show()
