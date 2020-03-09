import numpy as np
import json
import matplotlib.pyplot as plt
import os

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

basepath = "RIG_lunar_lander_experiments/RIG_lunar_lander_experiments/RIG_lunar_lander_experiments"
basepath = "2d_mountaincar/mountaincar_2d_traj_experiments/mountaincar_2d_traj_experiments"
dirs = os.listdir(basepath)
rewards_list = []
idxs = []
for i in range(len(dirs)):
    p = basepath + "/" + str(dirs[i])
    #if "exploration" in p: #and "reward" in p:
    print(p)
    rewards = get_average_reward(p, 5)
    rewards_list.append(rewards)
    idxs.append(i)

for (rlist, idx) in zip(rewards_list,idxs):
    plt.plot(rlist, label=str(dirs[idx]))
plt.legend()
plt.show()

# okay, so that is an issue. Which basically blows up all our experiments... NOBODY GOT ANY REWARD!
# crap. Well going to have to rerun this anyhow, but let's look at the trajectory plots anyway to see if we can see anything interesting.




# huh... so these experiments are interesting. The mountain car especially so. I'll have to look into it a little more
# so pretty interesting results actually. It seems to make relatively little difference on cartpole, where everything finds the solution right away.
# not sure what other sparse environments to try. Could try lunar lander or bipedal walker see if it can learn any of them
#or similar, just going to be annoying but doable.
# further the mountain car always seems to decline under exploration - perhaps because the exploration is getting bored of always going there or what not
#and doesn't decline, and the reward is not big enough perhaps? so anyhow, that's the aim at least, so yeah. Try to get that sorted out and if it doesn't work then ugh
#but can be doable.  but it seems that reward info gain helps with this and provides just enough info gain to sort it out properly
# I'm totally not sure how exactly this works. Like I'm worried everything will have gotten overwritten which is really difficult
# so it's just ugh but it is difficult. So let's check what the actual experiments which were run are, because I'm worried they were all overwritten in some stupid way
#which would be really irritating. just got to fix! it and try to understand what's happening. Aim is to get that running then set off experiments with the environment
#in the planning algorithms. I'm missing some things. Got to figure out why first, which is important

# I'm also going to setup a lunar lander experiment running quickly to see if I can get any useful intuitions out of that...
#then next step will be the planning sandbox
