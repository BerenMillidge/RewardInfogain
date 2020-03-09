# state space plots for the 2d mountain car env. Let's have a look and see what happened with it.
#I also need to see rewards
import numpy as np
import json
import matplotlib.pyplot as plt
import os

def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data

def got_reward(state):
    position, velocity, zpos, zvel = state
    goal_position = 0.45
    goal_velocity = 0
    zthresh = 0.5
    return bool(position >= goal_position and velocity >= goal_velocity and zpos >=-zthresh and zpos <= zthresh )


def collect_trajs(path):
    narray = np.zeros([1,4])
    for filename in os.listdir(path):
        if filename.endswith(".npy"):
            traj = np.load(os.path.join(path, filename),allow_pickle=True)
            traj = np.array(traj)
            for t in range(len(traj)):
                state = traj[t,:]
                print(got_reward(state))
            narray = np.concatenate((narray, traj), axis=0)
    return narray



def mountaincar_trajectory_plotter(trajs):
    trajs = np.reshape(trajs, (-1,4))
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,10))
    ax[0].set_xlim(-1.2, 0.6)
    ax[0].set_ylim(-0.07, 0.07)
    xs = trajs[:,0]
    xdots = trajs[:,1]
    ax[0].scatter(xs, xdots)
    ax[1].set_xlim(-10,10)
    ax[1].set_ylim(-0.5, 0.5)
    zs = trajs[:,2]
    zdots = trajs[:,3]
    ax[1].scatter(zs, zdots)
    plt.show()

seed = 5
basepath = "2d_mountaincar/mountaincar_2d_traj_experiments/mountaincar_2d_traj_experiments"
# so they all look kind of the same but this is unsurprising as they never actually discovered the reward.
#what is the condition on it and how fullfillable is it?
# the key issue is that the environment is simply too difficult in this case. Which is really frustrating.
# let's try to make it easier. Smaller zspace available and no effect of z dimension AT ALL on action.

dirs = os.listdir(basepath)
for d in dirs:
    for s in range(seed):
        print(s)
        path = os.path.join(basepath, d) + "/" + str(s)
        print(path)
        path +="/trajectories"
        trajs = collect_trajs(path)
        print(path)
        mountaincar_trajectory_plotter(trajs)
