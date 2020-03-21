# state space plots for the 2d mountain car env. Let's have a look and see what happened with it.
#I also need to see rewards
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
import pickle

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
                #print(got_reward(state))
            narray = np.concatenate((narray, traj), axis=0)
    return narray

def construct_coverage_score(zs,zdots,num_compartments=30, max_zs = 5, max_zdots =0.07):
    z_compartments_borders = np.linspace(-max_zs, max_zs, num_compartments)
    zdots_compartments_borders = np.linspace(-max_zdots, max_zdots, num_compartments)
    compartment_matrix = np.zeros([100,100])
    #this is going to be a VERY slow algorithm.
    for (z,zdot) in zip(zs, zdots):
        for i in range(len(z_compartments_borders)-1):
            for j in range(len(zdots_compartments_borders)-1):
                if z >=z_compartments_borders[i] and z <= z_compartments_borders[i+1] and zdot >= zdots_compartments_borders[j] and zdot <= zdots_compartments_borders[j+1]:
                    if compartment_matrix[i,j] == 0:
                        compartment_matrix[i,j] = 1
    return np.sum(compartment_matrix) / (num_compartments**2)



def mountaincar_trajectory_plotter(trajs):
    trajs = np.reshape(trajs, (-1,4))
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(5,5))
    ax[0].set_xlim(-1.2, 0.6)
    ax[0].set_ylim(-0.07, 0.07)
    xs = trajs[:,0]
    xdots = trajs[:,1]
    ax[0].scatter(xs, xdots)
    ax[1].set_xlim(-5,5)
    ax[1].set_ylim(-0.07,0.07)
    zs = trajs[:,2]
    zdots = trajs[:,3]
    ax[1].scatter(zs, zdots)
    plt.show()

seed = 5
sname = str(sys.argv[1] if len(sys.argv)>1 else "trajectory_2d_mountaincar_experiments")
basepath = sname + "/" + sname + "/" + sname
coverage_dict = {}
dirs = os.listdir(basepath)
for d in dirs:
    coverage_dict[str(d)] = []
    for s in range(seed):
        print(s)
        path = os.path.join(basepath, d) + "/" + str(s)
        print(path)
        path +="/trajectories"
        print(path)
        trajs = collect_trajs(path)
        print(path)
        #mountaincar_trajectory_plotter(trajs)
        score = construct_coverage_score(trajs[:,2],trajs[:,3])
        print("Coverage: ",score)
        coverage_dict[str(d)].append(score)

def plot_coverage_bar_chart(coverage_dict):
    conditions = coverage_dict.keys()
    conditions = ["R","E_RE","E_R","R_E_RI","RI_E","RI","E","R_RI","RE","EE"]
    coverages = [np.mean(np.array(val)) for val in coverage_dict.values()]
    stds = [np.std(np.array(val)) for val in coverage_dict.values()]
    plt.bar(conditions,coverages)
    plt.xlabel("Condition")
    plt.ylabel("Fraction Covered")
    plt.title("Coverage of Z statespace by condition")
    plt.savefig("coverage_barchart.jpg")
    plt.show()
#coverage_dict = pickle.load(open("coverage_scores","rb"))
print("Coverage_dict: ", coverage_dict)
pickle.dump(coverage_dict, open("no_z_dependence_coverage_scores","wb"))

plot_coverage_bar_chart(coverage_dict)
