import os
import sys
generated_name = str(sys.argv[1])
results_save_base = str(sys.argv[2])
save_path = str(sys.argv[3])
base_call = (f"python train.py --env_name 'BipedalWalker-v2' --env_std 0.0 --expl_scale 0.1 --save_every 10 --plan_horizon 30 --ensemble_size 15 --collect_trajectories True --action_repeat 4 --action_noise 0.0 --epsilon 0.1")
output_file = open(generated_name, "w")

seeds = 5
for s in range(seeds):
    log_path = results_save_base +"/reward_only"
    spath = save_path +"/reward_only/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward True"
    f" --use_exploration False"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/exploration_only"
    spath = save_path +"/exploration_only/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward False"
    f" --use_exploration True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/exploration_reward"
    spath = save_path +"/exploration_reward/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward True"
    f" --use_exploration True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/epsilon_greedy"
    spath = save_path +"/epsilon_greedy/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward True"
    f" --use_exploration False"
    f" --use_epsilon_greedy True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)


print("Experiment file genereated")
output_file.close()
