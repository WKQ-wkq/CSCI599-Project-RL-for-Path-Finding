import rooms
import agent as a
import matplotlib.pyplot as plot
import sys
import os
import csv
import pickle

## Hypeparameters setting of our solution
bias = 0
epsilon_decay = 0.0001
N = 200

gamma = 0.997

def episode(env, agent, nr_episode=0):

    ## Dynamic setting of the bias value (In the intermediate 2)
    if nr_episode == N:
        print("Reduce bias.")
        agent.bias = agent.bias/5

    state = env.reset()
    discounted_return = 0
    discount_factor = gamma
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return
    
params = {}
rooms_instance = sys.argv[1]

## create the floder to save.
save_folder = f"results/bias_{bias}_epsilon_decay_{epsilon_decay}_N_{N}"
if sys.argv[2] == "qlearning":
    save_folder += '_qlearning'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{save_folder}/{rooms_instance}.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = gamma
params["bias"] = bias
params["epsilon_decay"] = epsilon_decay
params["alpha"] = 0.1
params["env"] = env

if len(sys.argv) == 4:
    ## mode: train
    training_episodes = int(sys.argv[3])
    if sys.argv[2] == "qlearning":
        agent = a.QLearner(params)
        print("Use QLearning w/ NBTD.")
    else:
        agent = a.SARSALearner(params)
        print("Use SARSA w/ NBTD.")

    returns = [episode(env, agent, i) for i in range(training_episodes)]
    agent.save_model(f"{save_folder}/{rooms_instance}_episodes_{training_episodes}.pkl")

    x = range(training_episodes)
    y = returns

    ## Writing to the csv file
    save_data_file = f"{save_folder}/{rooms_instance}_training_data.csv"
    with open(save_data_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Episode', 'Return'])

        for i in range(len(y)):
            writer.writerow([x[i], y[i]])

    plot.plot(x,y)
    plot.title("Progress")
    plot.xlabel("Episode")
    plot.ylabel("Discounted Return")
    plot.savefig(f"{save_folder}/{rooms_instance}.png")
    plot.show(block=False)

    env.save_video()

elif len(sys.argv)==5:
    ## mode: test
    test_episodes = int(sys.argv[3])
    model_file = 'results/'+sys.argv[4]
    with open(model_file, 'rb') as f:
        agent = pickle.load(f)

    returns = [episode(env, agent, i) for i in range(test_episodes)]
    x = range(test_episodes)
    y = returns

    ## Writing to the csv file
    save_data_file = f"results/test_{sys.argv[4].replace('/','_').replace('.pkl','')}.csv"
    with open(save_data_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Episode', 'Return'])

        for i in range(len(y)):
            writer.writerow([x[i], y[i]])

    plot.plot(x,y)
    plot.title("Progress")
    plot.xlabel("Episode")
    plot.ylabel("Discounted Return")
    plot.savefig(f"results/test_{sys.argv[4].replace('/','_').replace('.pkl','')}.png")
    plot.show(block=False)

else:
    print('The number of parameters is incorrect.')