"""DQN agents trained on Breakthrough by independent Q-learning."""
from utils.environment.CIgame import GraphGame
from utils.environment.envhelper import gen_new_graphs
from utils.reinforcement_learning.rl_environment import Environment
from utils.validation import get_Validation, area_under_curve
from utils.reinforcement_learning.dqn import DQN
from utils.reinforcement_learning.CIGraphNN import CIGraphNN
from utils.params import Params
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import torch
import copy
import numpy as np
from datetime import datetime
from open_spiel.python.algorithms import random_agent


# Training parameters
params = Params("./utils/CI_ba_params.json")
# WandB – Initialize a new run
now = datetime.now()

def Generate_Batch_Graph(size,seed = None):
    Batch_Graph = [gen_new_graphs(graph_type=['barabasi_albert'],seed=seed+i) for i in range(size)]
    return np.array(Batch_Graph,dtype=object)
evaluation, eval_x = get_Validation(params.validation_test_size)#get_Validation(4,file_path)

def plot(CV_AUC):
    CV_AUC = np.array(CV_AUC)
    save_every= 500            
    num_train_episodes = int(5e5) 
    x = np.arange(500,num_train_episodes+save_every,save_every)
    min_value = np.argmin(CV_AUC)
    min_vc = x[min_value]
    plt.plot(x,CV_AUC,color='red')
    plt.plot(min_vc, CV_AUC[min_value], marker="o", markersize=5, markeredgecolor='orange', markerfacecolor="None",label="Min: "+str(min_vc))
    plt.title("AUC of LCC vs Nodes Plot BA")
    plt.xlabel("iteration")
    plt.ylabel("AUC")
    plt.legend()
    plt.savefig("CV.png")
    #plt.show()


def main(agents=None):
    game = "graph_attack_defend"
    env = Environment(game)
    num_actions = env.action_spec()["num_actions"]  
    size_CV = params.validation_test_size
    CV_AUC = []
    if agents == None:
        agents = [
            DQN(
                player_id=0,
                state_representation_size=params.centrality_features,
                global_feature_size = params.global_features,
                num_actions=num_actions,
                hidden_layers_sizes=params.hidden_layers,
                replay_buffer_capacity=int(params.replay_buffer_capacity),
                learning_rate=params.learning_rate,
                update_target_network_every=  params.update_target_network_every,
                learn_every=params.learn_every,
                discount_factor=params.discount_factor,
                min_buffer_size_to_learn=params.min_buffer_size_to_learn,
                power = params.epsilon_power,
                nsteps=params.nstep,
                epsilon_start=params.epsilon_start,
                epsilon_end=params.epsilon_end,
                epsilon_decay_duration=params.epsilon_decay_duration,
                batch_size=params.batch_size,
                GraphNN = CIGraphNN)
        ]
    #wandb.watch(agents[0]._q_network, log="all")
    agents.append(random_agent.RandomAgent(player_id=1, num_actions=num_actions))
    graph_batch_size = params.graph_batch_size
    for ep in tqdm(range(int(params.num_train_episodes))):
        if (ep) % params.graph_suffle == 0:
            Batch_Graph = Generate_Batch_Graph(graph_batch_size,seed=ep)
        time_step = env.reset(Batch_Graph[int(ep%graph_batch_size)].copy())
        while not time_step.last():
            agents_output = [agent.step(time_step) for agent in agents]
            actions = [agent_output.action for agent_output in agents_output]
            action_list = [actions[0], actions[0]]
            time_step = env.step(action_list)
        for agent in agents:
            agent.step(time_step)
        if (ep + 1) % params.eval_every == 0:
            AUC = []
            for i in range(size_CV):
                eval_step = env.reset(evaluation[i].copy())
                while not eval_step.last():
                    eval_output = [agent.step(eval_step, is_evaluation=True) for agent in agents]
                    actions = [agent_output.action for agent_output in eval_output]
                    action_list = [actions[0], actions[0]]
                    eval_step = env.step(action_list)
                lcc = env.get_state.lcc
                AUC.append(area_under_curve(eval_x[i][:len(lcc)],lcc))
            meanAUC = np.mean(AUC)
            CV_AUC.append(meanAUC)
        if (ep + 1) % params.save_every == 0:
            checkpoint = {'_q_network': agents[0]._q_network.state_dict(),'target_q_network': agents[0]._target_q_network.state_dict(),'_optimizer' :agents[0]._optimizer.state_dict()}
            title = params.checkpoint_dir+"_"+str(ep+1)
            torch.save(checkpoint, title)
    plot(CV_AUC)
        


if __name__ == "__main__":
     # if you call this script from the command line (the shell) it will
     # run the 'main' function
     main(agents=None)