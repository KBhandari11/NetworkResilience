from  utils.environment.envhelper import*
# Given an environment and an trained agent we implement the agent
def EvaluateModel(env, trained_agents,GRAPH):
    """Evaluates `trained_agents` against a new graph."""
    cur_agents = [agent for agent in trained_agents]
    time_step = env.reset(GRAPH)
    episode_rewards = []
    action_lists = []
    i = 0
    while not time_step.last():
        agents_output = [
            agent.step(time_step, is_evaluation=True) for agent in cur_agents
        ]
        action_list = [agent_output.action for agent_output in agents_output]
        action_lists.append(int(env.get_state.Graph.vs[action_list[0]]["name"])) # the action corresponds to node id in IGraph and name refers to the name of the node in the graph. 
        #action_lists.append(action_list[0])
        time_step = env.step([action_list[0],action_list[0]])
        i+=1
        episode_rewards.append(env.get_state._rewards[0])
    lcc = env.get_state.lcc
    return episode_rewards, lcc, action_lists

def eval_network_dismantle(graph, init_lcc):
    largest_cc = len(get_lcc(graph))
    cond = True if (largest_cc/init_lcc) <= 0.01 else False
    return cond, largest_cc

def EvaluateACTION(action_list,GRAPH):
    """Evaluates the env for given action_list"""
    lcc = [len(get_lcc(GRAPH))]
    act = []
    for action in action_list:
        ebunch = GRAPH.incident(GRAPH.vs.find(name=str(action)))
        GRAPH.delete_edges(ebunch)
        cond, l = eval_network_dismantle(GRAPH, lcc[0])
        lcc.append(l)
        act.append(action)
        if cond:
            break
    return None, lcc[0:GRAPH.vcount()], act