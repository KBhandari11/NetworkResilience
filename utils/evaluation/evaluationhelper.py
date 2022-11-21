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
        action_lists.append(action_list[0])
        time_step = env.step([action_list[0],action_list[0]])
        i+=1
        episode_rewards.append(env.get_state._rewards[0])
    lcc = env.get_state.lcc
    return episode_rewards, lcc, action_lists

# Given an environmnet with all action in  a list 
def EvaluateACTION(action_list,GRAPH):
    """Evaluates the env for given action_list"""
    lcc = [len(get_lcc(GRAPH))]
    act = []
    for action in action_list:
        ebunch = GRAPH.incident(GRAPH.vs.find(name=str(action)))
        GRAPH.delete_edges(ebunch)
        cond, l = network_dismantle(GRAPH, lcc[0])
        lcc.append(l)
        act.append(action)
        if cond:
            break
    return None, lcc[0:GRAPH.vcount()], act