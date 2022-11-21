# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
""" Graph Attack and Defense implemented in Python.
This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.
"""



import numpy as np
from torch_geometric import utils
from  utils.environment.envhelper import *
import pyspiel





class GraphGame(pyspiel.Game):
  """A Python version of the Graph game."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self,Graph):
    """Returns a state corresponding to the start of a game."""
    return GraphState(self,Graph)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    '''
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoardObserver(params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)
    '''
    return BoardObserver(params)


class GraphState(pyspiel.State):
  """A python version of the Tic-Tac-Toe state."""
  def __init__(self, game,Graph):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._is_terminal = False
    self.Graph = Graph
    self.num_nodes = self.Graph.vcount()
    #self.info_state = utils.from_networkx(self.Graph.to_networkx())
    self.info_state = from_igraph(self.Graph)
    self.global_feature = None
    self._rewards = np.zeros(_NUM_PLAYERS)
    self._returns = np.zeros(_NUM_PLAYERS)
    self.lcc = [len(get_lcc(self.Graph))]
    self.r = []
    self.alpha = 1 - self.Graph.subgraph(range(self.Graph.vcount()-1)).density()
    self.beta = [molloy_reed(self.Graph)]
    self.empty_index = torch.Tensor().reshape((2, 0))

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    #return pyspiel.PlayerId.TERMINAL if self._is_terminal else pyspiel.PlayerId.SIMULTANEOUS
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else pyspiel.PlayerId.SIMULTANEOUS
  
  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    all_nodes = np.array(self.Graph.vs["active"])
    active_nodes = np.where(all_nodes == 1)[0]
    if player == 0 :
        action_sequence = active_nodes#np.squeeze(np.append(active_nodes,np.where(all_nodes == 3)))
    elif player == 1:
        action_sequence = active_nodes 
    else:
        action_sequence =  active_nodes
    #return np.delete(action_sequence,-1) #for supernode
    return action_sequence

  def _apply_actions(self, actions):
    """Applies the specified action to the state."""
    #attack_node = self.board.nodes[actions[0]]["index"]
    attack_node = actions[0]
    #defend_node = self.board.nodes[actions[1]]["index"]
    defend_node = actions[1]
    if (actions[0] == actions[1]):
      self.Graph.vs[attack_node]["active"] = 0
    else: 
      self.Graph.vs[attack_node]["active"] = 0
      self.Graph.vs[defend_node]["active"] = 2
    #self.Graph.delete_vertices(attack_node)
    ebunch = self.Graph.incident(attack_node)
    self.Graph.delete_edges(ebunch)
    cond, l = network_dismantle(self.Graph, self.lcc[0])
    #self.info_state = utils.from_networkx(self.Graph.to_networkx())
    self.info_state = from_igraph(self.Graph)
    self.global_feature = None
    beta = molloy_reed(self.Graph)
    if beta == 0:
      beta = self.beta[-1]
      cond = True
    reward_1 = (self.lcc[-1] - l)/self.lcc[-1]
    reward_2 = (self.beta[-1] - beta)/self.beta[-1]
    self._rewards[0] = ((self.num_nodes-len(self.lcc))/self.num_nodes)* (self.alpha * reward_1 +(1-self.alpha)*reward_2)
    self._rewards[1] = -self._rewards[0]
    self._returns += self._rewards
    self.beta.append(beta)  
    self.lcc.append(l)
    self.r.append(self._rewards[0])
    self._is_terminal = cond
    
  def _action_to_string(self, player, action):
    """Action -> string."""
    return "{}({})".format(0 if player == 0 else 1, action)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return self._returns
  def rewards(self):
    """Total reward for each player over the course of the game so far."""
    return self._rewards

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return board_to_string(self.Graph)

  def new_initial_state(self,Graph):
      self.Graph = Graph
      #self.info_state = utils.from_networkx(self.Graph.to_networkx())
      self.info_state = from_igraph(self.Graph)
      self.global_feature = None
      self.lcc = [len(get_lcc(self.Graph))]
      self.r = []
      #self.alpha = (1-nx.density(self.Graph.subgraph(np.arange(len(self.Graph)-1)))) # For Supernode
      self.alpha = 1-self.Graph.density()
      self.beta = [molloy_reed(self.Graph)]


class BoardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""
  def __init__(self,params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)
    self.tensor = np.array([])
    self.dict = {"observation":self.tensor}
    #self.dict = {"observation":self.tensor}


  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    obs = self.dict["observation"]
    obs = np.zeros((state.num_nodes))
    all_nodes = np.array([(i, state.Graph.vs[i]["active"]) for i in state.Graph.vs.indices])
    self.tensor = all_nodes
    return self.tensor

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    return board_to_string(state.Graph)



# Register the game with the OpenSpiel library]
_NUM_PLAYERS = 2
_MAX_CELLS = 50
GRAPH = gen_new_graphs(['erdos_renyi', 'powerlaw','small-world', 'barabasi_albert'],seed=0)
#nx.write_adjlist(GRAPH, "/content/figure/Graph")

_GAME_TYPE = pyspiel.GameType(
    short_name="graph_attack_defend",
    long_name="Python Attack Defend",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=False,
    provides_observation_tensor=False,
    provides_factored_observation_string=True)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_MAX_CELLS,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_MAX_CELLS)
pyspiel.register_game(_GAME_TYPE, GraphGame)