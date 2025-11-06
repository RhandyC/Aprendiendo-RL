from Value_iteration.ValueIteration import ValueIteration
from MDP_definition.MDP import MDP

mdp = MDP()
solver = ValueIteration(mdp.reward_function, mdp.transition_model, gamma=0.9)
solver.train()