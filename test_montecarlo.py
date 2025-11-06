
import gymnasium as gym
import numpy as np

env = gym.make('Blackjack-v1')

print(env.observation_space)
print(env.action_space)

for i_episode in range(3):
    state = env.reset()
    while True:
        print(state)
        action = env.action_space.sample()
        c, reward, terminated, truncated, info  = env.step(action)
        done = terminated or truncated # El episodio termina si es 'terminado' O 'truncado'
        if done:
            print('End game! Reward: ', reward)
            print('You won :)\n') if reward > 0 else print('You lost :(\n')
            break