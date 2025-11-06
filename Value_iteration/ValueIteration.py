import numpy as np
import matplotlib.pyplot as plt

class ValueIteration:
    def __init__(self, reward_function, transition_model, gamma):
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]
        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
        self.gamma = gamma
        self.values = np.zeros(self.num_states)
        self.policy = None
        #SON COSITAS
        self.action_matrix = np.zeros(5)
        self.policy_path = np.zeros(5)

    def one_iteration(self):
        delta = 0
        for s in range(self.num_states):
            temp = self.values[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)

            self.values[s] = max(v_list)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def get_policy(self):
        pi = np.ones(self.num_states) * -1
        for s in range(self.num_states):
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)

            max_index = []
            max_val = np.max(v_list)
            for a in range(self.num_actions):
                if v_list[a] == max_val:
                    max_index.append(a)
            pi[s] = np.random.choice(max_index)
        return pi.astype(int)
    
    def simulate_policy(self, initial_state: int, max_steps: int = 5):
        """Simulates following the policy from an initial state."""
        current_state = initial_state
        for t in range(max_steps):
            action = self.policy[current_state]
            self.action_matrix[t] = action
            next_state = self.sample_next_state(self.transition_model[current_state, action])
            self.policy_path[t] = next_state
            current_state = next_state

    def sample_next_state(self, probabilities: np.ndarray) -> int:
        """Samples the next state from a probability distribution."""
        print("Probabilidades antes de normalizar:", probabilities)
        return np.random.choice(len(probabilities), p=probabilities)

    def print_policy(self):
        """Prints the action sequence followed in the simulation."""
        print("Action sequence:", self.action_matrix.tolist())

    def train(self, tol=1e-3):
        epoch = 0
        delta = self.one_iteration()
        delta_history = [delta]
        while delta > tol:
            epoch += 1
            delta = self.one_iteration()
            delta_history.append(delta)
            if epoch <= 10: 
                print("Values:", self.values)
            if delta < tol:
                break
        self.policy = self.get_policy()
        print(f'# iterations of policy improvement: {len(delta_history)}')
        print(f'delta = {delta_history}')
        print(self.policy)
        self.simulate_policy(0,5)
        self.print_policy()

        fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)
        ax.plot(np.arange(len(delta_history)) + 1, delta_history, marker='o', markersize=4,
                alpha=0.7, color='#2ca02c', label=r'$\gamma= $' + f'{self.gamma}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Delta')
        ax.legend()
        plt.tight_layout()
        plt.show()