import numpy as np

class HoldAgent:
    """ 
    A simple agent that always selects the "hold" action (0), 
    regardless of the current state.
    """

    def __init__(self):
        """ Initializes the agent. No learning or state tracking required. """
        self.state = 0  # Placeholder for consistency, not used in decision-making.

    def act(self, state):
        """ 
        Returns the fixed action "hold" (0) for any given state. 
        
        Parameters:
        - state: The current environment state (unused in this agent).
        
        Returns:
        - 0 (hold)
        """
        return 0

    def learn(self, state, action, reward, next_state):
        """ 
        No learning occurs, as this is a fixed-policy agent. 
        
        Parameters:
        - state: The current state (not used).
        - action: The action taken (always 0).
        - reward: The reward received (ignored).
        - next_state: The next state (not used).
        """
        pass


import numpy as np

class QLearningAgent:
    """
    A tabular Q-learning agent that learns an optimal policy using an epsilon-greedy strategy.
    """

    def __init__(self, alpha, gamma, epsilon, num_states, num_actions):
        """
        Initializes the Q-learning agent with a Q-table and hyperparameters.

        Parameters:
        - alpha (float): Learning rate (how much new information overrides old knowledge).
        - gamma (float): Discount factor (importance of future rewards).
        - epsilon (float): Exploration rate (probability of taking a random action).
        - num_states (int): The number of discrete states in the environment.
        - num_actions (int): The number of possible actions the agent can take.
        """
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.num_states = num_states
        self.num_actions = num_actions

        # Initialize Q-table with zeros, shape: (num_states, num_actions)
        self.q = np.zeros((num_states, num_actions))

    def act(self, state):
        """
        Selects an action using an epsilon-greedy policy.

        With probability epsilon, a random action is chosen (exploration).
        Otherwise, the action with the highest Q-value for the given state is selected (exploitation).

        Parameters:
        - state (int): The current state of the environment.

        Returns:
        - action (int): The action chosen by the agent.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore (random action)
        else:
            return np.argmax(self.q[state])  # Exploit (best action based on Q-values)

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Q-learning update rule.

        The update is based on the Temporal Difference (TD) learning formula:
        Q(s, a) ← Q(s, a) + α [r + γ max(Q(s', a')) - Q(s, a)]

        Parameters:
        - state (int): The current state before taking action.
        - action (int): The action taken.
        - reward (float): The reward received after taking the action.
        - next_state (int): The next state after the action was executed.
        """
        # Get the best Q-value for the next state
        best_next_action = np.argmax(self.q[next_state])

        # Compute the TD target and TD error
        td_target = reward + self.gamma * self.q[next_state, best_next_action]
        td_error = td_target - self.q[state, action]

        # Update the Q-value using the TD error
        self.q[state, action] += self.alpha * td_error




