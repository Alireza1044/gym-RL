import gym
import numpy as np

n_states = 40
n_iter = 2501
learning_rate = 1.0
epsilon = 1.0
gamma = 1.0
done = False

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()


# state = conversion of observation to state using numpy buckets
pos_buckets = np.linspace(-1.2, 0.6, n_states)
vel_buckets = np.linspace(-0.07, 0.07, n_states)


def discretize_state():
    pos_buckets = np.linspace(-1.2, 0.6, n_states)
    vel_buckets = np.linspace(-0.07, 0.07, n_states)


def obs_to_state(observation):
    a = np.digitize(observation[0], pos_buckets)
    b = np.digitize(observation[1], vel_buckets)
    return a, b


# main iteration

q_table = np.zeros([n_states, n_states, 3])

for i in range(n_iter):
    done = False
    observation = env.reset()
    total_reward = 0
    while not done:

        # render the last 5 iterations
        if i >= n_iter - 5:
            env.render()

        state = obs_to_state(observation)

        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = np.argmax(q_table[state[0]][state[1]])

        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if observation[0] >= 0.5:
            total_reward += 100

        new_state = obs_to_state(observation)

        # bellman equation to update Q-table
        q_table[state[0]][state[1]][action] = q_table[state[0]][state[1]][action] + learning_rate * \
                                              (reward + gamma * max(q_table[new_state[0]][new_state[1]]) -
                                               max(q_table[state[0]][state[1]]))

    learning_rate = learning_rate * 0.9995
    epsilon = max(0.05, epsilon * 0.9992)

    if i % 100 == 0:
        print 'iteration', i, 'total_reward', total_reward
        print 'lr', learning_rate, 'eps', epsilon
        print

    env.close()
