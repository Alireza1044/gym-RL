import gym
import numpy as np

n_degree = 360
n_velocity = 40
n_effort = 15
n_iter = 50000
learning_rate = 1.0 # alpha
epsilon = 0.7
gamma = 0.9
done = False

# Import and initialize Mountain Car Environment
env = gym.make('Pendulum-v0')
env.reset()

# state = conversion of observation to state using numpy buckets
pos_buckets = np.linspace(-np.pi, np.pi, n_degree)
vel_buckets = np.linspace(-8.0, 8.0, n_velocity)
effort_buckets = np.linspace(-2.0, 2.0, n_effort)


def obs_to_state(observation):
    a = np.digitize(get_theta(observation), pos_buckets)
    b = np.digitize(observation[2], vel_buckets)
    return a, b


def action_to_state(action):
    return np.digitize(action, effort_buckets)


def get_theta(observation):
    theta = np.arctan(observation[1] / observation[0])
    if observation[0] < 0:
        if observation[1] < 0:
            theta -= np.pi
        else:
            theta += np.pi
    return theta


# main iteration

q_table = np.zeros([n_degree + 1, n_velocity + 1, n_effort + 1])
scores = []
for i in range(n_iter):
    done = False
    total_reward = 0
    observation = env.reset()
    while not done:

        # render the last 5 iterations
        if i >= n_iter - 5:
            env.render()

        state = obs_to_state(observation)
        if np.random.uniform(0, 1) < epsilon:
            # print '++++++++++++++++++++++++++++inside random++++++++++++++++++++++++++++', epsilon
            action = env.action_space.sample()
        else:
            # print '---------------------------outside random---------------------------', epsilon
            action = np.argmax(q_table[state[0].max()][state[1].max()])

        action = np.argmax(q_table[state[0].max()][state[1].max()])

        observation, reward, done, info = env.step([action])
        total_reward += reward

        new_state = obs_to_state(observation)
        action_ = action_to_state(action)
        # bellman equation to update Q-table
        q_table[state[0].max()][state[1].max()][action_] = (1 - learning_rate) * \
                                                           q_table[state[0].max()][state[1].max()][
                                                               action_] + learning_rate * (reward + gamma * max(
            q_table[new_state[0].max()][new_state[1].max()]))

    learning_rate = learning_rate * 0.99998
    epsilon = epsilon * 0.99993
    scores.append(total_reward)
    epsilon = max(0.005, epsilon * 0.9962)

    if i % 100 == 0:
        print 'iteration', i, 'total_reward', total_reward
        print 'lr', learning_rate, 'eps', epsilon, 'mean score', np.mean(scores)
        scores = []
        print

    env.close()
exit(0)
