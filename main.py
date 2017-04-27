import pickle
import time
import gym
import matplotlib.pyplot as plt
from DQN.dqn import DQN
from DQN.memory import *
from DQN.network import *
from DQN.target import *
from flappy_bird.flappybird import FlappyBirdEnv

ENV_NAME = 'CartPole-v0'
N_TEST = 100
GOAL = 195


# ENV_NAME = 'CartPole-v1'
# N_TEST = 100
# GOAL = 475


# ENV_NAME = 'MountainCar-v0'
# N_TEST = 100
# GOAL = -110


# ENV_NAME = 'Acrobot-v1'
# N_TEST = 100
# GOAL = -100


def play_episode(env, action_select, step=None, render=False):
    ret = 0
    observation = env.reset()
    while True:
        if render: env.render()
        action = action_select(observation)
        nxt_observation, reward, done, _ = env.step(action)
        if step: step(observation, action, reward, done)
        observation = nxt_observation
        ret += reward
        if done: return ret


def train_episodes(env, agent, n_episodes):
    returns = []
    for _ in range(n_episodes):
        returns.append(play_episode(env, agent.epsilon_greedy, agent.step))
        # print('Train with reward {:.2f}'.format(returns[-1]))
    return returns


def test_episodes(env, agent, n_episodes, goal):
    returns = []
    for _ in range(n_episodes): returns.append(play_episode(env, agent.greedy_action))
    aver = np.average(returns)
    print('Test with average reward {:.2f}'.format(aver))
    if aver >= goal: print('The environment is considered solved')
    return returns


def plot_returns(returns, title, x_label, y_label, path):
    plt.plot(returns)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path)
    plt.clf()


def main_gym():
    env = gym.make(ENV_NAME)
    # env._max_episode_steps = float('inf')  # 打破最大step数限制
    log_dir = '/'.join(['log', ENV_NAME, time.strftime('%m-%d-%H-%M')]) + '/'
    agent = DQN(env.observation_space.shape, None, 2,
                env.action_space.n, log_dir,
                FCFeatures([20, 20]),
                DuelingDQN([10, 5], [10, 5]),
                ReturnPrioritizedReplay(5000, 100, 2, 5, 0.5, 1E-3, 100, 0.99),
                DoubleDQN(0.99),
                1E-2, 0.2, 0.01, 1E-3)
    agent.save_hyperparameters()
    train_returns, test_returns = [], []
    for _ in range(50):
        train_returns += train_episodes(env, agent, 10)
        test_returns += test_episodes(env, agent, N_TEST, GOAL)

    agent.save_sessions()
    with open(log_dir + 'rewards.pickle', 'wb') as f:
        pickle.dump((train_returns, test_returns), f)
    plot_returns(train_returns, 'Train Returns', 'Episode', 'Return', log_dir + 'train.png')
    plot_returns(test_returns, 'Train Returns', 'Episode', 'Return', log_dir + 'test.png')


def main_flappybird():
    env = FlappyBirdEnv()
    log_dir = '/'.join(['log', 'FappyBird', '0']) + '/'
    agent = DQN(env.observation_shape, [0, 255], 2,
                env.action_n, log_dir,
                ConvFeatures([('conv', {'weight': [4, 5, 2, 20], 'strides': [1, 1]}),
                              ('conv', {'weight': [10, 5, 20, 10], 'strides': [10, 5]}),
                              ('conv', {'weight': [2, 2, 10, 5], 'strides': [2, 2]})],
                             [50]),
                DuelingDQN([20, 20], [20, 20]),
                RandomReplay(5000, 50, 2),
                DoubleDQN(0.99),
                1E-2, 0.1, 0.001, 1E-4)
    agent.save_hyperparameters()
    train_episodes(env, agent, 5000)


if __name__ == "__main__":
    main_gym()
