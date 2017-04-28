import pickle
import time
import gym
import matplotlib.pyplot as plt
from DQN.dqn import *
from DQN.memory import *
from DQN.network import *
from DQN.target import *
from flappy_bird.flappybird import FlappyBirdEnv


def train_episodes(env, agent, n_episodes, render=False):
    returns = []
    for _ in range(n_episodes):
        returns.append(agent.explore(env, render))
        print('Train with reward {:.2f}'.format(returns[-1]))
    return returns


def test_episodes(env, agent, n_episodes, goal, render=False):
    returns = []
    for _ in range(n_episodes): returns.append(agent.exploit(env, render))
    aver = np.average(returns)
    print('Test with average reward {:.2f}'.format(aver))
    if aver >= goal:
        print('The environment is considered solved')
        return returns, True
    else:
        return returns, False


def plot_returns(returns, title, x_label, y_label, path):
    plt.plot(returns)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path)
    plt.clf()


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

def main_gym():
    env = gym.make(ENV_NAME)
    # env._max_episode_steps = float('inf')  # 打破最大step数限制
    log_dir = '/'.join(['log', ENV_NAME, time.strftime('%m-%d-%H-%M')]) + '/'
    # agent = DQN(env.observation_space.shape, None, 2,
    #             env.action_space.n, log_dir,
    #             FCFeatures([20, 20]),
    #             DuelingDQN([10, 5], [10, 5]),
    #             RandomReplay(5000, 100, 2),
    #             DoubleDQN(),
    #             1E-3, 0.5, 0.01, 1E-3)
    agent = BootstrappedDQN(env.observation_space.shape, None, 2,
                            env.action_space.n, log_dir,
                            FCFeatures([20, 20]),
                            DuelingDQN([10, 5], [10, 5]))
    agent.save_hyperparameters()
    train_returns, test_returns = [], []
    for _ in range(50):
        train_returns += train_episodes(env, agent, 10)
        tr = test_episodes(env, agent, N_TEST, GOAL)
        test_returns += tr[0]
        if tr[1]: break

    agent.save_sessions()
    with open(log_dir + 'rewards.pickle', 'wb') as f:
        pickle.dump((train_returns, test_returns), f)
    plot_returns(train_returns, 'Train Returns', 'Episode', 'Return', log_dir + 'train.png')
    plot_returns(test_returns, 'Test Returns', 'Episode', 'Return', log_dir + 'test.png')


def main_flappybird():
    env = FlappyBirdEnv()
    log_dir = '/'.join(['log', 'FappyBird', '0']) + '/'
    agent = DQN(env.observation_shape, [0, 255], 4,
                env.action_n, log_dir,
                ConvFeatures([('conv', {'weight': [8, 8, 4, 32], 'strides': [4, 4]}),
                              ('pooling', {'ksize': [2, 2], 'strides': [2, 2]}),
                              ('conv', {'weight': [4, 4, 32, 64], 'strides': [2, 2]}),
                              ('pooling', {'ksize': [2, 2], 'strides': [2, 2]}),
                              ('conv', {'weight': [3, 3, 64, 64], 'strides': [1, 1]}),
                              ('pooling', {'ksize': [2, 2], 'strides': [2, 2]})],
                             [256]),
                OriginalQLayer([256]),
                RandomReplay(5000, 100, 2),
                DoubleDQN(1),
                1E-3, 0.5, 0.001, 1E-4)
    agent.save_hyperparameters()
    train_episodes(env, agent, 5000)
    agent.save_sessions()


if __name__ == "__main__":
    main_gym()
