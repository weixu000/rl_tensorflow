import time
import gym
from DQN.dqn import *
from DQN.network import *
from flappy_bird.flappybird import FlappyBirdEnv


def train_episodes(env, agent, n_episodes, render=False):
    returns = []
    for _ in range(n_episodes):
        returns.append(agent.explore(env, render))
        # print('Train with reward {:.2f}'.format(returns[-1]))
    return returns


def test_episodes(env, agent, n_episodes, goal, render=False):
    returns = []
    for _ in range(n_episodes): returns.append(agent.exploit(env, render))
    aver = np.average(returns)
    # print('Test with average reward {:.2f}'.format(aver))
    if aver >= goal:
        # print('The environment is considered solved')
        return aver, True
    else:
        return aver, False


ENV_NAME = 'CartPole-v0'
N_TEST = 100
GOAL = 195


# ENV_NAME = 'CartPole-v1'
# N_TEST = 100
# GOAL = 475


# ENV_NAME = 'MountainCar-v0'
# N_TEST = 100
# GOAL = 0


# ENV_NAME = 'Acrobot-v1'
# N_TEST = 100
# GOAL = -100

def main_gym():
    env = gym.make(ENV_NAME)
    # env._max_episode_steps = float('inf')  # 打破最大step数限制
    log_dir = '/'.join(['log', ENV_NAME, time.strftime('%m-%d-%H-%M')]) + '/'

    # agent = DDQN(env.observation_space.shape, None, 1,
    #              env.action_space.n, log_dir,
    #              FCFeatures([20, 20]),
    #              DuelingDQN([10, 5], [10, 5]))
    # agent = BootstrappedDDQN(env.observation_space.shape, None, 1,
    #                          env.action_space.n, log_dir,
    #                          FCFeatures([20, 20]),
    #                          DuelingDQN([10, 5], [10, 5]))
    agent = ModelBasedDDQN(env.observation_space.shape, None, 1,
                           env.action_space.n, log_dir,
                           FCFeatures([20, 20]),
                           DuelingDQN([10, 5], [10, 5]),
                           EnvModel())

    agent.save_hyperparameters()
    for i in range(50):
        train_return = train_episodes(env, agent, 10)
        print('Train {} with rewards {}'.format(i, train_return))
        test_return, passed = test_episodes(env, agent, N_TEST, GOAL)
        print('Test {} with rewards {}'.format(i, test_return))
        if passed:
            print('The environment is considered solved')
            break
    else:
        print('The environment has not been solved')

    agent.plot_returns()
    agent.save_session()


def main_flappybird():
    env = FlappyBirdEnv()
    log_dir = '/'.join(['log', 'FappyBird', '0']) + '/'
    agent = DDQN(env.observation_shape, [0, 255], 4,
                 env.action_n, log_dir,
                 ConvFeatures([('conv', {'weight': [8, 8, 4, 32], 'strides': [4, 4]}),
                               ('pooling', {'ksize': [2, 2], 'strides': [2, 2]}),
                               ('conv', {'weight': [4, 4, 32, 64], 'strides': [2, 2]}),
                               ('pooling', {'ksize': [2, 2], 'strides': [2, 2]}),
                               ('conv', {'weight': [3, 3, 64, 64], 'strides': [1, 1]}),
                               ('pooling', {'ksize': [2, 2], 'strides': [2, 2]})],
                              [256]),
                 OriginalQLayer([256]),
                 1E-3, 0.5, 0.001, 1E-4)
    agent.save_hyperparameters()
    train_episodes(env, agent, 5000)
    agent.save_session()


if __name__ == "__main__":
    main_gym()
