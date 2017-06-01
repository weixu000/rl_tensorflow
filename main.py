import time
import gym
from DQN.dqn import *
from DQN.network import *
from flappy_bird.flappybird import FlappyBirdEnv


def run_episodes(play, n_episodes, max_timesteps=None, render=False):
    returns = []
    for _ in range(n_episodes):
        returns.append(play(max_timesteps, render))
    return returns


# ENV_NAME = 'CartPole-v0'
# N_TEST = 100
# GOAL = 195


ENV_NAME = 'CartPole-v1'
N_TEST = 100
GOAL = 475


# ENV_NAME = 'MountainCar-v0'
# N_TEST = 100
# GOAL = 0


# ENV_NAME = 'Acrobot-v1'
# N_TEST = 100
# GOAL = -100

def main_gym():
    env = gym.make(ENV_NAME)
    # env._max_episode_steps = None  # 打破最大step数限制
    log_dir = '/'.join(['log', ENV_NAME, time.strftime('%m-%d-%H-%M')]) + '/'

    # agent = DDQN(env, env.observation_space.shape, None, 1,
    #              env.action_space.n, log_dir,
    #              FCFeatures([20, 20]),
    #              DuelingDQN([10, 5], [10, 5]))
    agent = BootstrappedDDQN(env, env.observation_space.shape, None, 1,
                             env.action_space.n, log_dir,
                             FCFeatures([10, 5]),
                             DuelingDQN([3], [3]))
    # agent = ModelBasedDDQN(env, env.observation_space.shape, None, 1,
    #                        env.action_space.n, log_dir,
    #                        FCFeatures([10, 5]),
    #                        DuelingDQN([3], [3]),
    #                        EnvModel())

    agent.save_hyperparameters()
    for i in range(50):
        train_return = run_episodes(agent.explore, 10, None, True)
        print('Train {} with rewards {}'.format(i, train_return))
        test_return = run_episodes(agent.exploit, N_TEST, None, False)
        print('Test {} with rewards {}'.format(i, np.average(test_return)))
        if np.average(test_return) >= GOAL:
            print('The environment is considered solved')
            break
    else:
        print('The environment has not been solved')

    agent.plot_returns()
    agent.save_session()


def main_flappybird():
    env = FlappyBirdEnv()
    log_dir = '/'.join(['log', 'FappyBird', '0']) + '/'
    agent = DDQN(env, env.observation_shape, [0, 255], 4,
                 env.action_n, log_dir,
                 ConvFeatures([('conv', {'weight': [8, 8, 4, 32], 'strides': [4, 4]}),
                               ('pooling', {'ksize': [2, 2], 'strides': [2, 2]}),
                               ('conv', {'weight': [4, 4, 32, 64], 'strides': [2, 2]}),
                               ('pooling', {'ksize': [2, 2], 'strides': [2, 2]}),
                               ('conv', {'weight': [3, 3, 64, 64], 'strides': [1, 1]}),
                               ('pooling', {'ksize': [2, 2], 'strides': [2, 2]})],
                              [256]),
                 DuelingDQN([10, 5], [10, 5]))
    agent.save_hyperparameters()
    run_episodes(env, agent, 5000)
    agent.save_session()


if __name__ == "__main__":
    main_gym()
