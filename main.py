import os
import pickle
import time
import gym
import matplotlib.pyplot as plt
from DQN.dqn import DQN
from DQN.memory import *
from DQN.network import *
from DQN.target import *

# ENV_NAME = 'CartPole-v0'
# N_TEST = 100
# GOAL = 195
# MAX_EPISODES = 100


ENV_NAME = 'CartPole-v1'
N_TEST = 100
GOAL = 475
MAX_EPISODES = 200


# ENV_NAME = 'MountainCar-v0'
# N_TEST = 100
# GOAL = -110
# MAX_EPISODES = 1000


# ENV_NAME = 'Acrobot-v1'
# N_TEST = 100
# GOAL = -100
# MAX_EPISODES = 1000


def play_episode(env, action_select, step, render=False):
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


def train_episodes(env, agent, n_test, goal, n_episodes=None):
    rewards, aver_rewards = [], []
    i_episode = 0
    while True:
        # if i_episode % N_TEST == 0: play(env, agent.greedy_action, None, True)
        if len(rewards) and rewards[-1] >= goal:
            rewards.append(play_episode(env, agent.greedy_action, agent.step))
        else:
            rewards.append(play_episode(env, agent.epsilon_greedy, agent.step))

        aver_rewards.append(np.sum(rewards[-min(n_test, len(rewards)):]) / min(n_test, len(rewards)))
        print("Episode {} finished with reward {:.2f} and average {:.2f}".format(i_episode,
                                                                                 rewards[-1], aver_rewards[-1]))
        if len(rewards) > n_test and aver_rewards[-1] >= goal:  # 按openai gym 要求完成了环境
            print("Environment solved after {} episodes".format(i_episode))
            break
        i_episode += 1
        if n_episodes and i_episode >= n_episodes:  # 超出最大episode数
            r = 0
            for _ in range(n_test): r += play_episode(env, agent.greedy_action, None)
            r /= n_test
            print("Maximum {} of episodes exceeded with greedy average reward {}".format(n_episodes, r))
            break
    return rewards, aver_rewards


def save_rewards(log_dir, rewards, aver_rewards):
    with open(log_dir + 'rewards.pickle', 'wb') as f:
        pickle.dump((rewards, aver_rewards), f)


def plot_rewards(rewards, aver_rewards, n_aver):
    plt.plot(rewards, label='Return for each episode')
    plt.plot(aver_rewards, label='Average return for last {} episodes'.format(n_aver))
    plt.legend(frameon=False)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()


def main():
    env = gym.make(ENV_NAME)
    # env._max_episode_steps = float('inf')  # 打破最大step数限制
    log_dir = '/'.join(['log', ENV_NAME, time.strftime('%m-%d-%H-%M')]) + '/'
    os.makedirs(log_dir)
    agent = DQN(env.observation_space.shape, None, 1,
                env.action_space.n, log_dir,
                FCFeatures([20]),
                OriginalQLayer([20, 10]),
                RandomReplay(5000, 100, 2),
                DoubleDQN(),
                2E-3, 0.2, 1 - 1E-3)
    agent.save_hyperparameters()
    rewards, aver_rewards = train_episodes(env, agent, N_TEST, GOAL)
    save_rewards(agent.log_dir, rewards, aver_rewards)
    plot_rewards(rewards, aver_rewards, N_TEST)


if __name__ == "__main__":
    main()
