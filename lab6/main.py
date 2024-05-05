import random
from time import sleep

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# parameters
NUM_EPISODES = 15000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 1
EXPLORATION_DECAY_RATE = 0.0001
MAX_EPISODE_LENGTH = 200

SLIPPERY = True

def update_q_table(q_table, current_state, next_state, action, reward):
    q_table[current_state, action] = q_table[current_state, action] + LEARNING_RATE * ( reward + DISCOUNT_FACTOR * np.max(q_table[next_state, :]) - q_table[current_state, action] )

def main():
    # TRAINING
    # initialize environment for training
    training_env = gym.make(
        'FrozenLake-v1',
        map_name="8x8",
        is_slippery=SLIPPERY,
        max_episode_steps=MAX_EPISODE_LENGTH,
    )

    # constants
    ACTION_SPACE_SIZE = training_env.action_space.n
    STATE_SPACE_SIZE = training_env.observation_space.n

    # initialize the q table
    q = np.zeros((STATE_SPACE_SIZE, ACTION_SPACE_SIZE))

    episode_reward = []
    steps_taken = []
    print('Training started')
    for i in range(NUM_EPISODES):
        steps_taken.append(0)
        if i % 100 == 0:
            print(f'Episode {i}')
            episode_reward.append(0)
        state = training_env.reset()[0]
        terminated = False
        truncated = False
    
        while(not terminated and not truncated):
            global EXPLORATION_RATE
            # decide what to do and make step
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold < EXPLORATION_RATE:
                # explore - make a random move
                action = training_env.action_space.sample()
            else:
                # exploit - go through the best path found
                action = np.argmax(q[state,:])

            next_state, reward, terminated, truncated, _ = training_env.step(action)
            steps_taken[i] += 1

            episode_reward[int(i / 100)] += reward

            update_q_table(
                q,
                state,
                next_state,
                action,
                reward
            )

            state = next_state

        EXPLORATION_RATE = max(EXPLORATION_RATE - EXPLORATION_DECAY_RATE, 0)

    training_env.close()

    x1 = [ x for x in range(100, NUM_EPISODES+1, 100) ]
    y1 = [ 0 for _ in range(len(episode_reward))]
    for i in range(len(episode_reward)):
        y1[i] = episode_reward[i] / 100

    plt.plot(x1, y1)
    plt.savefig('training.png')
    plt.clf()

    x2 = [ x for x in range(100, NUM_EPISODES+1, 100) ]
    sums = []
    for i in range(NUM_EPISODES):
        if i % 100 == 0:
            sums.append(0)
        sums[int(i / 100)] += steps_taken[i]
    y2 = []
    for i in range(len(sums)):
        y2.append(sums[i] / 100)
    plt.plot(x2, y2)
    plt.savefig('steps_taken.png')
    plt.clf()
    
    x3 = [ x for x in range(100, NUM_EPISODES+1, 100) ]
    y3 = []
    for i in range(len(y2)):
        y3.append(0)
        y3[i] = y1[i] / y2[i]
    plt.plot(x3, y3)
    plt.savefig('score.png')
    plt.clf()

    # VISUALIZATION
    env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=SLIPPERY,
        render_mode="human",
        max_episode_steps=MAX_EPISODE_LENGTH,
    )

    print('Visualization started')
    for episode in range(20):
        state = env.reset()[0]
        reward = False
        truncated = False
        print(f"Episode {episode + 1}: ")
        sleep(0.1)
        while not truncated:
            env.render()
            sleep(0.1)
            action = np.argmax(q[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                env.render()
                if reward == 1:
                    print("The goal is reached")
                else:
                    print("Lost")
                sleep(1)
                break

            if truncated:
                env.render()
                print("Move limit reached")
                sleep(1)
                break

            state = next_state
    env.close()

if __name__ == '__main__':
    main()