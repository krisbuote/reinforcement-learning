import gym
import numpy as np

game_name = 'MountainCar-v0'
env = gym.make(game_name)

state_size = env.observation_space.shape[0]  # [position, velocity]. pos range [-1.2, 0.6], vel [-0.07, 0.07]
bounds_high = env.observation_space.high
bounds_low = env.observation_space.low
action_size = env.action_space.n

# Discretize state space
numBins = 100
pos_bins = np.linspace(bounds_low[0], bounds_high[0], numBins)
vel_bins = np.linspace(bounds_low[1], bounds_high[1], numBins)


def find_nearest(array, value):
    ''' Provides index for state space bin'''
    idx = (np.abs(array - value)).argmin()
    return idx


if __name__ == "__main__":

    alpha = 0.1 # Learning Rate
    gamma = 0.9 # Discount Factor
    epsilon = 0.1 # Exploration/Exploitation Threshold
    loading = False
    saving = True

    # Value tensor for discretized state-space. Dimensions: (position, velocity, numActions)
    if loading:
        Q = np.load('./sarsa-save/sarsa_Q_tensor.npy')
    else:
        Q = np.zeros((numBins, numBins, action_size))

    episode_rewards = []
    action_list = np.arange(action_size)

    for i_episode in range(10000):
        observation = env.reset()

        pos_raw, vel_raw = observation[0], observation[1]
        pos_idx, vel_idx = find_nearest(pos_bins, pos_raw), find_nearest(vel_bins, vel_raw)

        reward_sum = 0

        for t in range(500):
            # env.render()

            if np.random.random() > epsilon:
                # Check for equally valuable actions, choose randomly amongst them
                max_action = np.argmax(Q[pos_idx, vel_idx])
                action_choices = np.where(Q[pos_idx, vel_idx, :] == Q[pos_idx, vel_idx, max_action])[0]
                action = np.random.choice(action_choices)
            else:
                action = np.random.choice(action_size)

            observation, reward, done, info = env.step(int(action))
            reward_sum += reward

            next_pos_raw, next_vel_raw = observation[0], observation[1]
            next_pos_idx, next_vel_idx = find_nearest(pos_bins, next_pos_raw), find_nearest(vel_bins, next_vel_raw)

            # SARSA update
            if t > 0:
                Q[pos_idx,vel_idx,last_action] += alpha * (reward + gamma*Q[next_pos_idx,next_vel_idx,action] - Q[pos_idx,vel_idx,last_action])

            pos_idx, vel_idx = next_pos_idx, next_vel_idx
            last_action = action

            # If the goal position is reached
            if pos_bins[pos_idx] in pos_bins[-2:]:
                reward = 1
                Q[pos_idx, vel_idx, action] += reward
                break


        episode_rewards.append(reward_sum)
        print("Episode {} finished after {} timesteps".format(i_episode, t + 1))

        if saving and i_episode % 100 == 0:
            np.save('./sarsa-save/sarsa_Q_tensor', Q)
            np.save('./sarsa-save/sarsa_episode_rewards', np.array(episode_rewards))