import numpy as np
from keras.layers import Dense
from keras import Sequential
from keras.optimizers import Adam
import gym



class deepQagent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.01
        self.discount_rate = 0.9
        self.epsilon = 1
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.1
        self.model = self.build_net()

    def build_net(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=self.state_size))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mean_squared_error')

        return model

    def act(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            action_values = self.model.predict(state)[0]
            max_A = np.argmax(action_values)
            poss_actions = np.where(action_values == action_values[max_A])[0] # Check for equally good actions
            action = np.random.choice(poss_actions)

        return action


    def update_Q(self, state, action, reward, state_next):

        # TODO: Add experience replay or simulation for increased learning.
        Q_state = self.model.predict(state)
        Q_state_next = self.model.predict(state_next)
        # target = reward + self.discount_rate*np.amax(Q_state_next[0]) - Q_state[0][action] # Tabular update too slow
        target = reward + self.discount_rate*np.amax(Q_state_next[0]) # Drop the last term

        Q_state[0][action] = target

        # Update model
        self.model.fit(state, Q_state, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':

    env = gym.make('MountainCar-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = deepQagent(state_size, action_size)

    saving = True
    loading = True

    if loading:
        agent.load('./Q-learning-results/deep-Q-model.h5')
        print('Weights loaded')

    episode_rewards = []

    for i_episode in range(1000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        reward_sum = 0

        for t in range(200):
            env.render()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            reward_sum += reward
            agent.update_Q(state, action, reward, next_state)

            state = next_state

            if done:
                break

        if saving and i_episode % 10 == 0:
            agent.save('./Q-learning-results/deep-Q-model.h5')
            np.save('./Q-learning-results/deep-Q-episode_rewards', np.array(episode_rewards))

        episode_rewards.append(reward_sum)
        print("Episode {} finished after {} timesteps".format(i_episode, t + 1))



