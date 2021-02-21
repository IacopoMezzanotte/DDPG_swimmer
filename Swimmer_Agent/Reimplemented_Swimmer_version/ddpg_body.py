import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Noise
import Networks
import Buffer_and_updates
import Load_Save


'''created a test function, and a function for the plotting of the learning rate in case of adaptive learning rate
implemented also a time step decayed noise factor (see OUActionNoise) for a random noise for a better exploration 
with noise modulation,added the chance to choose an arbitrary amount of beginning completely exploration steps
added many plots and networks's summaries, also tried to change hyperparameters
'''

problem = "Swimmer-v2"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


# Noise's parameters
initial_noise_factor = 1.1
std_dev = 0.3
ou_noise = Noise.OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# Number of episode
total_episodes = 350

# Number of max time step per episode and number of total exploration step
max_episode_length = 1000
beginning_exploration_steps = 100000

# Used to update target networks
tau = 0.005

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
''' ***in case od decayed learning rate and you want to plot this just uncomment the following list*** 
# To store learning rates
lr_critic_list = []
lr_actor_list = []
'''
# To store noise's amount
total_noises = []
random_noises = []
# To store actions
actions_1 = []
actions_2 = []
# To store losses
actor_losses = []
critic_losses = []


# Testing
def test():
    n = 0
    state = env.reset()
    test_reward = 0
    while n <= max_episode_length:
        env.render()
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        a = np.array(actor_model(state))
        state, rew, d, _ = env.step(a)
        n +=1
        test_reward += rew
        if d:
            break
    print("test reward is ==> {}".format(test_reward))
    env.close()


'''
# To plot lr in case of adaptive one
def decayed_learning_rate_for_plotting(step, initial_learning_rate, decay_rate, decay_steps):
    return initial_learning_rate * decay_rate ** (step / decay_steps)
'''

# Creating networks
actor_model = Networks.get_actor(num_states)
critic_model = Networks.get_critic(num_states, num_actions)
target_actor = Networks.get_actor(num_states)
target_critic = Networks.get_critic(num_states, num_actions)
print("Do you want to Load the model? [y for yes]")
response = input("answere: ")
if response == 'y':
    actor_model_weights = Load_Save.loadmodel()[0]
    critic_model_weights = Load_Save.loadmodel()[1]
    target_actor_weights = Load_Save.loadmodel()[2]
    target_critic_weights = Load_Save.loadmodel()[3]
    actor_model.set_weights(actor_model_weights)
    critic_model.set_weights(critic_model_weights)
    target_actor.set_weights(target_actor_weights)
    target_critic.set_weights(target_critic_weights)
else:
    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())


buffer = Buffer_and_updates.Buffer(50000, 64, num_states, num_actions)


# Policy definition
def policy(state, noise_object, noise_factor, ep):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    total_noise = noise_factor * noise
    sampled_actions = sampled_actions.numpy() + total_noise
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    #To plot
    if ep == (total_episodes - 25):
        total_noises.append(total_noise)
        random_noises.append(noise_factor)

    return [np.squeeze(legal_action)]


# Main loop
n_step = 0
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0
    ep_length = 0
    noise_factor = initial_noise_factor
    while not (ep_length == max_episode_length):

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        if n_step > beginning_exploration_steps:
            actual_random_noise_factor = Noise.random_noise(noise_factor, n_step)
            action = np.array(policy(tf_prev_state, ou_noise, actual_random_noise_factor, ep))
            noise_factor = actual_random_noise_factor

        else:
            action = env.action_space.sample()
            action = np.expand_dims(action, axis=0)

        n_step += 1
        if n_step == beginning_exploration_steps:
            print("EXPLORATION PHASE ENDED ,NOW USING TRUE ACTION NOT SAMPLED ONE")

        # Receieve state and reward from environment.
        state, reward, done, info = env.step(action)
        buffer.record((prev_state, action, reward, state, done))
        episodic_reward += reward

        if n_step > beginning_exploration_steps and ((ep % 100) == 0):
            actions_1.append(action[0][0])
            actions_2.append(action[0][1])
        '''***uncomment the followings rows in case of adaptive lr, keep an aye to the parameters***
              (has to be equals to the ones in Networks.lr_scheduler_critic and etworks.lr_scheduler_actor)
        if ep == 1:
            a_lr = decayed_learning_rate_for_plotting(n_step, Networks.initial_actor_lr, 0.8, 50)
            c_lr = decayed_learning_rate_for_plotting(n_step, Networks.initial_critic_lr, 0.9, 50)
            lr_critic_list.append(a_lr)
            lr_actor_list.append(c_lr)
        '''
        critic_loss, actor_loss = buffer.learn(target_actor, target_critic, actor_model, critic_model)
        Networks.update_target(target_actor.variables, actor_model.variables, tau)
        Networks.update_target(target_critic.variables, critic_model.variables, tau)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    print("Episodic Reward n° {} is ==> {}".format(ep, episodic_reward))
    # Mean of last 100 episodes
    avg_reward = np.mean(ep_reward_list[-100:])

    avg_reward_list.append(avg_reward)
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))

# Plotting graphs
test()
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel(" Epsiodic Reward")
plt.show()
plt.plot(total_noises)
plt.xlabel("Time step per Episode")
plt.ylabel(" Episodic noise")
plt.show()
plt.plot(random_noises)
plt.xlabel("Time Step per episode")
plt.ylabel("decayed noise factor")
plt.show()

'''***To plot lr in case of adaptive one just uncomment***
plt.plot(lr_critic_list)
plt.xlabel("Time Step per episode")
plt.ylabel("critic_lr")
plt.show()
plt.plot(lr_actor_list)
plt.xlabel("Time Step per episode")
plt.ylabel("actor_lr")
plt.show()
'''

plt.show()
plt.plot(actions_1)
plt.xlabel("Time Step per episode")
plt.ylabel("actions_1")
plt.show()
plt.plot(actions_2)
plt.xlabel("Time Step per episode")
plt.ylabel("actions_2")
plt.show()
plt.show()
plt.plot(critic_losses)
plt.xlabel("Time Step per episode")
plt.ylabel("critic loss")
plt.show()
plt.plot(actor_losses)
plt.xlabel("Time Step per episode")
plt.ylabel("actor loss")
plt.show()

# Save models
Load_Save.save(actor_model, critic_model, target_actor, target_critic)
