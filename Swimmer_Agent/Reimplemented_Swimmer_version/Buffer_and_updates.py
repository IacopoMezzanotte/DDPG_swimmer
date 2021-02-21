import gym
import tensorflow as tf
import numpy as np
import Networks

'''changed the return type to implement the plotting of the losses tried also to change
gamma, buffer capacity, batch size'''

problem = "Swimmer-v2"
env = gym.make(problem)
# Discount factor for future rewards
gamma = 0.999


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, num_states=env.observation_space.shape[0],
                 num_actions=env.action_space.shape[0]):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        # My version constructor
        self.num_states = num_states
        self.num_actions = num_actions

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.done_buffer = np.zeros((self.buffer_capacity,1))

    # Takes (s,a,r,s',d) observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, target_actor, target_critic,
               actor_model, critic_model):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.

        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            next_state_batch = tf.cast(next_state_batch, dtype=tf.float32)
            temp = tf.concat((next_state_batch, target_actions), axis=1)
            y = reward_batch + gamma * (1 - done_batch) * target_critic(
                temp, training=True
            )

            critic_value = critic_model(tf.concat((state_batch, action_batch), axis=1), training=True)
            critic_loss = tf.math.reduce_mean((y - critic_value)**2)

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        Networks.critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            state_batch = tf.cast(state_batch, dtype=tf.float32)
            critic_value = critic_model(tf.concat((state_batch, actions), axis=1), training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        Networks.actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

        return critic_loss, actor_loss

    # We compute the loss and update parameters

    def learn(self, target_actor, target_critic, actor_model, actor_critic):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])
        done_batch = tf.cast(done_batch, dtype=tf.float32)

        critic_loss, actor_loss = self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch, target_actor, target_critic,
                    actor_model, actor_critic)
        return critic_loss, actor_loss
