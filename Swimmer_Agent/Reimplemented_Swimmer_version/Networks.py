import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential

'''we have tried adaptive lr, SGD with adaptive lr and momentum (mid success), adam (mid success)
, many changes in the networks and in the activation function: RElu, PReLU and leaky relu (success), swish'''

# Learning rate for actor-critic models also for exponential decay, and optimizers definition
initial_critic_lr = 0.002
initial_actor_lr = 0.001
''' ***Uncomment this if you want to activate the decayed lr***
lr_scheduler_critic = tf.keras.optimizers.schedules.ExponentialDecay(initial_critic_lr, decay_steps=100, decay_rate=0.9)
lr_scheduler_actor = tf.optimizers.schedules.ExponentialDecay(initial_actor_lr, decay_steps=50, decay_rate=0.8)
   
    *** if you want to use other optimizers here there are some solutions also with decayed lr rate***
# critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler_critic)
# actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler_actor)
# critic_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler_critic, momentum=0.25)
# actor_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler_actor, momentum=0.15)
# critic_optimizer = tf.keras.optimizers.SGD(critic_lr, momentum=0.35)
# actor_optimizer = tf.keras.optimizers.SGD(actor_lr, momentum=0.3)
'''
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_actor_lr)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


'''***here you can modify the target and actor network, some examples:***
out = layers.Dense(128, activation=tf.keras.activations.swish)
out = layers.PReLU(alpha_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
out = layers.Dense(256, activation=tf.nn.leaky_relu)
out = layers.Dense(200, activation=tf.keras.activations.elu)
'''
# Actor definition
def get_actor(num_states):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    inputs = layers.Input(shape=(num_states))
    out = layers.Dense(400, activation="relu")(inputs)
    out = layers.Dense(300, activation="relu")(out)
    outputs = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


'''*** in case you want to unify actions and states uncomment and substitute in the get critic***
    input_shape = 10
    inputs = layers.Input(shape=input_shape)
    out = layers.Dense(400, activation="relu")(inputs)
    out = layers.Dense(300, activation="relu")(out)
    outputs = layers.Dense(1)(out)
'''


# Critic definition
def get_critic(num_states, num_actions):

    # same input for action and states
    input_shape = 10
    inputs = layers.Input(shape=input_shape)
    out = layers.Dense(400, activation=tf.keras.activations.elu)(inputs)
    out = layers.Dense(300, activation=tf.keras.activations.elu)(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model
