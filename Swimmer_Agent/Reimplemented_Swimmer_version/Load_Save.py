import os
from keras.models import load_model

# Save the weights
def save(actor_model, critic_model, target_actor, target_critic):
    print("Do you want to save the model? [y for yes]")
    response = input("answere: ")
    if response == 'y':
        actor_model.save_weights("swimmer_actor.h5")
        critic_model.save_weights("swimmer_critic.h5")

        target_actor.save_weights("swimmer_target_actor.h5")
        target_critic.save_weights("swimmer_target_critic.h5")


'''***remember to add your path***'''
# Load models
models_dir = 'path to the models'
def loadmodel():

    filename_a = os.path.join(models_dir, 'swimmer_actor.h5')
    filename_c = os.path.join(models_dir, 'swimmer_critic.h5')
    filename_ta = os.path.join(models_dir, 'swimmer_target_actor.h5')
    filename_tc = os.path.join(models_dir, 'swimmer_target_critic.h5')
    filenames = []
    models = []
    filenames.append(filename_a)
    filenames.append(filename_c)
    filenames.append(filename_ta)
    filenames.append(filename_tc)

    for filename in filenames:
        try:
            models.append(load_model(filename))
            print("\nModel loaded successfully from file %s\n" % filename)
        except OSError:
            print("\nModel file %s not found!!!\n" % filename)

    return models



