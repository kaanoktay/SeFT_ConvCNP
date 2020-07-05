import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import medical_ts_datasets
import numpy as np

from .training_utils import (
    preprocessing
)

from .model import (
    convCNP
)

tf.random.set_seed(0)

def main():
    ## Hyperparameters
    batch_size = 32
    num_modalities = 37
    points_per_hour = 30
    filter_size = 64
    num_epochs = 4
    init_learning_rate = 1e-3
    num_points = 50 * points_per_hour # 48 hours + 1 hour before and later ---> 50 hours

    ## Load data (epochs doesn't matter because it iterates over the dataset indefinetely)
    transformation = preprocessing(dataset='physionet2012', epochs=num_epochs, batch_size=batch_size)
    train_iter, steps_per_epoch, val_iter, val_steps, test_iter, test_steps = transformation._prepare_dataset_for_training()

    ## Create the grid used for the functional representation
    grid = tf.linspace(-1.0, 49.0, num_points)

    ## Initialize the model
    model = convCNP(grid, points_per_hour, num_modalities, batch_size, num_points, filter_size)

    ## Learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init_learning_rate,
        decay_steps=1000,
        decay_rate=0.9
    )

    ## Optimizer function
    opt = keras.optimizers.Adam(
        learning_rate=lr_schedule
    )

    ## Loss function
    loss_fn = keras.losses.BinaryCrossentropy(
        from_logits=False,
        name="Loss"
    )

    ## Compile the model
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), 
                 keras.metrics.AUC(curve="ROC", name="auroc"), 
                 keras.metrics.AUC(curve="PR", name="auprc")]
    )

    ## Fit the model to the input data
    model.fit(
        train_iter,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_iter,
        validation_steps=val_steps,
        verbose=1
    )

if __name__ == "__main__":
    main()