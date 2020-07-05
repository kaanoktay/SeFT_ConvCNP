import os
import sys
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import medical_ts_datasets
import numpy as np

from .training_utils import (
    preprocessing,
    argumentParser
)

from .model import (
    convCNP
)

checkpoint_filepath = '/Users/kaanoktay/Desktop/Master/SeFT_ConvCNP/checkpoints/cp.ckpt'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(0)
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

def main():
    ## Parse the command line arguments
    args = argumentParser()

    ## Hyperparameters
    num_modalities = 37 #Constant for this dataset
    batch_size = args.batch_size #Default: 16
    points_per_hour = args.points_per_hour #Default: 30
    num_epochs = args.num_epochs #Default: 10
    init_learning_rate = args.init_learning_rate #Default: 1e-3
    kernel_size = args.kernel_size #Default: 5
    dropout_rate_conv = args.dropout_rate_conv #Default: 0.2
    dropout_rate_dense = args.dropout_rate_dense #Default: 0.2
    filter_size = args.filter_size #Default: 64
    lr_decay_steps = args.lr_decay_steps #Default: 1000
    lr_decay_rate = args.lr_decay_rate #Default: 0.95

    ## Load data (epochs doesn't matter because it iterates over the dataset indefinetely)
    transformation = preprocessing(dataset='physionet2012', epochs=num_epochs, batch_size=batch_size)
    train_iter, steps_per_epoch, val_iter, val_steps, test_iter, test_steps = transformation._prepare_dataset_for_training()

    ## Create the grid used for the functional representation
    num_points = 50 * points_per_hour # 48 hours + 1 hour before and later ---> 50 hours
    grid = tf.linspace(-1.0, 49.0, num_points)

    ## Initialize the model
    model = convCNP(grid, points_per_hour, num_modalities, batch_size, num_points, 
                    kernel_size, dropout_rate_conv, dropout_rate_dense, filter_size)

    ## Learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init_learning_rate,
        decay_steps=lr_decay_steps,
        decay_rate=lr_decay_rate
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

    ## Callback for saving the weights of the best model
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_auprc',
        mode='max',
        save_best_only=True
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
        steps_per_epoch=steps_per_epoch-1,
        validation_data=val_iter,
        validation_steps=val_steps-1,
        verbose=1,
        callbacks=[model_checkpoint_callback]
    )

if __name__ == "__main__":
    main()
