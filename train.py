from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow.keras.applications.vgg19 as vgg19
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import datetime,subprocess


batch_size = 128
STEPS_PER_EPOCH = 100
num_classes = 10
epochs = 500
# save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'keras_cifar10_final_model.h5'

startT=datetime.datetime.now()
log_dir="log/" + startT.strftime("%Y%m%d-%H%M%S")
checkpoint_dir = log_dir + "/checkpoints"
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
save_dir = log_dir + "/saved_models"

# Horovod: initialize Horovod.
hvd.init()
print('local_rank=%d, rank=%d, size=%d, batch size=%d' % (hvd.local_rank(), hvd.rank(), hvd.size(), batch_size))
print(f"Loading data, hvd.size = {hvd.size()}")

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
# x_train = x_train[:500]
y_train = keras.utils.to_categorical(y_train, num_classes)
# y_train = y_train[:500]
y_test = keras.utils.to_categorical(y_test, num_classes)

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_train, tf.float32),
             tf.cast(y_train, tf.int64))
)
dataset = dataset.repeat().shuffle(STEPS_PER_EPOCH*batch_size).batch(batch_size)

model = vgg19.VGG19(include_top=True,pooling='avg',weights=None,classes=10,input_shape=(32,32,3))


# initiate RMSprop optimizer
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
# Horovod: add Horovod DistributedOptimizer.
# TODO: deactivate compression
opt = hvd.DistributedOptimizer(opt, compression=hvd.Compression.fp16)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],
              experimental_run_tf_function=False)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch', profile_batch=0)

# All the callbacks
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=10, verbose=1),
]

if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/checkpoint-{epoch}.h5'))
    callbacks.append(tensorboard_callback)

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0


history = model.fit(dataset,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          # batch_size=batch_size,
          steps_per_epoch=STEPS_PER_EPOCH,
          verbose=verbose,
          callbacks=callbacks)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
