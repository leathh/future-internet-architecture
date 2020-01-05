from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow.keras.applications.vgg19 as vgg19
import tensorflow as tf
import datetime,subprocess


batch_size = 128
STEPS_PER_EPOCH = 100
num_classes = 10
epochs = 500
# save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'keras_cifar10_final_model.h5'
CHECKPOINT = 'final_2m.h5'  # checkpoint to be restored
CHECKf = 'final_m.h5'  # checkpoint to be restored

startT=datetime.datetime.now()
log_dir="log/" + startT.strftime("%Y%m%d-%H%M%S")
checkpoint_dir = log_dir + "/checkpoints"
print(checkpoint_dir)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
save_dir = log_dir + "/saved_models"
# print(save_dir)
time_file_path = log_dir + "time_file.txt"
timing = {}
print(startT)

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
dataset = dataset.repeat().batch(batch_size)

model = vgg19.VGG19(include_top=True,pooling='avg',weights=None,classes=10,input_shape=(32,32,3))
model.load_weights(CHECKf)

# initiate RMSprop optimizer
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)


# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],
              experimental_run_tf_function=False)

'''
Bug causing wrong validation accuracy
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
'''
x_test = x_test.astype('float32')

# TODO: keras.metrics needed??


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
