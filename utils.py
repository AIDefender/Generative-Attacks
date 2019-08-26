import tensorflow as tf
from tensorflow import keras
import numpy as np


def get_model(epoch=10):
    """[summary]
    
    Keyword Arguments:
        epoch {int} -- number of epoches during training (default: {10})
    
    Returns:
        trained model, together with images and labels for adversarial attack
    """
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = np.reshape(train_images / 255.0, [-1, 28, 28, 1])
    test_images = np.reshape(test_images / 255.0, [-1, 28, 28, 1])

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='valid',
                            activation=tf.nn.relu),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(48, (3, 3), padding='same', activation=tf.nn.relu),
        keras.layers.BatchNormalization(axis=-1, epsilon=1e-6),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(64, (2, 2), padding='same', activation=tf.nn.relu),
        keras.layers.BatchNormalization(axis=-1, epsilon=1e-6),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(3168, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    try:
        model = keras.models.load_model('mnist_model.h5')
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print('Model loaded!')
    except:
        print('Train from scratch!')

        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=epoch)
        model.save('mnist_model.h5')
        _, test_acc = model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)

    return model, test_images, test_labels


class AdversarialBase():
    '''
    Base class for all adversarial algorithms
    '''

    def __init__(self, model, images, labels, loss_object, num_samples=50):
        self.model = model
        self.images = images[:num_samples]
        self.labels = labels[:num_samples]
        self.loss_object = loss_object
        self.num_samples = num_samples

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print('Evaluating model with clean pictures:')
        self._evaluate(self.images, self.labels)

    def attack(self):
        raise NotImplementedError


    def _evaluate(self, images, labels):
        _, test_acc = self.model.evaluate(images, labels)
        print('Test accuracy:', test_acc)

    def _get_num(self, tensor, label=None):
        if label is not None:
            print(label+':')
        num = self.sess.run(tensor)

        return num
