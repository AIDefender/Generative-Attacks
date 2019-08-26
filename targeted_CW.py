import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import get_model, AdversarialBase
import matplotlib.pyplot as plt
import copy
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target', '-t', type=int, default=3)
args = parser.parse_args()


class TargetedCW(AdversarialBase):

    def __init__(self, model, images, labels, alpha_list=[10], lambda_=0.0001, grad_steps=300,
                 target=3, loss_object=tf.keras.losses.CategoricalCrossentropy()):

        super(TargetedCW, self).__init__(
            model, images, labels, loss_object, num_samples=10)

        self.alpha_list = alpha_list
        self.grad_steps = grad_steps
        self.target = target
        self.lambda_ = lambda_  # add _ to avoid strange highlight...
        print('Original classes:', self.model.predict_classes(self.images))

    def attack(self):

        for alpha in self.alpha_list:
            print('Generating Adversarial pictures with alpha'+str(alpha)+'......')

            lamd = self.lambda_
            perturbed_image = copy.deepcopy(self.images)
            for i in range(self.grad_steps):
                if i % 10 == 0:
                    gradient, loss = self._generate_grad(
                        self.images, perturbed_image, required_label=self.target)
                    print('CE loss after {} steps'.format(i), loss)
                else:
                    gradient, _ = self._generate_grad(
                        self.images, perturbed_image, required_label=self.target)
                if i % 50 == 0 and i != 0:
                    if alpha > 3:
                        alpha *= 0.995
                        print('alpha adjusted with value', alpha)
                    if lamd < 0.03:
                        lamd *= 1.02
                        print('lambda adjusted with value', lamd)

                perturbed_image -= alpha*gradient
                perturbed_image = np.clip(perturbed_image, 0, 1)
                if loss < 2.23:  # empirical threshold
                    break
                if i % 150 == 0 and i != 0:
                    message = input(
                        'Not yet achieved threshold 2.23, terminal or not?(y/n)')
                    if message == 'y':
                        break

            change_in_picture = tf.losses.mean_squared_error(
                tf.squeeze(perturbed_image), tf.squeeze(self.images))
            print("The L2 distance between clean and adversarial picture is",
                  self._get_num(change_in_picture))

            adv_pred = self._get_num(self.model(tf.reshape(
                tf.cast(perturbed_image, tf.float32), [-1, 28, 28, 1])))

            adv_class = adv_pred.argmax(axis=-1)
            print('adversarial class = {}'.format(adv_class))

            attck_suc_rate = np.sum(
                [adv_class == [self.target for _ in range(self.num_samples)]])/len(self.labels)*100

            print('The success rate of targeted adversarial attack is:',
                  attck_suc_rate, '%')
            for i in range(self.num_samples):
                plt.imshow(np.squeeze(perturbed_image[i]), cmap='gray')
                plt.savefig('./pictures/' +
                            str(self.labels[i])+'to'+str(self.target))
                print('picture saved!')

    def _one_hot(self, loc, size):
        a = np.zeros(size)
        a[:, loc] = 1

        return a

    def _generate_grad(self, original_image, input_image, required_label):

        input_label = self._one_hot(
            loc=required_label, size=(self.num_samples, 10))
        input_image = tf.reshape(input_image, [-1, 28, 28, 1])

        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = self.model(tf.cast(input_image, tf.float32))
            loss = self.loss_object(input_label, prediction)
            
        gradient = tf.gradients(loss, input_image)
        gradient += 2*self.lambda_*(input_image-original_image)

        return np.reshape(self._get_num(gradient), (-1, 28, 28, 1)), self._get_num(loss)


def main():

    model, test_images, test_labels = get_model()
    algo = TargetedCW(model, test_images, test_labels, target=args.target)
    algo.attack()


if __name__ == '__main__':
    main()
