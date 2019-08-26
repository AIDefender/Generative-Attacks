import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import get_model, AdversarialBase


class CW(AdversarialBase):

    def __init__(self, model, images, labels, alpha_list=[0.1, 0.2, 0.3], lambda_=1, grad_steps=10,
                 loss_object=tf.keras.losses.CategoricalCrossentropy()):

        super(CW, self).__init__(model, images, labels, loss_object)

        self.alpha_list = alpha_list
        self.grad_steps = grad_steps
        self.lambda_ = lambda_  # add _ to avoid strange highlight...

    def attack(self):

        for alpha in self.alpha_list:
            print('Generating Adversarial pictures with alpha'+str(alpha)+'......')

            image_probs = self.model.predict(self.images, steps=1)

            perturbed_image = tf.identity(self.images)
            for _ in range(self.grad_steps):
                gradient = self._generate_grad(
                    self.images, perturbed_image, image_probs)
                perturbed_image += alpha*gradient
                perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
            change_in_picture = tf.losses.mean_squared_error(
                tf.squeeze(perturbed_image), tf.squeeze(self.images))
            print("The L2 distance between clean and adversarial picture is",
                  self._get_num(change_in_picture))

            adv_pred = self._get_num(self.model(tf.reshape(
                tf.cast(perturbed_image, tf.float32), [-1, 28, 28, 1])))

            adv_class = adv_pred.argmax(axis=-1)

            attck_suc_rate = np.sum(
                [adv_class != self.labels])/len(self.labels)*100

            print('The success rate of adversarial attack is:', attck_suc_rate, '%')

    def _generate_grad(self, original_image, input_image, input_label):

        input_image = tf.reshape(input_image, [-1, 28, 28, 1])
        
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = self.model(tf.cast(input_image, tf.float32))
            loss = self.loss_object(input_label, prediction)
            
        gradient = tf.gradients(loss, input_image)
        gradient -= 2*self.lambda_*(input_image-original_image)

        return gradient


def main():

    model, test_images, test_labels = get_model()
    algo = CW(model, test_images, test_labels)
    algo.attack()


if __name__ == '__main__':
    main()
