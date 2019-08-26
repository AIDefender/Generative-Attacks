import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from utils import get_model, AdversarialBase


class iFGSM(AdversarialBase):

    def __init__(self, model, images, labels,
                 epsilon=[2, 4, 8, 16], alpha=0.1,
                 loss_object=tf.keras.losses.CategoricalCrossentropy()):

        super(iFGSM, self).__init__(model, images, labels, loss_object)

        self.alpha = alpha/255.0
        self.epsilon = epsilon

        self.image_probs = model.predict(self.images, steps=1)

    def attack(self):

        for _, eps in enumerate(self.epsilon):
            print('Generating Adversarial pictures with epsilon'+str(eps)+'......')
            image_perturbed = tf.identity(self.images)
            for _ in range(int(min(eps+4, 1.25*eps))):
                perturbations = self._create_adversarial_pattern(
                    tf.reshape(image_perturbed, [-1, 28, 28, 1]), self.image_probs)

                adv_x = image_perturbed + self.alpha*perturbations
                adv_x = tf.clip_by_value(adv_x, 0, 1)

                image_perturbed = adv_x

            change_in_picture = tf.losses.mean_squared_error(
                tf.squeeze(image_perturbed), tf.squeeze(self.images))
            print('L2 distance between clean and adversarial pictures:',self._get_num(change_in_picture))


            adv_pred = self._get_num(self.model(tf.reshape(
                tf.cast(adv_x, tf.float32), [-1, 28, 28, 1])))

            adv_class = adv_pred.argmax(axis=-1)

            attck_suc_rate = np.sum(
                [adv_class != self.labels])/len(self.labels)*100
                
            print('The success rate of adversarial attack is:', attck_suc_rate, '%')

    def _create_adversarial_pattern(self, input_image, input_label):

        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = self.model(tf.cast(input_image, tf.float32))
            loss = self.loss_object(input_label, prediction)

        gradient = tf.gradients(loss, input_image)
        
        signed_grad = tf.sign(gradient)
        return signed_grad


def main():

    model, test_images, test_labels = get_model()
    algo = iFGSM(model, test_images, test_labels)
    algo.attack()


if __name__ == '__main__':
    main()
