import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import get_model,AdversarialBase
import copy

class BlackBoxCW(AdversarialBase):

    def __init__(self, model,images,labels,alpha_list=[10],lambda_=1,grad_steps=30,
                    loss_object=tf.keras.losses.CategoricalCrossentropy()):
        
        super(BlackBoxCW,self).__init__(model,images,labels,loss_object,num_samples=100)     
        
        self.alpha_list=alpha_list
        self.grad_steps=grad_steps
        self.lambda_=lambda_ # add _ to avoid strange highlight...

    def attack(self):

        for alpha in self.alpha_list:

            image_probs = self.model.predict(self.images, steps=1)

            perturbed_image=copy.deepcopy(self.images)


            for j in range(self.grad_steps):
                coordinate=np.random.choice(28)
                gradient=self._generate_grad(self.images,perturbed_image,image_probs,coordinate)

                perturbed_image[:,coordinate,:,:]+=alpha*gradient
                perturbed_image=np.clip(perturbed_image,0,1)


            change_in_picture=tf.losses.mean_squared_error(tf.squeeze(perturbed_image),tf.squeeze(self.images))
            print("The L2 distance between clean and adversarial picture is",self._get_num(change_in_picture))
            
            adv_pred=self._get_num(self.model(tf.reshape(tf.cast(perturbed_image,tf.float32),[-1,28,28,1])))
            
            adv_class = adv_pred.argmax(axis=-1)
            print('adversarial class = {}'.format(adv_class))

            attck_suc_rate=np.sum([adv_class!=self.labels])/len(self.labels)*100
            print('The success rate of adversarial attack is:',attck_suc_rate,'%')


      

    def _generate_grad(self,original_image,input_image, input_label,coordinate):

        input_image=tf.reshape(input_image,[-1,28,28,1])
        def f(image):
            return self._get_num(self.model(tf.cast(image,tf.float32)))

        prediction = self._get_num(self.model(tf.cast(input_image,tf.float32)))

        gradient_loss_to_pred=-np.sum(input_label/prediction,axis=-1)
        gradient_pred_to_img=self._zoo(original_image,f,coordinate)
        gradient=np.reshape(np.reshape(gradient_loss_to_pred,[-1,1,])*gradient_pred_to_img,[-1,28,1])


        gradient-=2*self.lambda_*(input_image[:,coordinate,:,:]-original_image[:,coordinate,:,:])

        return np.reshape(self._get_num(gradient),[-1,28,1])

    def _zoo(self,image,f,coordinate):
        h=1
        grad=np.zeros((np.shape(image)[0],28))
        for i in range(28):

            eye=np.zeros_like(image)
            eye[:,coordinate,i,:]=1
            
            grad[:,i]=np.sum((f(image+h*eye)-f(image-h*eye)))/(2*h)

        return grad
      
def main():
    
    model,test_images,test_labels=get_model()
    algo=BlackBoxCW(model,test_images,test_labels)
    algo.attack()


if __name__=='__main__':
    main()
