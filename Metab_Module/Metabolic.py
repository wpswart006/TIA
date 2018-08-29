from keras.models import load_model
import numpy as np


class Predictor:

   def __init__(self, scale=210, mean=0.1806061555437761, path_to_model='metabolic_model.h5'):
       """Inititalize the Metabolic module with default paramaters for normalizing the data (scale being the largest value in the dataset, and mean
       being the arithmitic mean of the value's present), and a path to the tensorflow mdel
       """
       self.model = load_model(path_to_model)
       self.scale = scale
       self.mean = mean

   def predict_to_prob(self, age, waist, systolic):
       """ Prediction function that takes a patient's age, waist circumference in centimeter and systolic blood presure in mm/hg
       and returns the predicted probability as a percentage of certainty in the form [chance being not at risk,chance being at risk]
       """
       x = np.array([[age, waist, systolic]]).astype('float32')
       x /= self.scale
       x -= self.mean
       return self.model.predict(x)

   def predict_to_class(self, age, waist, systolic):
       """ Prediction function that takes a patient's age, waist circumference in centimeter and systolic blood presure in mm/hg
       and calls the predict function to retrun a string that indicates the class
       """
       res = self.predict_to_prob(age, waist, systolic)
       if np.argmax(res) == 0:
           return "Little Metabolic Syndrome Risk"
       elif np.argmax(res) == 1:
           return "At risk of Metabolic Syndrome"

   def predict(self, age, waist, systolic):
        """ Prediction function that takes a patient's age, waist circumference in centimeter and systolic blood presure in mm/hg
        and returns either true or false, indicitave of whether a patient is deemd at risk or not
        """
        res = self.predict_to_prob(age, waist, systolic)
        if np.argmax(res) == 0:
           return False
        elif np.argmax(res) == 1:
           return True
        return "Error"
