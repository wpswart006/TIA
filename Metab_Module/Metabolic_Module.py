from keras.models import load_model
import numpy as np
class Metabolic:
    
   def __init__(self,scale = 210,mean = 0.1806061555437761,path_to_model = 'metabolic_model.h5'):
       self.model = load_model(path_to_model)
       self.scale =scale
       self.mean = mean
       
   def predict(self,age,waist,systolic):
       x = np.array([[age,waist,systolic]]).astype('float32')
       x /= self.scale
       x -= self.mean
       return self.model.predict(x)
   
   def predict_to_class(self,age,waist,systolic):
       res =self.predict(age,waist,systolic)
       if np.argmax(res) == 0:
           return "Little Metabolic Syndrome Risk"
       elif np.argmax(res) == 1:
           return "At risk of Metabolic Syndrome"
       
