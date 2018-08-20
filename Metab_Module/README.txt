Metabolic is a class that is used to predict the chance that a set of data might be an indication of Metabolic Syndrome.

The class has 3 methods: a constructor, __init__, numerical predict method, predict, and a class prediction method, predict_to_class.

__init__(self, scale=210, mean=0.1806061555437761, path_to_model='metabolic_model.h5'):
Inititalize the Metabolic module with default paramaters for normalizing the data (scale being the largest value in the dataset, and mean
being the arithmitic mean of the value's present), and a path to the tensorflow mdel.

predict(self, age, waist, systolic):
Prediction function that takes a patient's age, waist circumference in centimeter and systolic blood presure in mm/hg
and returns the predicted probability as a percentage of certainty in the form [chance being not at risk,chance being at risk]

predict_to_class(self, age, waist, systolic):
Prediction function that takes a patient's age, waist circumference in centimeter and systolic blood presure in mm/hg
and calls the predict function to retrun a string that indicates the class

The required packages can be installed by runnung:

pip install -r requirements.txt