# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:19:30 2018

@author: WP
"""

import json,numpy as np


class Predictor:
    def __init__(self,symptoms = "symptoms.json",thresh = 15):
        """
        Initialize the Vitamin a deficiency module with default paramters symptoms and thresh, where symptoms is a a string pointing to the path of thejson file
        and thresh is the threshold value in which case a patient is at risk
        """
        self.thresh = thresh
        with open(symptoms,'r') as read:
            self.data = json.load(read)

    def predict(self,symptoms):
        """
        Predictor method that takes as paramater a list of numbers as long as the number of symptoms in symptoms.json. Returns true if a candiate is at risk,
        False if not, and error otherwise
        """        
        if not len(symptoms) == len(self.data):
            return "Error"
        _symptoms = np.array(symptoms)
        score = 0
        for i,s in enumerate(self.data):
            score +=self.data[s]['Significance']*symptoms[i]
        score*=np.sum(symptoms)
        
        if score > self.thresh:
            return True
        return False
