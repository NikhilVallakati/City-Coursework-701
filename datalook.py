import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import seaborn as sns  

class Datalook:
    def __init__(self, twitter):
        self.twitter = twitter

    def show(self):
        #lets check for null values
        self.twitter.isnull().mean()*100 
        print("self.twitter: ")
        print(self.twitter.head())
        print(f" Data Available since {self.twitter.created_at.min()}")
        print(f" Data Available upto {self.twitter.created_at.max()}")
        #lets check latest and oldest self.twitter members in the dataframe
        print(f" Data Available since {self.twitter.user_created_at.min()}")
        print(f" Data Available upto {self.twitter.user_created_at.max()}")
        print('The oldest user in the data was',self.twitter.loc[self.twitter['user_created_at'] == '2006-03-21 21:04:12', 'user_name'].values)
        print('The newest user in the data was',self.twitter.loc[self.twitter['user_created_at'] == '2019-05-19 10:49:59', 'user_name'].values)
        #lets explore created_at column
        