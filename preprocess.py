from numpy import array
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import random as random
import emd
import seaborn as sns
#import pyemd as pyemd
import os.path
import threading
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import optimizers
from keras import backend as K
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


from PyEMD import EMD, EEMD, Visualisation
import datetime

from sklearn.preprocessing import MinMaxScaler

from numpy.random import seed
seed(6345212)
tf.random.set_seed(6345213)
NUM_OF_ITR = 1
TEST_NUM = 1000


# data = pd.read_csv("C:\\Users\\Warren\\Desktop\\1-1000 målepunkter.csv", sep=";", chunksize=100000, start=1, error_bad_lines=False, engine='python')
# print(data.get_chunk(1000))

HOUSE_NUMBER_START = 1
HOUSE_NUMBER_END = 5000
NUMBER_OF_HOUSE_SAMPLES = 200
DATA_FOLDER = "C:\\Users\\Warren\\Desktop\\HouseData\\"
DATA_FOLDER_FIGURES = "C:\\Users\\Warren\\Desktop\\HouseData\\Figures\\"
PREMADE_CSV = DATA_FOLDER + "premade.csv"
ORIG_DATA = DATA_FOLDER + "data\\"
MOVING_AVG_CSV = DATA_FOLDER + "moving_average_{}_{}_{}_{}.csv"

SW_IMF = DATA_FOLDER + "sw_imf_{}_{}_{}_{}.csv" #date, Lookback, seasonal, weekday_weekend
SW_IMF1 = DATA_FOLDER + "sw_imf1.csv"
SW_IMF2 = DATA_FOLDER + "sw_imf2.csv"
SW_IMF3 = DATA_FOLDER + "sw_imf3.csv"
SW_IMF4 = DATA_FOLDER + "sw_imf4.csv"
SW_IMF5 = DATA_FOLDER + "sw_imf5.csv"

SEASON = ['fall', 'spring','winter','summer']
DAYS_OF_WEEK = [0,1,2,3,4]
YEAR = [2015,2016,2017,2018,2019]
YEAR2 = [2015,2016,2017,2018]
WEEK_NUM = [22]
TEST_WEEK_NUM = [23]
TRAIN_DAY_OF_YEAR = [158,159,160,161,162,163]
TEST_DAY_OF_YEAR = [164]

TRAIN_DAY_OF_WEEK = [2]
TEST_DAY_OF_WEEK = [2]

#test_date_string = "15/01/{}".format(2019)

ENABLE_WEEKDAY_SPLIT = False
ENABLE_SEASON_SPLIT = True
TEST_TYPE = 'EMD'
LOOKBACK = 4
DAYS = 0

env_setup_cases = [

    # lookback test
    {
        "lookback": 1, "season_split": True,"weekday_split": True
    },
    {
        "lookback": 2, "season_split": True,"weekday_split": True
    },
    {
        "lookback": 4, "season_split": True,"weekday_split": True
    },
    {
        #"lookback": 8, "season_split": True,"weekday_split": True
    },
    {
        #"lookback": 12, "season_split": True,"weekday_split": True
    },
    #seasonal_weekday+weekend (winter and summer test days)
    {
        "lookback": 4, "season_split": True, "weekday_split": False
    },
    # allseason_weekday (winter and summer test days)
    {
        "lookback": 4, "season_split": False, "weekday_split": True
    },
    #allseason_weekday+weekend (winter and summer test)
    {
        "lookback": 4, "season_split": False, "weekday_split": False
    },

]

season_dates_list = [
    # "18/07/{}".format(2018),
    # "18/01/{}".format(2019),
    # "17/04/{}".format(2019),
    # "17/10/{}".format(2018),
    "24/12/{}".format(2018),  # Christmas eve
    "25/12/{}".format(2018),  # Christmas
    "26/12/{}".format(2018),  # Christmas 26th

]

other_season_dates_list = [
               # "18/07/{}".format(2018),
               # "18/01/{}".format(2019),
               # "17/04/{}".format(2019),
               # "17/10/{}".format(2018),
    "24/12/{}".format(2018),  # Christmas eve
    "25/12/{}".format(2018),  # Christmas
    "26/12/{}".format(2018),  # Christmas 26th

]

season_dates_list_weekends_monday = [
              "15/10/{}".format(2018),
              "16/07/{}".format(2018), #Monday
              "15/04/{}".format(2018),
              "15/07/{}".format(2018), #Sunday
              "15/01/{}".format(2018),
]

holidays_dates_list = [
#    "15/07/{}".format(2018),  # the football game
#    "16/07/{}".format(2018),  # day after the football game
#
#             "11/02/{}".format(2018), #fastelavn # Sunday
#               "01/04/{}".format(2018), # Easter # Sunday
#               #"17/05/{}".format(2018), # Great Prayer Day #
#               "01/05/{}".format(2018), # International Workers Day
#               "05/06/{}".format(2018), # Danish Constitution Day
#               "23/06/{}".format(2018), # Sankt Hans
#               "11/11/{}".format(2018), # Saint Mortens
               "24/12/{}".format(2018), # Christmas eve
               "25/12/{}".format(2018),# Christmas
               "26/12/{}".format(2018),# Christmas 26th
#               "31/12/{}".format(2018),# New Years Eve
 ]

PRODUCE_IMFS = True
IMF_GRAPH = False
EMD_XX = 100
test_christmas = False

def get_random_color(size):
    letters = '0123456789abcdef';
    color = '#';

    for i in range(0,6):
        color += letters[random.randint(0,15)];

    return color;


TEST_CASES_CNN0_LSTM_TIMESTEPS_2LAYERS = [
   ###############################
    {
        "Case": "CNN", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64,
        "epochSize": 50,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 210,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 240", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 240,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
]

TEST_CASES_CNN_KERNEL_SIZES_2LAYERS = [
   ###############################

    {
        "Case": "4", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2":4, "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "8", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2":8, "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "12", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2":12, "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2":24, "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "3 days", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2":72, "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "1-week", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2":24*5*1, "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
]

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK4_WEEKDAYS = [
   ###############################

    {
        "Case": "CNN", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64,
        "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 210,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 240", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 240,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # LOOKBACK4_WEEKDAYS

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK4_WEEKDAYS_24kernel = [
   ###############################
{
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24,  "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # _LOOKBACK4_WEEKDAYS_24kernel

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK4_WEEKDAYS_EMD = [
   ###############################

    {
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 210,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # _LOOKBACK4_WEEKDAYS_EMD

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK4_WEEKDAYS_EMD_24kernel = [
   ###############################

    {
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "filterSize2": 48, "kernelSize2": 24, "epochSize": 50,
        "timestep": 480, "sample": 1, "lstmNodes": 210,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # _LOOKBACK4_WEEKDAYS_EMD_24kernel

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK4_WEEKENDS_WEEKDAYS = [
   ###############################
{
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # LOOKBACK4_WEEKENDS_WEEKDAYS

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK4_WEEKENDS_WEEKDAYS_24kernel = [
   ###############################
{
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # LOOKBACK4_WEEKENDS_WEEKDAYS_24kernel

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK4_WEEKENDS_WEEKDAYS_EMD = [
   ###############################

    {
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 210,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 240", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 240,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] #

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK4_WEEKENDS_WEEKDAYS_EMD_24kernel = [
   ###############################
{
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 672, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": True, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # _LOOKBACK4_WEEKENDS_WEEKDAYS_EMD_24kernel


TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK2 = [
   ###############################
    {
        "Case": "CNN", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64,
        "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 240, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 240, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 240, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 240, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 240, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 240, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 250", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 240, "sample": 1, "lstmNodes": 250,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 300", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 240, "sample": 1, "lstmNodes": 300,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
]

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK1 = [
   ###############################
    {
        "Case": "CNN", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64,
        "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
]

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK1_WEEKDAYS_3kernel = [
   ###############################
{
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24,  "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # _LOOKBACK1_WEEKDAYS_3kernel

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK1_WEEKDAYS_WEEKENDS_3kernel = [
   ###############################
{
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24,  "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # _LOOKBACK1_WEEKDAYS_WEEKENDS_3kernel

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK1_WEEKDAYS_24kernel = [
   ###############################
{
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24,  "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # _LOOKBACK1_WEEKDAYS_24kernel

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK1_WEEKDAYS_WEEKENDS_24kernel = [
   ###############################
{
        "Case": "CNN-LSTM 24", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 24,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
{
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24,  "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # _LOOKBACK1_WEEKDAYS_WEEKENDS_24kernel

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK1_WEEKDAYS_WEEKENDS_Holiday_24kernel = [
   ###############################
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # _LOOKBACK1_WEEKDAYS_WEEKENDS_Holiday_24kernel

TEST_CASES_CNN_LSTM_TIMESTEPS = [
   ###############################
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "filterSize2": 48, "kernelSize2": 24, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 200,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
] # _LOOKBACK4_WEEKDAYS_WEEKENDS_Holiday_24kernel

TEST_CASES_CNN_LSTM_TIMESTEPS_LOOKBACK1_WEEKENDS_WEEKDAYS = [
   ###############################
    {
        "Case": "CNN", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64,
        "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "EMD CNN", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64,
        "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 180", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 180,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 200", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 210,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 240", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 168, "sample": 1, "lstmNodes": 240,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
]

TEST_CASES_CNN_LSTM_TIMESTEPS_CNN_KERNEL_TEST_POOLING = [
   ###############################

    {
        "Case": "Pooling Kernel 4", "CaseNum":1, "filterSize1":48, "kernelSize1":4,
        "batchSize":64, "epochSize":50,
        "activation": "relu", "removeNoise":False, "addPool":True, "lstm":False, "color": "purple", "model": "cnn"
    },
    # {
    #     "Case": "NO Pooling Kernel 4", "CaseNum":1, "filterSize1":48, "kernelSize1":4,
    #     "batchSize":64, "epochSize":50,
    #     "activation": "relu", "removeNoise":False, "addPool":False, "lstm":False, "color": "green", "model": "cnn"
    # },
{
        "Case": "Pooling Kernel 3", "CaseNum":1, "filterSize1":48, "kernelSize1":3,
        "batchSize":64, "epochSize":50,
        "activation": "relu", "removeNoise":False, "addPool":True, "lstm":False, "color": "purple", "model": "cnn"
    },
    # {
    #     "Case": "NO Pooling Kernel 3", "CaseNum":1, "filterSize1":48, "kernelSize1":3,
    #     "batchSize":64, "epochSize":50,
    #     "activation": "relu", "removeNoise":False, "addPool":False, "lstm":False, "color": "green", "model": "cnn"
    # },
] #_CNN_KERNEL_TEST_POOLING

TEST_CASES_CNN_LSTM_TIMESTEPS_CNN_KERNEL_TEST = [
   ###############################

    {
        "Case": "kernel 3", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 3,
        "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "kernel 4", "CaseNum":1, "filterSize1":48, "kernelSize1":4,
        "batchSize":64, "epochSize":50,
        "activation": "relu", "removeNoise":False, "addPool": False, "lstm":False, "color": "green", "model": "cnn"
    },
{
        "Case": "kernel 12", "CaseNum":1, "filterSize1":48, "kernelSize1":12,
        "batchSize":64, "epochSize":50,
        "activation": "relu", "removeNoise":False, "addPool": False, "lstm":False, "color": "green", "model": "cnn"
    },
{
        "Case": "kernel 24", "CaseNum":1, "filterSize1":48, "kernelSize1":24,
        "batchSize":64, "epochSize":50,
        "activation": "relu", "removeNoise":False, "addPool": False, "lstm":False, "color": "green", "model": "cnn"
    },

] #_CNN_KERNEL_TEST

TEST_CASES_CNN_LSTM_TIMESTEPS_CNN_KERNEL_TEST_2LAYERS_24kernel = [
   ###############################

    {
        "Case": "kernel 3 48f", "CaseNum": 1, "filterSize1": 48, "filterSize2": 48, "kernelSize1": 3, "kernelSize2": 24,
        "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "kernel 4 48f", "CaseNum":1, "filterSize1":48, "filterSize2": 48, "kernelSize1":4,  "kernelSize2": 24,
        "batchSize":64, "epochSize":50,
        "activation": "relu", "removeNoise":False, "addPool": False, "lstm":False, "color": "green", "model": "cnn"
    },
    {
        "Case": "kernel 3 24f", "CaseNum": 1, "filterSize1": 48, "filterSize2": 24, "kernelSize1": 3, "kernelSize2": 24,
        "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "kernel 4 24f", "CaseNum": 1, "filterSize1": 48, "filterSize2": 24, "kernelSize1": 4, "kernelSize2": 24,
        "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },

] #_CNN_KERNEL_TEST_2LAYERS_24kernel

TEST_CASES_CNN_LSTM_TIMESTEPS_CNN_KERNEL_TEST_2LAYERS_4kernel = [
   ###############################
    {
        "Case": "kernel 4 12f", "CaseNum": 1, "filterSize1": 48, "filterSize2": 48, "kernelSize1": 4, "kernelSize2": 4,
        "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
] #_CNN_KERNEL_TEST_2LAYERS_4kernel

TEST_CASES_CNN_LSTM_TIMESTEPS_CNN_FILTER_TEST = [
   ###############################
    {
        "Case": "Flt 24", "CaseNum":1, "filterSize1":24, "kernelSize1":4,
        "batchSize":64, "epochSize":50,
        "activation": "relu", "removeNoise":False, "addPool":False, "lstm":False, "color": "blue", "model": "cnn"
    },
    {
        "Case": "Flt 48", "CaseNum":1, "filterSize1":48, "kernelSize1":4,
        "batchSize":64, "epochSize":50,
        "activation": "relu", "removeNoise":False, "addPool":False, "lstm":False, "color": "blue", "model": "cnn"
    },
    {
        "Case": "Flt 52", "CaseNum":1, "filterSize1":52, "kernelSize1":4,
        "batchSize":64, "epochSize":50,
        "activation": "relu", "removeNoise":False, "addPool":False, "lstm":False, "color": "blue", "model": "cnn"
    },
    {
        "Case": "Flt 100", "CaseNum": 1, "filterSize1": 100, "kernelSize1": 4,
        "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "blue", "model": "cnn"
    },
    {
        "Case": "Flt 150", "CaseNum": 1, "filterSize1": 150, "kernelSize1": 4,
        "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "blue", "model": "cnn"
    },
{
        "Case": "Flt 200", "CaseNum": 1, "filterSize1": 200, "kernelSize1": 4,
        "batchSize": 64, "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "blue", "model": "cnn"
    },
] #

A_TEST_CASES = [
    {
        "Case":"Increasing EPOCH", "CaseNum":1, "filterSize1":24, "filterSize2":24, "kernelSize1":2, "kernelSize2":2, "batchSize":32, "epochSize":50
    },{
        "Case":"Increasing EPOCH", "CaseNum":2, "filterSize1":24, "filterSize2":24, "kernelSize1":2, "kernelSize2":2, "batchSize":32, "epochSize":100
    },{
        "Case":"Increasing EPOCH", "CaseNum":3, "filterSize1":24, "filterSize2":24, "kernelSize1":2, "kernelSize2":2, "batchSize":32, "epochSize":150
    },{
        "Case":"Increasing EPOCH", "CaseNum":4, "filterSize1":24, "filterSize2":24, "kernelSize1":2, "kernelSize2":2, "batchSize":32, "epochSize":200
    },{
        "Case":"Increasing Batch 32 to 64", "CaseNum":1, "filterSize1":24, "filterSize2":24, "kernelSize1":2, "kernelSize2":2, "batchSize":64, "epochSize":50
    },{
        "Case":"Increasing Batch 32 to 64", "CaseNum":2, "filterSize1":24, "filterSize2":24, "kernelSize1":2, "kernelSize2":2, "batchSize":64, "epochSize":100
    },{
        "Case":"Increasing Batch 32 to 64", "CaseNum":3, "filterSize1":24, "filterSize2":24, "kernelSize1":2, "kernelSize2":2, "batchSize":64, "epochSize":150
    },{
        "Case":"Increasing Batch 32 to 64", "CaseNum":4, "filterSize1":24, "filterSize2":24,
        "kernelSize1":2,
        "kernelSize2":2,
        "batchSize":64,
        "epochSize":200
    },{
        "Case":"Increasing kernal 2 to 3",
        "CaseNum":1,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":3,
        "kernelSize2":3,
        "batchSize":64,
        "epochSize":50
    },{
        "Case":"Increasing kernal 2 to 3",
        "CaseNum":2,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":3,
        "kernelSize2":3,
        "batchSize":64,
        "epochSize":100
    },{
        "Case":"Increasing kernal 2 to 3",
        "CaseNum":3,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":3,
        "kernelSize2":3,
        "batchSize":64,
        "epochSize":150
    },{
        "Case":"Increasing kernal 2 to 3",
        "CaseNum":4,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":3,
        "kernelSize2":3,
        "batchSize":64,
        "epochSize":200
    },{
        "Case":"Increasing kernal 3 to 4",
        "CaseNum":1,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":4,
        "kernelSize2":4,
        "batchSize":64,
        "epochSize":50
    },{
        "Case":"Increasing kernal 3 to 4",
        "CaseNum":2,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":4,
        "kernelSize2":4,
        "batchSize":64,
        "epochSize":100
    },{
        "Case":"Increasing kernal 3 to 4",
        "CaseNum":3,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":4,
        "kernelSize2":4,
        "batchSize":64,
        "epochSize":150
    },{
        "Case":"Increasing kernal 3 to 4",
        "CaseNum":4,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":4,
        "kernelSize2":4,
        "batchSize":64,
        "epochSize":200
    },{
        "Case":"Increasing kernal 4 to 5",
        "CaseNum":1,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":5,
        "kernelSize2":5,
        "batchSize":64,
        "epochSize":50
    },{
        "Case":"Increasing kernal 4 to 5",
        "CaseNum":2,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":5,
        "kernelSize2":5,
        "batchSize":64,
        "epochSize":100
    },{
        "Case":"Increasing kernal 4 to 5",
        "CaseNum":3,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":5,
        "kernelSize2":5,
        "batchSize":64,
        "epochSize":150
    },{
        "Case":"Increasing kernal 4 to 5",
        "CaseNum":4,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":5,
        "kernelSize2":5,
        "batchSize":64,
        "epochSize":200
    },{
        "Case":"Increasing kernal 5 to 6",
        "CaseNum":1,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":6,
        "kernelSize2":6,
        "batchSize":64,
        "epochSize":50
    },{
        "Case":"Increasing kernal 5 to 6",
        "CaseNum":2,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":6,
        "kernelSize2":6,
        "batchSize":64,
        "epochSize":100
    },{
        "Case":"Increasing kernal 5 to 6",
        "CaseNum":3,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":6,
        "kernelSize2":6,
        "batchSize":64,
        "epochSize":150
    },{
        "Case":"Increasing kernal 5 to 6",
        "CaseNum":4,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":6,
        "kernelSize2":6,
        "batchSize":64,
        "epochSize":200
    },{
        "Case":"Increasing kernal 6 to 12",
        "CaseNum":1,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":12,
        "kernelSize2":12,
        "batchSize":64,
        "epochSize":50
    },{
        "Case":"Increasing kernal 6 to 12",
        "CaseNum":2,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":12,
        "kernelSize2":12,
        "batchSize":64,
        "epochSize":100
    },{
        "Case":"Increasing kernal 6 to 12",
        "CaseNum":3,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":12,
        "kernelSize2":12,
        "batchSize":64,
        "epochSize":150
    },{
        "Case":"Increasing kernal 6 to 12",
        "CaseNum":4,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":12,
        "kernelSize2":12,
        "batchSize":64,
        "epochSize":200
    },{
        "Case":"Increasing kernal 12 to 24",
        "CaseNum":1,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":24,
        "kernelSize2":24,
        "batchSize":64,
        "epochSize":50
    },{
        "Case":"Increasing kernal 12 to 24",
        "CaseNum":2,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":24,
        "kernelSize2":24,
        "batchSize":64,
        "epochSize":100
    },{
        "Case":"Increasing kernal 12 to 24",
        "CaseNum":3,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":24,
        "kernelSize2":24,
        "batchSize":64,
        "epochSize":150
    },{
        "Case":"Increasing kernal 12 to 24",
        "CaseNum":4,
        "filterSize1":24,
        "filterSize2":24,
        "kernelSize1":24,
        "kernelSize2":24,
        "batchSize":64,
        "epochSize":200
    }
]

TEST_CASES_CNN_LSTM_IMF_GRAPH = [

   ###############################
    {
        "Case": "CNN", "CaseNum": 1, "filterSize1": 48, "kernelSize1": 4, "kernelSize2": 24, "batchSize": 64,
        "epochSize": 50,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": "green", "model": "cnn"
    },
    {
        "Case": "CNN-LSTM 64", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 64,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 84", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 84,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 120", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 120,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
    {
        "Case": "CNN-LSTM 150", "CaseNum": 1, "filterSize1": 84, "kernelSize1": 4, "batchSize": 64, "epochSize": 50,
        "timestep": 120, "sample": 1, "lstmNodes": 150,
        "activation": "relu", "removeNoise": False, "addPool": False, "lstm": False, "color": get_random_color(1),
        "model": "cnn-lstm"
    },
]


def load_data(path, li):
    print("processing " + path)
    df = pd.DataFrame()
    if len(li) != 0:
        for index, chunk in enumerate(pd.read_csv(path, chunksize=1000000, sep=";", names=['num', 'type', 'post', 'xx', 'timestamp', 'watt']), start=1):
            df = df.append(chunk.loc[chunk['num'].isin(li)].loc[:, ['timestamp', 'watt']])

    return df

def calculate_mean(df1):
    return df1.groupby(['timestamp']).mean()

def calculate_sum(df1, date_string, season, year, days_of_week):

    if ENABLE_SEASON_SPLIT:
        if date_string:
            if int(date_string[3:5]) in [2,3,4]:
                season = 'spring'
            elif int(date_string[3:5]) in [5,6,7]:
                season = 'summer'
            elif int(date_string[3:5]) in [8,9,10]:
                season = 'fall'
            elif int(date_string[3:5]) in [11,12,1]:
                season = 'winter'

    if int(date_string[6:11]) == 2019:
        year = YEAR
    else:
        year = YEAR2

    df1 = df1.set_index('timestamp')
    df1.index = pd.to_datetime(df1.index)
    xx = []
    df1 = df1[df1.index.year.isin(year)]

    if ENABLE_WEEKDAY_SPLIT:
        if datetime.datetime.strptime(date_string, "%d/%m/%Y").weekday() in [5,6]:
            days_of_week = [5,6]
        else:
            days_of_week = [0, 1, 2, 3, 4]
    else:
        days_of_week = [0,1,2,3,4,5,6]


    if 'spring' in season:
        month = df1[df1.index.month.isin([2, 3, 4])]
        day = month[month.index.dayofweek.isin(days_of_week)]
        xx = xx + [day.loc[:, 'watt']]
        # xx = xx + [df1[df1.index.dayofweek  == any(days_of_week) and (df1.index.month == any([2, 3, 4]))].loc[year].loc[:, 'watt']]

    if 'summer' in season:
        month = df1[df1.index.month.isin([5,6,7])]
        day =  month[month.index.dayofweek.isin(days_of_week)]
        xx = xx + [day.loc[:, 'watt']]

    if 'fall' in season:
        month = df1[df1.index.month.isin([8,9,10])]
        day =  month[month.index.dayofweek.isin(days_of_week)]
        xx = xx + [day.loc[:, 'watt']]

    if 'winter' in season:
        month = df1[df1.index.month.isin([11,12,1])]
        day = month[month.index.dayofweek.isin(days_of_week)]
        xx = xx + [day.loc[:, 'watt']]

    xx_concat = pd.concat(xx)

    sumxx = xx_concat.groupby(xx_concat.index).sum()

    v = sumxx[sumxx <= 0].index
    v2 = v - datetime.timedelta(days=1)
    sumxx[v] = sumxx[v2]

    return sumxx

def calculate_hour_mean(df1, date_string, season, year, days_of_week):

    hours = []
    if ENABLE_SEASON_SPLIT:
        if date_string:
            if int(date_string[3:5]) in [2,3,4]:
                season = 'spring'
            elif int(date_string[3:5]) in [5,6,7]:
                season = 'summer'
            elif int(date_string[3:5]) in [8,9,10]:
                season = 'fall'
            elif int(date_string[3:5]) in [11,12,1]:
                season = 'winter'

    if int(date_string[6:11]) == 2019:
        year = YEAR
    else:
        year = YEAR2

    df1 = df1.set_index('timestamp')
    df1.index = pd.to_datetime(df1.index)
    xx = []
    df1 = df1[df1.index.year.isin(year)]
    date = datetime.datetime.strptime(date_string, "%d/%m/%Y")
    df1 = df1[df1.index < date]

    if ENABLE_WEEKDAY_SPLIT:
        if date.weekday() in [5,6]:
            days_of_week = [5,6]
        else:
            days_of_week = [0, 1, 2, 3, 4]
    else:
        days_of_week = [0,1,2,3,4,5,6]


    if 'spring' in season:
        month = df1[df1.index.month.isin([2, 3, 4])]
        day = month[month.index.dayofweek.isin(days_of_week)]

    if 'summer' in season:
        month = df1[df1.index.month.isin([5,6,7])]
        day =  month[month.index.dayofweek.isin(days_of_week)]

    if 'fall' in season:
        month = df1[df1.index.month.isin([8,9,10])]
        day =  month[month.index.dayofweek.isin(days_of_week)]

    if 'winter' in season:
        month = df1[df1.index.month.isin([11,12,1])]
        day = month[month.index.dayofweek.isin(days_of_week)]

    for hour in range(0, 24):
        hour_mean = day[day.index.hour == hour]
        xx = hour_mean.loc[:, 'watt']
        hours = hours + [xx.groupby(xx.index.hour).mean()._values[0]]


    return hours

def process_imf(df1_sum, process=False, save=True, test_data=True):

    #test_date = datetime.datetime.strptime(test_date_string, "%d/%m/%Y")
    #train = df1_sum[df1_sum.index < test_date]

    data = []

    time_vect = range(0, len(df1_sum))
    x = df1_sum

    #   test = df1_sum[df1_sum.index >= test_date]
    #   test = test[test.index < test_date + datetime.timedelta(days=1)]


    # if not process and os.path.isfile('C:\\Users\\Warren\\Desktop\\imf0.csv'):
    #     print('Reading IMFs 1,2,3,4 from files...')
    #     imf0 = np.genfromtxt(DATA_FOLDER + "imf0.csv", delimiter=',')
    #     imf1 = np.genfromtxt(DATA_FOLDER + "imf1.csv", delimiter=',')
    #     imf2 = np.genfromtxt(DATA_FOLDER + "imf2.csv", delimiter=',')
    #     imf3 = np.genfromtxt(DATA_FOLDER + "imf3.csv", delimiter=',')
    #     imf4 = np.genfromtxt(DATA_FOLDER + "imf4.csv", delimiter=',')
    #     imf5 = np.genfromtxt(DATA_FOLDER + "imf5.csv", delimiter=',')
    #
    # else:
    print('Generating IMFs 1,2,3,4,5 and storing to files...')

    #imf = emd.sift.sift(x, 0.00005, max_imfs=20)
    emd = EMD()
    emd.FIXE = 24 if test_data else 24
    imfs = emd(x)
    imfs, res = emd.get_imfs_and_residue()

    # vis = Visualisation()
    # vis.plot_imfs(imfs=imfs, residue=res, t=time_vect, include_residue=True)
    # vis.plot_instant_freq(time_vect, imfs=imfs)
    # vis.show()

    # components = EEMD()
    # components.FIXE = 2
    # components(x)
    #
    #
    # imfs, res = components[:-1], components[-1]
    #
    # vis = Visualisation()
    # vis.plot_imfs(imfs=imfs, residue=res, t=time_vect, include_residue=True)
    # vis.plot_instant_freq(time_vect, imfs=imfs)
    # vis.show()

    xx = np.delete(imfs, [0], axis=0).sum(axis=0)

    if IMF_GRAPH:
        import matplotlib.pyplot as plt

        print('Generating IMF Graph...')
        plt.figure( figsize=(16,8) )
        plt.plot(time_vect, xx, 'b', label="Denoised")
        plt.plot(time_vect, x, 'r', label="Original")
        plt.legend(loc='best')
        plt.show()

    return xx

def visualize_load_profile_by_season(df1, all_years):
    df1 = calculate_sum(df1, None, season=SEASON, year=YEAR, days_of_week=DAYS_OF_WEEK)

    plt.figure(figsize=(12, 8))

    plt.subplot(311, frameon=False)
    fig, axis = plt.subplots(4, frameon=False)

    for season in range(0, 4):
        year_list = df1.index.strftime("%Y").unique().tolist()
        print(year_list)


        for year in year_list:
            if season == 0:
                start_date = datetime.datetime.strptime("01/02/{}".format(year), "%d/%m/%Y")
                end_date = datetime.datetime.strptime("01/05/{}".format(year), "%d/%m/%Y")
            elif season == 1:
                start_date = datetime.datetime.strptime("01/05/{}".format(year), "%d/%m/%Y")
                end_date = datetime.datetime.strptime("01/08/{}".format(year), "%d/%m/%Y")
            elif season == 2:
                start_date = datetime.datetime.strptime("01/08/{}".format(year), "%d/%m/%Y")
                end_date = datetime.datetime.strptime("01/11/{}".format(year), "%d/%m/%Y")
            else:
                start_date = datetime.datetime.strptime("01/11/{}".format(int(year)-1), "%d/%m/%Y")
                end_date = datetime.datetime.strptime("01/02/{}".format(year), "%d/%m/%Y")

            train = df1[df1.index >= start_date]
            train = train[train.index < end_date]

            train = pd.DataFrame({'watt': train.values}, index=train.index)
            axis[season].plot(range(0, len(train)), train['watt'], label='{}'.format(year))
            #axis[season].title('{} - {}'.format(str(start_date), str(end_date)))

    plt.legend(loc='best')
    plt.show()

def visualize_load_profile_by_year(df1, date_start, date_end, all_years):
    plt.figure(figsize=(12, 8))

    if all_years:
        year_list = df1.index.strftime("%Y").unique().tolist()
        print(year_list)


        for year in year_list:
            if not date_start.endswith("}"):
                date_start = date_start[:-4] + '{}'
            if not date_end.endswith("}"):
                date_end = date_end[:-4] + '{}'

            start_date_string = date_start.format(year)
            end_date_string = date_end.format(year)

            start_date = datetime.datetime.strptime(start_date_string, "%d/%m/%Y")
            end_date = datetime.datetime.strptime(end_date_string, "%d/%m/%Y")

            train = df1[df1.index >= start_date]
            train = train[train.index < end_date]

            train = pd.DataFrame({'watt': train.values}, index=train.index)
            plt.plot(range(0, len(train)), train['watt'], label='{} - {}'.format(start_date_string, end_date_string))

    else:
        start_date = datetime.datetime.strptime(date_start, "%d/%m/%Y")
        end_date = datetime.datetime.strptime(date_end, "%d/%m/%Y")

        train = df1[df1.index >= start_date]
        train = train[train.index < end_date]

        train = pd.DataFrame({'watt': train.values}, index=train.index)
        plt.plot(train.index, train['watt'], label='{} - {}'.format(start_date_string, end_date_string))

    plt.legend(loc='best')
    plt.title("{} - {}".format(date_start, date_end))
    plt.show()

def visualize_days(df1, dates, title):
    plt.figure(figsize=(12, 8))

    for date_string in dates:
        try:
            df1_sum = calculate_sum(df1, date_string, season=SEASON, year=YEAR, days_of_week=DAYS_OF_WEEK)

            date = datetime.datetime.strptime(date_string, "%d/%m/%Y")

            train = df1_sum[df1_sum.index >= date]
            train = train[train.index < date+datetime.timedelta(days=1)]
            train = pd.DataFrame({'watt': train.values}, index=train.index)
            plt.plot(list(range(0,24)), train['watt'], label='{}'.format(date_string))

        except Exception:
            print('Error {}'.format(date_string))

    plt.legend(loc='best')
    plt.title(title)
    plt.ylabel('Consumption (kWh)')
    plt.xlabel('Hour')
    plt.show()

def visualize_date_against_average(df1, dates, title):
    plt.figure(figsize=(12, 8))
    days_of_week = [0,1,2,3,4,5,6]
    fig, axis = plt.subplots(len(dates), frameon=False)
    count = 0

    for date_string in dates:

        hours = []
        xx = []
        try:
            df1_sum = calculate_sum(df1, date_string, season=SEASON, year=YEAR, days_of_week=DAYS_OF_WEEK)
            hours = calculate_hour_mean(df1, date_string, season=SEASON, year=YEAR, days_of_week=DAYS_OF_WEEK)
            date = datetime.datetime.strptime(date_string, "%d/%m/%Y")

            train = df1_sum[df1_sum.index >= date]
            train = train[train.index < date + datetime.timedelta(days=1)]
            train = pd.DataFrame({'watt': train.values}, index=train.index)


            axis[count].plot(list(range(0, 24)), train['watt'], label='{}'.format(date_string))
            axis[count].plot(list(range(0,24)), hours, label='Average')
            axis[count].legend(loc='best')
            count+=1
            print('average consumption during season {}: {}'.format(date_string, list(train['watt']._values)))
            print('average consumption on date {}: {}'.format(date_string, hours))
        except Exception:
            print('Error {}'.format(date_string))


    plt.title(title)
    plt.ylabel('Consumption (kWh)')
    plt.xlabel('Hour')
    plt.show()

def visualize_season(df1, date_string):
    plt.figure(figsize=(12, 8))

    try:
        #df1 = calculate_sum(df1, date_string, season=SEASON, year=YEAR, days_of_week=DAYS_OF_WEEK)
        date = datetime.datetime.strptime(date_string, "%d/%m/%Y")

        train = df1[df1.index <= date]
        train = train[train.index.year.isin(YEAR)]
        train = pd.DataFrame({'watt': train.values}, index=train.index)
        plt.plot(list(range(0, len(train))), train['watt'], label='{}'.format(date_string))

        plt.legend(loc='best')
        plt.title("Season_{}".format(date_string))
        plt.ylabel('Consumption (kWh)')
        plt.xlabel('Hour')
        plt.show()
    except Exception:
        print('Error {}'.format(date_string))

def visualize_month_wise_box_plot(df1, dates, title):
    plt.figure(figsize=(12, 8))

    for date_string in dates:
        if date_string:
            if int(date_string[3:5]) in [2, 3, 4]:
                season = 'spring'
            elif int(date_string[3:5]) in [5, 6, 7]:
                season = 'summer'
            elif int(date_string[3:5]) in [8, 9, 10]:
                season = 'fall'
            elif int(date_string[3:5]) in [11, 12, 1]:
                season = 'winter'

        try:
            df1_sum = calculate_sum(df1, date_string, season=SEASON, year=YEAR, days_of_week=DAYS_OF_WEEK)

            date = datetime.datetime.strptime(date_string, "%d/%m/%Y")

            train = df1_sum[df1_sum.index <= date]

            # Calculate ACF and PACF upto 50 lags
            # acf_50 = acf(train.values, nlags=24)
            # pacf_50 = pacf(train.values, nlags=24)

            # Draw Plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 3), dpi=100)
            plot_acf(train.values, ax=axes[0])
            plot_pacf(train.values, ax=axes[1])
            plt.show()

            diff = list()
            days_in_year = 365
            for i in range(0, len(train.values)):
                value = train.values[i] - train.values[i - 24]
                diff.append(value)
            plt.plot(diff)
            plt.show()

            #train = train[train.index < date+datetime.timedelta(days=1)]
            train = pd.DataFrame({'watt': train.values}, index=train.index)
            plt.hist(diff)
            plt.show()



            #
            # from statsmodels.tsa.seasonal import seasonal_decompose
            # decomposition = seasonal_decompose(train)
            # fig = decomposition.plot()
            # plt.show()
            #
            # decomposition = seasonal_decompose(diff)
            # fig = decomposition.plot()
            # plt.show()

            train['year'] = [d.year for d in train.index]
            train['month'] = [d.month for d in train.index]
            train['hour'] = [d.hour for d in train.index]
            #train['day'] = [d.dayofyear for d in train.index]
            years = train['year'].unique()

            fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=80)
            sns.boxplot(x='year', y='watt', data=train, ax=axes[0])
            sns.boxplot(x='month', y='watt', data=train.loc[~train.year.isin([1991, 2020]), :])

            # Set Title
            axes[0].set_title('Year-wise Box Plot\n', fontsize=18);
            axes[1].set_title('Month-wise Box Plot\n{}'.format(''), fontsize=18)
            plt.show()

        except Exception:
            print('Error {}'.format(date_string))

def store_sliding_window(path, df1, date_string, start, end):
    print("calculating and saving moving average for {} -- {}".format(date_string, path))
    test_date = datetime.datetime.strptime(date_string, "%d/%m/%Y")
    train = df1[df1.index < test_date]
    #train = train[train.index.dayofyear < (TEST_DAY_OF_YEAR[-1] - 7)]
    #test = df1[df1.index.year == 2017]
    #test = test[test.index.dayofyear == TEST_DAY_OF_YEAR] # predict this year

    # get test data
    end = len(train)

    new_start = start
    new_end = start + ((24*DAYS)*LOOKBACK) + 24
    data_end = end
    SHIFT = 1
    #df2 = pd.DataFrame()


    while (new_start + ((24*DAYS)*LOOKBACK + 24) ) < data_end:
        dataframe = pd.DataFrame(data=train.loc[:]._values[new_start:new_end], columns=["watt"]).transpose()
        dataframe.to_csv(path, mode='a', header=False)
        new_start+=SHIFT
        new_end+=SHIFT

    #df2.to_csv(MOVING_AVG_CSV, mode='a', header=False)

def naive(df1, days_of_week):

    # get train data
    month = df1[df1.index.month.isin([5, 6])]
    day = month[month.index.dayofweek.isin(days_of_week)]
    train = day

    # get test data
    month = df1[df1.index.month.isin([7])]
    day = month[month.index.dayofweek.isin(days_of_week)]
    test = day

    # get train data
    #week = df1[df1.index.df.week.isin(TRAIN_DAY_OF_YEAR)]
    #day = week[week.index.dayofweek.isin(days_of_week)]
    train =  df1[df1.index.dayofweek.isin(TRAIN_DAY_OF_WEEK)]

    # get test data
    #week = df1[df1.index.df.dayofyear.isin(TEST_DAY_OF_YEAR)]
    #day = week[week.index.dayofweek.isin(days_of_week)]
    test = df1[df1.index.dayofyear.isin(TEST_DAY_OF_YEAR)]




    if test_christmas:
        latest_year_to_test = YEAR[-1]
        date_string = "25/12/{}".format(latest_year_to_test)
        christmas_date = datetime.datetime.strptime(date_string, "%d/%m/%Y")
        train_day_of_week = christmas_date.weekday()
        christmas_day_of_year = christmas_date.timetuple().tm_yday

        train = df1[df1.index.dayofweek.isin(TRAIN_DAY_OF_WEEK)]
        train = train[train.index.dayofyear < christmas_day_of_year]
        test =  df1[df1.index.year == latest_year_to_test]
        test = test[test.index.dayofyear == christmas_day_of_year]

    else:

        # ------------------------------------------------------------
        # get train data
        # week = df1[df1.index.df.week.isin(TRAIN_DAY_OF_YEAR)]

        # train = df1[df1.index.year == 2017]
        # train = train[train.index.dayofweek.isin(TRAIN_DAY_OF_YEAR)]
        # # train =  df1[df1.index.dayofyear.isin(TRAIN_DAY_OF_YEAR)]
        #
        # # get test data
        # # week = df1[df1.index.df.dayofyear.isin(TEST_DAY_OF_YEAR)]
        # # day = week[week.index.dayofweek.isin(days_of_week)]
        # last_day_of_train = train.index[-1].dayofyear
        # train = train[train.index.dayofyear != last_day_of_train]
        # test_day = last_day_of_train + 7
        # test = df1[df1.index.year == train.index[-1].year]
        # test = test[test.index.dayofyear == last_day_of_train]  # predict the last day of year's friday

        # ------------------------------------------------------------
        # get train data
        # week = df1[df1.index.df.week.isin(TRAIN_DAY_OF_YEAR)]
        test_date = datetime.datetime.strptime(test_date_string, "%d/%m/%Y")
        train = df1[df1.index < test_date]
        test = df1[df1.index >= test_date]
        test = test[test.index < test_date+datetime.timedelta(days=1)]

    train = pd.DataFrame({'watt': train.values}, index=train.index)
    test = pd.DataFrame({'watt': test.values}, index=test.index)
    dd = np.asarray(train.watt)
    y_hat = test.copy()
    y_hat['naive'] = dd[len(dd) - 25:len(dd) - 1]
    plt.figure(figsize=(12, 8))
    plt.plot(train.index, train['watt'], label='Train')
    plt.plot(test.index, test['watt'], label='Test')
    plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
    plt.legend(loc='best')
    plt.title("Naive Forecast")
    plt.show()

    mse = mean_squared_error(y_hat.watt, y_hat.naive)
    print('Naive MSE: %f' % mse)
    mae = mean_absolute_error(y_hat.watt, y_hat.naive)
    print('Naive MAE: %f' % mae)

def arima(df1, days_of_week):
    # get train data
    # month = df1[df1.index.month.isin([5, 6])]
    # day = month[month.index.dayofweek.isin(days_of_week)]
    # train = day
    #
    # # get test data
    # month = df1[df1.index.month.isin([7])]
    # day = month[month.index.dayofweek.isin(days_of_week)]
    # test = day

    # train.plot(figsize=(15,8), title= 'Watts', fontsize=14)
    # test.plot(figsize=(15,8), title= 'Watts', fontsize=14)
    # plt.show()
    # get train data
    #week = df1[df1.index.df.week.isin(TRAIN_DAY_OF_YEAR)]
    #day = week[week.index.dayofweek.isin(days_of_week)]
    train = df1[df1.index.dayofweek.isin(TRAIN_DAY_OF_WEEK)]
    # train =  df1[df1.index.dayofyear.isin(TRAIN_DAY_OF_YEAR)]

    # get test data
    # week = df1[df1.index.df.dayofyear.isin(TEST_DAY_OF_YEAR)]
    # day = week[week.index.dayofweek.isin(days_of_week)]
    last_day_of_train = train.index[-1].dayofyear
    train = train[train.index.dayofyear != last_day_of_train]
    test_day = last_day_of_train + 7
    test = df1[df1.index.dayofyear == last_day_of_train]  # predict the last day of year's friday

    train = pd.DataFrame({'watt': train.values}, index=train.index)
    test = pd.DataFrame({'watt': test.values}, index=test.index)
    dd = np.asarray(train.watt)
    y_hat = test.copy()
    fit1 = sm.tsa.statespace.SARIMAX(train.watt, order=(2, 1, 4), seasonal_order=(0, 1, 1, 24)).fit()
    #y_hat['arima'] = fit1.predict(start="2015-06-13", end="2015-06-14", dynamic=True)
    y_hat['arima'] = fit1.predict(start=test.index[0], end=test.index[-1], dynamic=True)

    plt.figure(figsize=(12, 8))
    plt.plot(train.index, train['watt'], label='Train')
    plt.plot(test.index, test['watt'], label='Test')
    plt.plot(y_hat.index, y_hat['arima'], label='ARIMA')
    plt.legend(loc='best')
    plt.title("ARIMA Forecast")
    plt.show()

    mse = mean_squared_error(y_hat.watt, y_hat.arima)
    print('ARIMA MSE: %f' % mse)
    mae = mean_absolute_error(y_hat.watt, y_hat.arima)
    print('ARIMA MAE: %f' % mae)

    #
    # sm.tsa.seasonal_decom
    # pose(train.watt).plot()
    # result = sm.tsa.stattools.adfuller(train.watt)
    # plt.show()

def load_sliding_window_for_convolution(path):
    df = pd.DataFrame()
    if os.path.isfile(path):

        print("loading sliding window file")
        for index, chunk in enumerate(pd.read_csv(path, chunksize=1000000, sep=",", names=range(1,((24*DAYS)*LOOKBACK  + 24) + 1)),
                                      start=1):
            df = df.append(chunk.loc[:,1:((24*DAYS)* LOOKBACK + 24)])

    print("DONE loading sliding window file")
    return df

def process_convolution_network(sliding_window, df1, num_of_itr, noise_removed, test_date_string):

    test_epochs = []
    plt.title("{}".format(test_date_string.replace("/", "_")))
    fig, axis = plt.subplots(2, frameon=False)
    case_num = 0
    orig_sliding_window = sliding_window
    reuse = False
    imf_input = None
    for testCase in TEST_CASES_CNN_LSTM_TIMESTEPS:

        itr = 0
        total_conv_mae = 0
        total_conv_mse = 0
        avg_train_mae = 0
        avg_test_mae = 0
        y_hat_list = []
        test_color = testCase["color"]

        print("\n\nRunning Test Case {}, Num {}\n".format(testCase['Case'], testCase['CaseNum']))

        while itr < num_of_itr:

            tf.keras.backend.clear_session()
            K.clear_session()

            model = tf.keras.Sequential()
            print('running cov num {}...'.format(itr))

            test_date = datetime.datetime.strptime(test_date_string, "%d/%m/%Y")
            train2 = df1[df1.index < test_date]
            train2 = train2[len(train2) - (24*DAYS*LOOKBACK):len(train2)]

            test = df1[df1.index >= test_date]
            test = test[test.index < test_date + datetime.timedelta(days=1)]

            from sklearn.preprocessing import MinMaxScaler

            if testCase['removeNoise']:
               sliding_window = noise_removed
            else:
                sliding_window = orig_sliding_window

            items = sliding_window[(sliding_window < 0).any(axis=1)]

            dataset = sliding_window.loc[:,:]._values
            #dataset = np.reshape(dataset, (-1,1))
            scaler = MinMaxScaler(feature_range=(0,1))
            dataset = scaler.fit_transform(dataset)
            X = dataset[:, 0:(24*DAYS)*LOOKBACK]
            Y = dataset[:, len(dataset[0]) - 24:len(dataset[0])]

            if testCase['model'] == 'cnn':
                X = np.expand_dims(X, axis=2)
                model.add(keras.layers.Input(shape=(24*DAYS*LOOKBACK, 1)))
                model.add(keras.layers.Conv1D(filters=testCase['filterSize1'], kernel_size=testCase['kernelSize1'], activation=testCase['activation'], padding="causal"))
                if testCase['addPool']:
                    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
               # model.add(keras.layers.Conv1D(filters=testCase['filterSize2'], kernel_size=testCase['kernelSize2'], activation=testCase['activation'], padding="causal"))
             #   if testCase['addPool']:
             #       model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
                model.add(keras.layers.Flatten())
                model.add(keras.layers.Dense(24))

            elif testCase['model'] == 'cnn-lstm':
                X = X.reshape((X.shape[0], testCase["sample"], testCase["timestep"],  1))
                model.add(keras.layers.Input(shape=(testCase["sample"], testCase["timestep"], 1)))
                model.add(keras.layers.TimeDistributed(keras.layers.Conv1D(filters=testCase['filterSize1'], kernel_size=testCase['kernelSize1'], activation=testCase['activation'], padding="causal"))) #144
                if testCase['addPool']:
                    model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling1D(pool_size=2, strides=1))) # 140/2 = 70
                #model.add(keras.layers.TimeDistributed(keras.layers.Conv1D(filters=testCase['filterSize2'], kernel_size=testCase['kernelSize2'], activation=testCase['activation'], padding="causal"))) #144
                model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
                #model.add(keras.layers.LSTM(testCase['lstmNodes'], return_sequences=True, activation='sigmoid'))
               # model.add(keras.layers.Dropout(0.2))
 #               model.add(keras.layers.LSTM(testCase['lstmNodes'], return_sequences=True, activation='tanh'))

                model.add(keras.layers.LSTM(testCase['lstmNodes'], activation='tanh'))
                model.add(keras.layers.Dense(24))

            elif testCase['model'] == 'lstm':
                X = X.reshape((X.shape[0], X.shape[1], 1))
                model.add(keras.layers.Input(shape=(X.shape[1], 1)))
                model.add(keras.layers.LSTM(12, activation='relu'))
                model.add(keras.layers.Dense(24))


            es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, verbose=0)
#            mc = ModelCheckpoint('best_model_{}_{}.h5'.format(testCase["model"], testCase["removeNoise"]), monitor='val_loss', mode='min', save_best_only=True,  verbose=0)

            opt = keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=opt, loss='mse', metrics=['accuracy', 'mse', 'mae'])

            tensorboard = TensorBoard(
                log_dir='.\logs',
                histogram_freq=1,
                write_images=True
            )

            keras_callbacks = [
                es
            ]

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                                test_size=0.2,
                                                                shuffle=False,
                                                                stratify = None
                                                                )  # split the data: 80% training, 20% test

            history = model.fit(
                X_train, Y_train, validation_data=(X_test, Y_test), epochs=testCase['epochSize'], batch_size=testCase['batchSize'], verbose=0,
                                callbacks=keras_callbacks
                                )


            #plt.plot(history.history['mae'])
            #plt.plot(history.history['val_mae'])
            # plt.plot(history.history['mse'])
            # plt.plot(history.history['val_mse'])
            # plt.title('Model MAE')
            # plt.ylabel('MAE')
            # plt.xlabel('epoch')
            # plt.legend(['train', 'validation'], loc='upper left')
            # plt.show()


            #model = tf.keras.models.load_model('best_model_{}_{}.h5'.format(testCase["model"], testCase["removeNoise"]))
            #tt = model.evaluate(X_test, Y_test, batch_size=10, verbose=0)
            #print("TRAIN MAE: {}".format(history.history['mae'][-1]))
            #print("VAL MAE: {}".format(history.history['val_mae'][-1]))
            # plt.plot(tt.history['mae'])
            # plt.plot(tt.history['val_mae'])
            # plt.title('Model MAE')
            # plt.ylabel('MAE')
            # plt.xlabel('epoch')
            # plt.legend(['train', 'validation'], loc='upper left')
            # plt.show()

            # demonstrate prediction
            train = pd.DataFrame({'watt': train2.values}, index=train2.index)
            test = pd.DataFrame({'watt': test.values}, index=test.index)
            x_input = train._values

            if testCase['removeNoise']:
                if not reuse:
                    imf_input = process_imf(train2, test_data=False)
                    reuse = True

                x_input = imf_input

            #scaler = MinMaxScaler(feature_range=(0, 1))

            x_input = scaler.fit_transform(x_input.reshape(-1,1))
            x_input = x_input[0:24*DAYS*LOOKBACK]

            #x_input = scaler.transform(x_input)
            #x_input[len(x_input)-24:len(x_input)] = [10] * 24
            #x_input = np.expand_dims(x_input, axis=2)
            if testCase['model'] == 'cnn' or testCase['model'] == 'emd-cnn':
                x_input = x_input.reshape((1, 24*DAYS*LOOKBACK, 1))
            elif testCase['model'] == 'cnn-lstm' or testCase['model'] == 'emd-cnn-lstm':
                x_input = x_input.reshape((1, testCase["sample"], testCase["timestep"], 1))
            elif testCase['model'] == 'lstm':
                x_input = x_input.reshape((1, X.shape[1], 1))

            yhat = model.predict(x_input, verbose=0)
            yhat = scaler.inverse_transform(yhat)
            #
            #print(yhat)

            #train = pd.DataFrame({'watt': train.values}, index=train.index)
            #test = pd.DataFrame({'watt': test.values}, index=test.index)
            y_hat = test.copy()
            y_hat['cnn'] = yhat[0]
            y_hat_list.append(yhat[0])
            # plt.figure(figsize=(12, 8))
            # plt.plot(train.index, train['watt'], label='Train')
            # plt.plot(test.index, test['watt'], label='Test')
            # plt.plot(y_hat.index, y_hat['cnn'], label='CNN Forecast')
            # plt.legend(loc='best')
            # plt.title("CNN Forecast")
            # plt.show()
            avg_train_mae+=history.history['mae'][-1]
            avg_test_mae+=history.history['val_mae'][-1]
            total_conv_mae+=(sum(abs(test['watt'] - y_hat['cnn'])) / 24)
            total_conv_mse+=(sum(pow(test['watt'] - y_hat['cnn'], 2)) / 24)

            #
            #
            # # print(model.summary())
            # index = 0
            # found = False;
            # for item in test_epochs:
            #     if item['testcase'] == testCase["Case"]:
            #         test_epochs[index]['mse_train'] = test_epochs[index]['mse_train'] + history.history['loss']
            #         test_epochs[index]['mse_val'] = test_epochs[index]['mse_val'] + history.history['val_loss']
            #         test_epochs[index]['predicted'] = test_epochs[index]['predicted'] + y_hat['cnn']
            #         test_epochs[index]['predicted_mae'] = test_epochs[index]['predicted_mae'] + (sum(abs(test['watt'] - y_hat['cnn'])) / 24)
            #         found = True
            #         break
            #     index = index + 1
            #
            # if not found:
            #     test_epochs.append({
            #         "testcase": testCase["Case"],
            #         "mse_train": history.history['mse'],
            #         "mse_val": history.history['val_mse'],
            #         "train": train['watt'],
            #         "test": test['watt'],
            #         "predicted": y_hat['cnn'],
            #         "predicted_mae": sum(abs(test['watt'] - y_hat['cnn'])) / 24,
            #         "color": testCase["color"]
            #     })

            itr+=1

        print("{} -> {}: {} ".format(test_date_string, testCase["Case"], total_conv_mse/NUM_OF_ITR))
        axis[0].barh(testCase["Case"], total_conv_mse/NUM_OF_ITR, color=test_color)
        axis[0].text(total_conv_mse/NUM_OF_ITR +.25, case_num, str(round(total_conv_mse/NUM_OF_ITR, 2)), color='blue')
        case_num += 1
        xx = [0]*24
        for yhat in y_hat_list:
            xx = xx+yhat

        xx = xx/NUM_OF_ITR

        axis[1].plot(train.index, train['watt'], label='Train', color="blue")
        axis[1].plot(test.index, test['watt'], label='Test', color="blue")
        axis[1].plot(y_hat.index, xx, label='CNN Forecast', color=test_color)

        #
        # print("Case Name: {}\n"
        #       "Case Num: {}\n"
        #       "Average MAE: {}\n"
        #       "Average Train MAE: {}\n"
        #       "Average Test MAE: {}\n".format(
        #     testCase['Case'],
        #     testCase['CaseNum'],
        #     round((total_conv_mae / num_of_itr),2),
        #     round((avg_train_mae / num_of_itr),2),
        #     round((avg_test_mae / num_of_itr),2)))
    #plt.show()
    if not os.path.isdir(DATA_FOLDER_FIGURES + "layer1\\LB_{}\\{}\\{}\\".format(LOOKBACK, "Seasonal" if ENABLE_SEASON_SPLIT else "AllSeason", "Weekday" if ENABLE_WEEKDAY_SPLIT else "Weekday_Weekend")):
        os.makedirs(DATA_FOLDER_FIGURES + "layer1\\LB_{}\\{}\\{}\\".format(LOOKBACK, "Seasonal" if ENABLE_SEASON_SPLIT else "AllSeason", "Weekday" if ENABLE_WEEKDAY_SPLIT else "Weekday_Weekend"))

    #plt.figure(figsize=(16,8))
    plt.savefig(DATA_FOLDER_FIGURES + "layer1\\LB_{}\\{}\\{}\\{}_{}_timestep{}_itr{}_flt{}_{}.png".format(LOOKBACK, "Seasonal" if ENABLE_SEASON_SPLIT else "AllSeason", "Weekday" if ENABLE_WEEKDAY_SPLIT else "Weekday_Weekend", TEST_NUM,test_date_string.replace("/", "_"), "NA", testCase['filterSize1']
    ,NUM_OF_ITR, TEST_TYPE))

    # legend = []
    # colors = get_random_color(len(test_epochs))

    # for item in test_epochs:
    #     plt.plot([x/NUM_OF_ITR for x in item["mse_train"]], color=item["color"])
    #     plt.plot([x/NUM_OF_ITR for x in item["mse_val"]], color=get_random_color(1))
    #     legend.append(['{}_train'.format(item["testcase"])])
    #     legend.append(['{}_val'.format(item["testcase"])])

    # plt.title('Model MAE')
    # plt.ylabel('MAE')
    # plt.xlabel('epoch')
    # plt.legend(legend, loc='upper left')
    # plt.show()

    # legend = []
    #
    # plt.plot(item["train"].index, item["train"], 'b')
    # legend.append(['Train'])
    #
    # plt.plot(item["test"].index, item["test"], 'b')
    # legend.append(['Test'])
    #
    # for item in test_epochs:
    #     plt.plot(item["predicted"].index, [x/NUM_OF_ITR for x in item["predicted"]], color=item["color"])
    #     legend.append(['{}'.format(item["testcase"])])
    #
    # plt.title('Model MAE')
    # plt.ylabel('MAE')
    # plt.xlabel('epoch')
    # plt.legend(legend, loc='upper left')
    # plt.show()
    #
    # case = []
    # case_mae = []
    # colors = []
    #
    # for item in test_epochs:
    #     case.append('{}'.format(item["testcase"]))
    #     case_mae.append(item["predicted_mae"]/NUM_OF_ITR)
    #     colors.append(item["color"])
    #
    # plt.bar(case, case_mae, color=colors)
    # plt.show()


myset = set()
df1 = pd.DataFrame()

if os.path.isfile(PREMADE_CSV):
    print("loading premade file")
    for index, chunk in enumerate(pd.read_csv(PREMADE_CSV, chunksize=1000000, sep=",", names=['timestamp', 'watt']), start=1):
        df1 = df1.append(chunk.loc[:, ['timestamp', 'watt']])
    print("Done.")
else:
    print('reading...')
    dataxx = {0: [],
              1: [],
              2: [],
              3: [],
              4: []
    }

    random_houses = sorted(random.sample(list(range(HOUSE_NUMBER_START, HOUSE_NUMBER_END)), k=NUMBER_OF_HOUSE_SAMPLES))

    for random_house in random_houses:
        if 1 <= random_house <= 1000:
            dataxx[0] = dataxx.get(0) + [random_house]
        elif 1001 <= random_house <= 2000:
            dataxx[1] = dataxx.get(1) + [random_house]
        elif 2001 <= random_house <= 3000:
            dataxx[2] = dataxx.get(2) + [random_house]
        elif 3001 <= random_house <= 4000:
            dataxx[3] = dataxx.get(3) + [random_house]
        elif 4001 <= random_house <= 5000:
            dataxx[4] = dataxx.get(4) + [random_house]

    df1 = load_data(ORIG_DATA + "1-1000 målepunkter.csv", dataxx[0])
    print(df1)
    df1 = df1.append(load_data(ORIG_DATA + "1001-2000 målepunkter.csv", dataxx[1]))
    df1 = df1.append(load_data(ORIG_DATA + "2001-3000 målepunkter.csv", dataxx[2]))
    df1 = df1.append(load_data(ORIG_DATA + "3001-4000 målepunkter.csv", dataxx[3]))
    df1 = df1.append(load_data(ORIG_DATA + "4001-5000 målepunkter.csv", dataxx[4]))
    df1.to_csv(PREMADE_CSV, header=False)


#visualize_days(df1, holidays_dates_list, "Danish Holidays")
#visualize_date_against_average(df1, holidays_dates_list, "Danish Holidays")

#visualize_month_wise_box_plot(df1, season_dates_list, "Seasonal Days")
# for env_setup_case in env_setup_cases:
#     LOOKBACK = env_setup_case['lookback']
#     ENABLE_WEEKDAY_SPLIT = env_setup_case['weekday_split']
#     ENABLE_SEASON_SPLIT = env_setup_case['season_split']

DAYS = 5 if ENABLE_WEEKDAY_SPLIT else 7


if not (ENABLE_SEASON_SPLIT and ENABLE_WEEKDAY_SPLIT):
    season_dates_list = other_season_dates_list

for date_string in holidays_dates_list:
    imf_path = SW_IMF.format(date_string.replace("/", "_"), LOOKBACK,
                             "Seasonal" if ENABLE_SEASON_SPLIT else "AllSeason",
                             "Weekday" if ENABLE_WEEKDAY_SPLIT else "Weekday_Weekend")

    sliding_window_path = MOVING_AVG_CSV.format(date_string.replace("/", "_"), LOOKBACK,
                                                "Seasonal" if ENABLE_SEASON_SPLIT else "AllSeason",
                                                "Weekday" if ENABLE_WEEKDAY_SPLIT else "Weekday_Weekend")

    df1_sum = calculate_sum(df1, date_string, season=SEASON, year=YEAR, days_of_week=DAYS_OF_WEEK )
    #visualize_load_profile_by_season(df1_sum, all_years=True)
    #visualize_load_profile_by_year(df1_sum, date_start='15/04/{}', date_end='16/04/{}', all_years=True)

    #naive(df1_sum, days_of_week=DAYS_OF_WEEK)
    #simple_average(df1_sum, days_of_week=DAYS_OF_WEEK)
    #moving_average(df1_sum, days_of_week=DAYS_OF_WEEK)
    #simple_exponential_smoothing(df1_sum, days_of_week=DAYS_OF_WEEK)
    #holt_linear(df1_sum, days_of_week=DAYS_OF_WEEK)
    #holt_winter(df1_sum, days_of_week=DAYS_OF_WEEK)
    #arima(df1_sum, days_of_week=DAYS_OF_WEEK)

    start = 0
    end = len(df1_sum)

    #visualize_season(df1_sum, date_string)
    imfs = None
    if PRODUCE_IMFS:
        imfs = process_imf(df1_sum)
        if not os.path.isfile(imf_path):
            #imfs = process_imf(df1_sum)
            df = pd.DataFrame(data=imfs, columns=["watt"], index=df1_sum.index)
            store_sliding_window(imf_path, df, date_string, start, len(imfs))

    if not os.path.isfile(sliding_window_path):
        store_sliding_window(sliding_window_path, df1_sum, date_string, start, end)

    df_sliding_window = load_sliding_window_for_convolution(sliding_window_path)
    imf_sw = load_sliding_window_for_convolution(imf_path)

    total_conv_mae = 0
    print("SeasonalSplit: {}, WeekdaySplit: {}".format(ENABLE_SEASON_SPLIT, ENABLE_WEEKDAY_SPLIT))
    process_convolution_network(df_sliding_window, df1_sum, NUM_OF_ITR, imf_sw, date_string)


print("Done!")