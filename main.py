import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error

sns.set(rc={'figure.figsize': [15, 7]}, font_scale=1.2)
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')