import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import logging
import random
from ta import add_all_ta_features as talib

# Configure logging
logging.basicConfig(level=logging.INFO)

