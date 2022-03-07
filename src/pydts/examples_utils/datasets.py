import pandas as pd
from pydts.config import *

DATASETS_DIR = os.path.join(os.path.dirname((os.path.dirname(__file__))), 'datasets')

def load_LOS_simulated_data():
    os.path.join(os.path.dirname(__file__))
    return pd.read_csv(os.path.join(DATASETS_DIR, 'LOS_simulated_data.csv'))