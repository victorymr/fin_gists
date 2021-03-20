## save global variables

import sys
import importlib
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.stats import lognorm
import random

from datetime import date
import pdb
import comp_data
from yfinance import yfinance as yf

Input_dict = {}
comp = yf.Ticker("MSFT")
