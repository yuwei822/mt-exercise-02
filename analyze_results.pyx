import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

dropout_rates = [0, 0.2, 0.4, 0.6, 0.8]
logs_folder = './logs'

def parse_log(dropout):
    logfile = os.path.join(logs_folder, f'dropout_{str(dropout).replace(".", "_")}.log')
    epochs, train_ppl, valid_ppl = [], [], []
    with open(logfile
