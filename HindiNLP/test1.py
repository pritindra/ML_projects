# this is a test with a different dataset untill a 
# better dataset is gained

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#import tensorflow as tf

df = pd.read_csv("dataset/Hindi_bible_with_authors.csv")
df = df.drop("Unnamed: 0", axis=1)
print(df.head(5))


