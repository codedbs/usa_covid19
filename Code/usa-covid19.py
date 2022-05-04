# Import and Install required packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pyspark.sql.functions as F
import pyspark.sql.types as T


# Initializing PySpark

from pyspark import SparkContext
sc = SparkContext(appName='covid19anaysis')

from pyspark.sql import SQLContext
# initializing SQLContext
sqlContext = SQLContext(sc)