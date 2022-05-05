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

# load the data using pandas
pdf = pd.read_csv("../Data/us-counties.csv")
pdf.head(2)




# using createDataFrame we can covert pandas dataframe into pyspark
sdf = sqlContext.createDataFrame(pdf)
sdf.show(5)


#creating a temporary table to data manipulate

sdf.registerTempTable('covid_data')


# check the schema (columns & their data types)
sdf.printSchema()


# to print the number of rows in the dataset
sdf.count()



## some filters into the data

#  get the latest date available in the dataset
latest_date = sdf.agg(F.max('date').alias('max_data'))
latest_date.show()


# the latest date avilable is 2020-04-21
latest_date = latest_date.collect()
latest_date

latest_date = latest_date[0]['max_data']
latest_date

# filtering the data where this date is present
sdf_filtered = sdf.where(
    "date = '{}'".format(latest_date)
)

sdf_filtered.show(2)


## testing using sql

latest_date_sql = sqlContext.sql(
    """ SELECT * FROM covid_data WHERE date = '{}'""".format(latest_date)
)
latest_date_sql.show(10)


## Overall Statistics

overall_stats = sdf_filtered.agg(
    F.sum("cases").alias("total_cases"), # to sum the values in column 'cases'
    F.sum("deaths").alias("total_deaths"),
    F.count("*").alias("number_of_records"), # to count the number of records in the dataset
    F.countDistinct("county").alias("number_of_counties"), # to get the distinct count of counties in column 'county'
    F.countDistinct("state").alias("number_of_states")
)

overall_stats.show(1, False)


# sort the data at column-'county' and show top 10 records
sdf_filtered.orderBy("county").show(10, False)


# register the dataframe as a table
sdf_filtered.registerTempTable("covid19_20200413")



# here, we are grouping the data at 'county' level and this will take sum of cases, states and others
# and summarize it at 'county' level 
county_summary = sdf_filtered.groupBy(
    "county"
).agg(
    F.sum("cases").alias("total_cases"),
    F.sum("deaths").alias("total_deaths"),
    F.count("*").alias("number_of_records"),
    F.countDistinct("state").alias("number_of_states")
)

# order the county in alphabetical order and show top 20 records
county_summary.orderBy("county").show(20, False)


# noting for each county there should be no state with same name 
# so, here we are comparing the 'number_of_records' and 'number_of_states'
# and filtering only those records where both does not match
filtered_county_rec = county_summary.where(
    F.col("number_of_records") != F.col("number_of_states")
)

# checking the count of filtered records
# here count is zero - indicating that there are no duplicates
filtered_county_rec.count()

# so the dataframe returned is empty - containing no rows
filtered_county_rec.show()



## Some joins operations



left_table = county_summary.where(
    "county in ('Adair', 'Addison')"
)
left_table.show(5, False)

right_table = county_summary.where(
    "county in ('Ada', 'Accomack')"
)
right_table.show(5, False)

inner_table = county_summary.join(
    left_table,
    on=["county"],
    how="inner"
)

inner_table.show(100, False)


# applying left join
left_table_joined = left_table.join(
    right_table,
    on=["county"],
    how="left"
)

left_table_joined.show(100, False)

# applying right join
right_table_joined = left_table.join(
    right_table,
    on=["county"],
    how="right"
)

right_table_joined.show(100, False)


# applying  full join
outer_table_joined = left_table.join(
    right_table,
    on=["county"],
    how="outer"
)

outer_table_joined.show(100, False)


# get percentage # of cases for each state in Adair
# filter data for county - Adair
adir_overall = county_summary.where(
    "county in ('Adair')"
)
adir_overall.show(5, False)

# join the Adair summary to its states
perc_cases_statewise = sdf_filtered.join(
    adir_overall,
    on="county",
    how="inner"
)

perc_cases_statewise.show(10, False)


# calculate percentage of cases and deaths
perc_cases_statewise = perc_cases_statewise.withColumn(
    "perc_cases",
    F.col("cases")/F.col("total_cases")
).withColumn(
    "perc_deaths",
    F.col("deaths")/F.col("total_deaths")
)

perc_cases_statewise.show(10, False)