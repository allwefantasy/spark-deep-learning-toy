# -*- coding: utf-8

import os

from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"

'''
./bin/spark-submit \
--py-files /Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest/data/job2.zip \
--master "local[*]"  /Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest/python/example/allwefantasy/multi_file.py
'''

sc = SparkContext("local[*]", "App Name")
spark = SparkSession.builder.getOrCreate()
session = spark

# zip包目录结构
from job2.job import Job


@F.udf(returnType=StringType())
def wow(s):
    jack = Job()
    return jack.echo(s)


documentDF = session.createDataFrame([
    ("我是奥特曼",),
    ("I wish Java could use case classes",),
    ("Logistic regression models are neat",)
], ["text"])

documentDF.select(wow(F.col("text")).alias("e")).show()
