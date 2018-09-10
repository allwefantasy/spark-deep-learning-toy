# -*- coding: utf-8

import glob
import os
import jieba
import zipfile

from pyspark import SparkContext
from pyspark import SparkFiles
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"

'''
./bin/spark-submit \
--py-files /Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest/data/job.zip \
--files /Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest/data/dic.zip \
--master "local[*]"  /Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest/python/example/allwefantasy/jieba_example.py
'''


class TextAnalysis(object):
    clf = None

    @staticmethod
    def load_dic(zipResources):
        baseDir = SparkFiles.getRootDirectory()
        jieba.dt.tmp_dir = "/tmp/jieba"
        if not TextAnalysis.is_loaded():
            for zr in zipResources:
                with zipfile.ZipFile(baseDir + '/' + zr, 'r') as f:
                    f.extractall(SparkFiles.getRootDirectory())
                print("字典路径:{}".format(baseDir + '/' + zr))
            globPath = baseDir + "/dic/*.dic"
            dicts = glob.glob(globPath)
            for dictFile in dicts:
                temp = dictFile if os.path.exists(dictFile) else SparkFiles.get(dictFile)
                print("加载字典:{}".format(temp))
                jieba.load_userdict(temp)

            jieba.cut("nice to meet you")
            TextAnalysis.clf = "SUCCESS"

    @staticmethod
    def is_loaded():
        return TextAnalysis.clf is not None


sc = SparkContext("local[*]", "App Name")
spark = SparkSession.builder.getOrCreate()
session = spark

##获取字典名字
dicName = [file.split("/")[-1] for file in spark.sparkContext._conf.get("spark.files").split(",") if
           file.endswith(".zip")]


def lcut(s):
    TextAnalysis.load_dic(dicName)
    words = jieba.lcut(s)
    return words


lcut_udf = F.udf(lcut, ArrayType(StringType()))

documentDF = session.createDataFrame([
    ("我是奥特曼",),
    ("I wish Java could use case classes",),
    ("Logistic regression models are neat",)
], ["text"])

documentDF.withColumn("analized_text", lcut_udf(F.col("text"))).show()

# print("-----{}".format(os.listdir(baseDir)))
# # zip包目录结构
# from job.job import Job
#
# print("-----{}".format(Job()))
#
#
# @F.udf(returnType=StringType())
# def wow(s):
#     print("-----{}".format(Job()))
#     return ""
#
#
# documentDF.withColumn("analized_text", wow(F.col("text"))).show()
