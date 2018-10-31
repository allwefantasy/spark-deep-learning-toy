from pyspark import SQLContext
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.streaming import StreamingContext
import os

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"

conf = SparkConf().setMaster("local[*]").setAppName("spark streaming")
# 必须要用conf=conf
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 10)

dStream = ssc.socketTextStream("127.0.0.1", 9000)


def func(rdd):
    def iter(iterator):
        for item in iterator:
            print(item)

    rdd.foreachPartition(iter)


def func2(rdd):
    '''
        利用之前的知识，
        我们回顾下：
        从rdd创建dataframe
        dataframe创建表
        使用sql
        返回dataframe进行各种保存
    '''
    sqlContext = SQLContext(rdd.context)
    newrdd = rdd.map(lambda line: [line])
    df = sqlContext.createDataFrame(newrdd, StructType([StructField(name="content", dataType=StringType())]))
    df.createOrReplaceTempView("data")
    sqlContext.sql("select content from data").show()


dStream.map(lambda line: line).foreachRDD(func)
ssc.start()
ssc.awaitTermination()
