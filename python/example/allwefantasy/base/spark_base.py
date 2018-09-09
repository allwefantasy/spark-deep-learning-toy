from pyspark import SQLContext
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
import os

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"


class _SparkBase(object):
    @classmethod
    def start(cls, conf=SparkConf()):
        cls.sc = SparkContext(master='local[*]', appName=cls.__name__, conf=conf)
        cls.sql = SQLContext(cls.sc)
        cls.session = SparkSession.builder.getOrCreate()
        cls.dataDir = "/Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest"

    @classmethod
    def shutdown(cls):
        cls.session.stop()
        cls.session = None
        cls.sc.stop()
        cls.sc = None
