import logging
from random import Random

import pyspark.sql.functions as F
from pyspark import SparkConf
from pyspark.sql.types import *

from example.allwefantasy.base.spark_base import _SparkBase

logger = logging.getLogger(__name__)


class DataSourceExample(_SparkBase):
    def parquet(self):
        self.session.read.format("parquet").load("path")

    def csv(self):
        self.session.read.format("csv").load("path")

    def libsvm(self):
        self.session.read.format("libsvm").load("path")

    def mysql(self):
        self.session.read.format("jdbc").option("url", "jdbc:mysql://127.0.0.1:3306/wow") \
            .option("dbtable", "test") \
            .option("driver", "com.mysql.jdbc.Driver") \
            .load()

    def json(self):
        self.session.read.format("json").load("path")
