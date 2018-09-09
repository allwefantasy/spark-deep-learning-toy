import logging
from random import Random

import pyspark.sql.functions as F
from pyspark import SparkConf
from pyspark.ml.feature import ImputerModel
from pyspark.sql.types import *

from example.allwefantasy.base.spark_base import _SparkBase


class FeatureExample(_SparkBase):
    def impute(self):
        from pyspark.ml.feature import Imputer

        df = self.session.createDataFrame([
            (1.0, float("nan")),
            (2.0, float("nan")),
            (float("nan"), 3.0),
            (4.0, 4.0),
            (5.0, 5.0)
        ], ["a", "b"])

        # 默认采用平均值进行填充
        imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
        model = imputer.fit(df)
        model.transform(df).show()

        # 我们也可以设置为中位数，以及判定哪些是缺失值
        # null 则自动被认为缺失值
        imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"], strategy="median",
                          missingValue=float("nan"))
        model = imputer.fit(df)
        model.transform(df).show()

        ## fit过程一般我们认为是一个学习的过程，我们也可以吧这个过程保留下来
        ## 遗憾的是，我们暂时没有办法变更参数
        model.write().overwrite().save("/tmp/wow")
        model = ImputerModel.read().load("/tmp/wow")
        model.transform(df).show()

    def outlier(self):
        df = self.session.createDataFrame([
            (1.0, float("nan")),
            (2.0, float("nan")),
            (10000.0, 3.0),
            (4.0, 4.0),
            (5.0, 5.0)
        ], ["a", "b"])

        def kill_outlier():
            quantiles = df.stat.approxQuantile("a", [0.25, 0.5, 0.75], 0.0)
            Q1 = quantiles[0]
            Q3 = quantiles[2]
            IQR = Q3 - Q1
            lowerRange = Q1 - 1.5 * IQR
            upperRange = Q3 + 1.5 * IQR

            Q2 = quantiles[1]

            @F.udf(returnType=DoubleType())
            def wrapper(a):
                if a < lowerRange or a > upperRange:
                    return Q2
                else:
                    return a

            newDF = df.withColumn("out_a", wrapper(F.col("a")))
            newDF.show()

        kill_outlier()


if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.sql.execution.arrow.enabled", "true")
    FeatureExample.start(conf=conf)
    FeatureExample().impute()
    FeatureExample.shutdown()
