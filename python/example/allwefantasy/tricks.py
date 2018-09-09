import logging
from random import Random

import pyspark.sql.functions as F
from pyspark import SparkConf
from pyspark.sql.types import *

from example.allwefantasy.base.spark_base import _SparkBase
import example.allwefantasy.time_profile as TimeProfile
import pandas as pd

logger = logging.getLogger(__name__)


class PySparkOptimize(_SparkBase):
    # 普通udf处理
    def trick1(self):
        df = self.session.range(0, 1000000).select("id", F.rand(seed=10).alias("uniform"),
                                                   F.randn(seed=27).alias("normal"))

        @F.udf('double')
        def plus_one(v):
            return v + 1

        TimeProfile.profile(lambda: df.withColumn('v2', plus_one(df.uniform)).count())()
        TimeProfile.print_prof_data(clear=True)

        @F.pandas_udf('double', F.PandasUDFType.SCALAR)
        def pandas_plus_one(v):
            return v + 1

        TimeProfile.profile(lambda: df.withColumn('v2', pandas_plus_one(df.uniform)).count())()
        TimeProfile.print_prof_data(clear=True)

    # 尝试换成arrow
    def trick4(self):
        df = self.session.range(0, 1000000).select("id", F.rand(seed=10).alias("uniform"),
                                                   F.randn(seed=27).alias("normal"))
        # 更少的内存和更快的速度
        TimeProfile.profile(lambda: df.toPandas())()
        TimeProfile.print_prof_data(clear=True)

    # 聚合函数处理
    def trick2(self):
        @F.udf('integer')
        def random(v):
            return Random().randint(0, 3)

        df = self.session.range(0, 100).withColumn("v", random(F.col("id"))).select("id", "v",
                                                                                    F.rand(seed=10).alias("uniform"),
                                                                                    F.randn(seed=27).alias("normal"))

        @F.pandas_udf(df.schema, F.PandasUDFType.GROUPED_MAP)
        def subtract_mean(pdf):
            return pdf.assign(uniform=pdf.uniform - pdf.uniform.mean())

        df.groupby('v').apply(subtract_mean).show()

        @F.pandas_udf(StructType([
            StructField(name="v", dataType=IntegerType()),
            StructField(name="add_all", dataType=DoubleType())
        ]), F.PandasUDFType.GROUPED_MAP)
        def addAll(pdf):
            return pd.DataFrame(data={"v": pdf.v[0], 'add_all': [pdf.uniform.sum() + pdf.normal.sum()]})

        df.groupby('v').apply(addAll).show()

    def trick3(self):
        df = self.session.range(0, 1000000).select("id", F.rand(seed=10).alias("uniform"),
                                                   F.randn(seed=27).alias("normal"))
        # 更少的内存和更快的速度
        TimeProfile.profile(lambda: df.toPandas())()
        TimeProfile.print_prof_data(clear=True)


if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.sql.execution.arrow.enabled", "true")
    PySparkOptimize.start(conf=conf)
    PySparkOptimize().trick2()
    PySparkOptimize.shutdown()
