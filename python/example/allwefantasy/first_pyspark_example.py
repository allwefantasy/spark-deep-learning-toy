from pyspark import SQLContext
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
import example.allwefantasy.time_profile as TimeProfile
import logging

from example.allwefantasy.base.spark_base import _SparkBase

logger = logging.getLogger(__name__)


class _SparkExample(_SparkBase):
    def first_example(self):
        '''
         question, what if their no header in csv file?
        '''
        document_df = self.session.read.csv(
            self.dataDir + "/data/FL_insurance_sample.csv",
            encoding="utf-8",
            header=True)

        def concat(a, b):
            return str(a) + str(b)

        # document_df.select(F.udf(concat)(F.col("policyID"), F.col("statecode"))).show(truncate=False)
        table = {
            "abc": "jack",
            "jack": "abc"
        }

        def jack(f):
            return table["abc"]

        items1 = self.sc.textFile(self.dataDir + "/data/FL_insurance_sample.csv"). \
            map(jack)

        print(items1.collect())

    def csv_without_head(self):
        document_df = self.session.read.csv(
            self.dataDir + "/data/raw_file.txt",
            encoding="utf-8",
            header=False,
            schema=StructType(
                [StructField("a1", StringType()), StructField("b1", StringType()), StructField("c1", StringType())])
        )
        document_df.show()

    def find_bug(self):
        buffer = []

        def abc(item):
            buffer.append(item)
            return item

        rdd1 = self.sc.textFile(self.dataDir + "/data/raw_file.txt"). \
            map(lambda f: f.split(",")). \
            filter(lambda f: len(f) > 0). \
            map(lambda f: abc(f))
        print(rdd1.collect())
        logger.warn("--------------")
        for item in buffer:
            print(item)
        logger.warn("--------------")

    def cache_persist(self):
        document_df = self.session.read.csv(
            self.dataDir + "/data/FL_insurance_sample.csv",
            encoding="utf-8",
            header=True)

        def concat(a, b):
            return str(a) + str(b)

        new_d_df = document_df.select(F.udf(concat)(F.col("policyID"), F.col("statecode")))

        TimeProfile.profile(lambda: new_d_df.count())()
        logger.warn("normal:")
        TimeProfile.print_prof_data(clear=True)

        logger.warn("cached:")
        new_d_df.cache()
        new_d_df.count()
        TimeProfile.profile(lambda: new_d_df.count())()
        TimeProfile.print_prof_data(clear=True)

        logger.warn("unpersist:")
        new_d_df.unpersist()
        TimeProfile.profile(lambda: new_d_df.count())()
        TimeProfile.print_prof_data(clear=True)

    def broadcast(self):
        document_df = self.session.read.csv(
            self.dataDir + "/data/FL_insurance_sample.csv",
            encoding="utf-8",
            header=True)
        # almost 100m
        huge_dic = dict([(i, bytearray(1024 * 8 * 1024)) for i in range(100)])

        def run():
            def m(index):
                return huge_dic[index]

            newdf = document_df.select(F.udf(m)(F.lit(1)))
            [newdf.count() for i in range(10)]

        TimeProfile.profile(run)()
        TimeProfile.print_prof_data(clear=True)

        huge_dic_br = self.sc.broadcast(huge_dic)

        def run2():
            def m(index):
                return huge_dic_br.value[index]

            newdf = document_df.select(F.udf(m)(F.lit(1)))
            [newdf.count() for i in range(10)]

        TimeProfile.profile(run2)()
        TimeProfile.print_prof_data(clear=True)

        TimeProfile.profile(run)()
        TimeProfile.print_prof_data(clear=True)

    def how_to_create_data_from_non_rdd(self):
        # 自动类型推倒
        data = [
            ("Hi I heard about Spark", "Hi I heard about Spark", 2.0, 3.0, 1, 2),
            ("I wish Java could use case classes", "I wish Java could use case classes", 3.0, 4.0, 0, 4),
            ("Logistic regression models are neat", "Logistic regression models are neat", 4.0, 5.0, 2, 5)
        ]
        df = self.session.createDataFrame(data, ["sentence", "sentence2", "f1", "f2", "preds", "i1"])
        df.show()

        df = self.session.createDataFrame([{'name': 'Alice', 'age': 1}, {'name': 'Alice', 'age': 1}])
        df.show()

        df = self.session.createDataFrame([
            ("Hi I heard about Spark", "Hi I heard about Spark", 2.0, 3.0, 1, 2),
            ("I wish Java could use case classes", "I wish Java could use case classes", 3.0, 4.0, 0, 4),
            ("Logistic regression models are neat", "Logistic regression models are neat", 4.0, 5.0, 2, 5)
        ], schema=StructType([
            StructField(name="sentence", dataType=StringType()),
            StructField(name="sentence2", dataType=StringType()),
            StructField(name="f1", dataType=DoubleType()),
            StructField(name="f2", dataType=DoubleType()),
            StructField(name="preds", dataType=LongType()),
            StructField(name="i1", dataType=LongType())
        ]))
        df.show()


if __name__ == '__main__':
    _SparkExample.start()
    _SparkExample().broadcast()
    _SparkExample.shutdown()
