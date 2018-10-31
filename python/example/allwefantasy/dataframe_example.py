from pyspark import Row
from pyspark.sql.types import *
import pyspark.sql.functions as F

from example.allwefantasy.base.spark_base import _SparkBase


class DataframeExample(_SparkBase):
    def __init__(self):
        self.df = self.session.read.csv(
            self.dataDir + "/data/raw_file.txt",
            encoding="utf-8",
            header=False,
            schema=StructType(
                [StructField("a1", StringType()), StructField("b1", StringType()), StructField("c1", StringType())])
        )

    def select_usage(self):
        self.df.groupBy(F.col("a1")).\
            agg(F.count("b1").alias("b1count")).\
            select(F.col("a1"), F.col("b1count")).\
            show()

    def udf_usage(self):
        def concat(column, wow):
            return column + ":" + str(wow)

        concat_udf = F.udf(concat, StringType())
        self.df.select(concat_udf(F.col("a1"), F.lit(1))).show()

    def with_column_usage(self):
        def concat(column, wow):
            return column + ":" + str(wow)

        concat_udf = F.udf(concat, StringType())
        self.df.withColumn("a1", concat_udf(F.col("a1"), F.lit("-"))).show()
        # 如果我要修改所有列/很多列怎么办？
        expres = [(column, concat_udf(F.col(column))) for column in self.df.schema.fields]
        newdf = self.df
        for s_expr in expres:
            newdf = newdf.withColumn(s_expr[0], s_expr[1])

        newdf.show()

    def save_usage(self):
        # 思考： 如果想自己定义怎么写入呢?,比如我想写成一个jack.txt
        self.df.write.mode("overwrite").format("json").save(self.dataDir + "/data/tmp")
        file = self.dataDir + "/data/jack.txt"

        def write(rows):
            with open(file, "w") as f:
                for row in rows:
                    f.write(row[0] + row[1] + row[2] + "\n")

        self.df.rdd.repartition(1).foreachPartition(write)

    def rdd_dataframe(self):
        # rdd -> dataframe
        rdd = self.session.sparkContext.textFile(self.dataDir + "/data/raw_file.txt")
        # 这是第一种办法
        # rdd = rdd.map(lambda line: line.split(","))
        # df = self.session.createDataFrame(rdd, StructType(
        #     [StructField("a1", StringType()), StructField("b1", StringType()), StructField("c1", StringType())]))
        # 这是第二种办法。 第一种办法速度更快些，不需要inferSchema
        rdd = rdd.map(lambda line: Row(**dict([(i[0], i[1]) for i in list(zip(["a1", "b1", "c1"], line.split(",")))])))
        df = self.session.createDataFrame(rdd)
        df.show()

        # dataframe -> rdd
        for item in df.rdd.collect():
            print(item)

    def sql_dataframe(self):
        self.df.createOrReplaceTempView("test")
        self.session.sql("select * from test").show()
        newdf = self.session.sql("select * from test")
        newdf.select("*").show()
        # or
        # newdf.write.....


if __name__ == '__main__':
    DataframeExample.start()
    DataframeExample().with_column_usage()
    DataframeExample.shutdown()
