from pyspark import Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, SparkContext, SparkSession
import random
import pyspark.sql.functions as F
import os

from pyspark.rdd import Partitioner
from pyspark.sql.types import *

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"

sc = SparkContext("local[*]", "Pipeline")
spark = SparkSession.builder.getOrCreate()
session = spark

data = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])
## 最简单的
(train, validate, test) = data.randomSplit([0.6, 0.2, 0.2])
train.show()

## 如果你想每个分类都按比例切分呢？

# [0.0-0.6),[0.6-0.8) [0.8-1.0)
labelMap = dict(data.groupby("label").agg(F.count("id").alias("c")).select("label", "c").collect())


# 分区的高阶用法,我们让每个分类都在一个分区里
def partitionFun(key):
    return labelMap[float(key)]


def processPartition(iter):
    def split_wow():
        f = random.random()
        print(f)
        split_tag = 0
        if f < 0.6:
            split_tag = 0
        if 0.6 <= f < 0.8:
            split_tag = 1
        if 0.8 <= f:
            split_tag = 2
        return split_tag

    for element in iter:
        split_tag = split_wow()
        temp = element[1].asDict()
        temp["__split__"] = split_tag
        yield Row(**temp)


new_rdd = data.rdd.map(lambda r: (r['label'], r)).partitionBy(len(labelMap), partitionFun).mapPartitions(
    processPartition)

new_fields = data.schema.fields + [StructField(name="__split__", dataType=IntegerType())]
print(new_fields)
df = session.createDataFrame(new_rdd,
                             schema=StructType(new_fields))

df.show()
