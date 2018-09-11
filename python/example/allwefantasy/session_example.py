from pyspark.sql import SparkSession
import logging
import os
from pyspark.sql.types import *

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"
logger = logging.getLogger(__name__)

base_dir = "/Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest"
session = SparkSession.builder.master("local[*]").appName("example").getOrCreate()

'''
当拿到session，你都会干嘛?

1. 读取结构化数据
2. 读取非结构化数据后转化为结构化数据
3. 还可以读取流
'''

# 结构化
# 快捷读法

csv_file = base_dir + "/data/FL_insurance_sample.csv"

data = session.read.csv(
    csv_file,
    encoding="utf-8",
    header=True)

# 标准方式
data = session.read.format("csv").option("encoding", "utf-8").option("header", "true").load(
    csv_file)
# or
data = session.read.format("csv").options(encoding="utf-8", header=True).load(csv_file)

# 非结构化数据读取
# os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"
rdd = session.sparkContext.textFile(base_dir + "/data/raw_file.txt").map(lambda line: line.split(","))
data = session.createDataFrame(rdd, StructType([
    StructField(name="a1", dataType=StringType()),
    StructField(name="a2", dataType=StringType()),
    StructField(name="a3", dataType=StringType())
]))

data.show()



