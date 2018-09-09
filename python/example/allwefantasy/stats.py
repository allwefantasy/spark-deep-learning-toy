from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, \
    StringIndexer, OneHotEncoder

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f
import os

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"

spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()
session = spark

dataDir = "/Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest"
file = dataDir + "/data/FL_insurance_sample.csv"

question_ask = session.read.csv(
    file, encoding="utf-8",
    header=True)

# 新增加一个问题长度列

q_l_df = question_ask.withColumnRenamed("fl_site_limit", "q_l").selectExpr("cast (q_l as double) as q_l", "county")
# q_l_df = question_ask.withColumn("q_l", f.length("question").cast(FloatType()))
# q_l_df = question_ask.drop("q_l")

# 查看某个字段的一些统计值，比如数量，平均值，方差，最大最小值等
q_l_df.describe("q_l").show()
print(q_l_df.schema)
# 查看四分位，中位数，以及75%位置的数
q_l_df.stat.approxQuantile("q_l", [0.25, 0.5, 0.75], 0.0)

# 查看数据质量，看看某个字段是不是有为空的情况
q_l_df.where(f.col("county").isNull()).count()

# 过滤掉有缺失值的列
q_l_df_fix = q_l_df.where(f.col("county").isNotNull())

# 如果表字段只要有一个是空的，我都过滤掉：
q_l_df.na.drop()
q_l_df.na.drop(thresh=1)  # 只要一个为空就过滤掉

# 或者我填充下缺失值
q_l_df_fix = q_l_df.na.fill({'county': "未收集"})
# 看下效果
q_l_df.na.fill({'county': "未收集"}). \
    where(f.col("county").isNull()).count()

# 其实长度也有为null的
# q_l_df.where(f.col("Residential").isNull()).select(f.length("Residential")).show()

# 把某个字段转化为一个数字
string_index = StringIndexer(inputCol="county", outputCol="county_number")
q_l2_df = string_index.fit(q_l_df_fix).transform(q_l_df_fix)
# .select(f.col("section_number"), f.col("section"))

# 看看现在的表结构
q_l2_df.printSchema()

# 恩，我想知道两个字段没有关联关系
q_l2_df.corr("q_l", "county_number")
# 算两个字段的方差
q_l2_df.cov("q_l", "county_number")

# 我想知道现在某个字段分布
q_l2_df.groupBy("county").agg(f.count("county").alias("c")).orderBy(f.desc("c"))
# 中位分布
ss = q_l2_df.groupBy("county").agg(f.count("county").alias("question_num"))
ss.describe("question_num").show()
ss.stat.approxQuantile("question_num", [0.25, 0.5, 0.75], 0.0)
# 我想知道出现次数占比50%以上的某个类别是什么
q_l2_df.freqItems(["county"], support=0.5).show()

# 我采样后再用pandas 做单机处理
q_l2_df.sample(False, 0.01).toPandas()

# 我想根据字段获取不重复的记录
q_l2_df.dropDuplicates(['county']).show()

# 我想替换表里的一些数据
q_l2_df.replace(['未收集'], ["天呐"], subset=["county"]). \
    where(f.col("county") == "天呐").show()
