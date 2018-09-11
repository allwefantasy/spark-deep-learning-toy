from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.sql import SparkSession
import logging

from pyspark.sql.types import StructField, StructType, BinaryType, StringType, ArrayType, ByteType
from sklearn.naive_bayes import GaussianNB
import os
from sklearn.externals import joblib
import pickle
import scipy.sparse as sp
from sklearn.svm import SVC
import io
import codecs

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"
logger = logging.getLogger(__name__)

base_dir = "/Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest"
spark = SparkSession.builder.master("local[*]").appName("example").getOrCreate()

data = spark.read.format("libsvm").load(base_dir + "/data/mllib/sample_libsvm_data.txt")


# 这段代码哪里错了？
def train(trainData):
    gnb = GaussianNB()
    X = [i["features"] for i in trainData]
    y = [i["label"] for i in trainData]
    model = gnb.fit(X, y)
    joblib.dump(model, '/tmp/sk_example.pkl')


# 如果数据量太大怎么办
def train2(trainData):
    gnb = GaussianNB()
    X = []
    y = []
    for i in trainData:
        X.append(i["features"])
        y.append(i["label"])
    model = gnb.fit(X, y)
    joblib.dump(model, '/tmp/sk_example.pkl')


# 如果不支持partial_fit怎么办？
def train3(trainData):
    gnb = GaussianNB()
    X = []
    y = []
    batch_size = 100
    count = 0
    for i in trainData:
        X.append(i["features"])
        y.append(i["label"])
        count += 1
        if count == batch_size:
            model = gnb.partial_fit(X, y)
        count = 0
        X.clear()
        y.clear()

    joblib.dump(model, '/tmp/sk_example.pkl')


# 使用稀疏存储，稀疏存储是个什么概念
# 如何使用稀疏存储
def train4(trainData):
    svc = SVC()
    y = []

    row_n = []
    col_n = []
    data_n = []
    row_index = 0
    feature_size = 0
    for i in trainData:
        feature = i["features"]
        feature_size = len(feature)
        dic = [(i, a) for i, a in enumerate(feature)]
        sv = SparseVector(feature_size, dic)
        for c in sv.indices:
            row_n.append(row_index)
            col_n.append(c)
            data_n.append(sv.values[list(sv.indices).index(c)])

        y.append(i["label"])

        row_index += 1
    X = sp.csc_matrix((data_n, (row_n, col_n)), shape=(row_index, feature_size))
    model = svc.fit(X, y)
    joblib.dump(model, '/tmp/sk_example.pkl')


# 我想一次训练多个模型怎么办?
# data.repartition(1).rdd.foreachPartition(train4)

# 把数据广播出去，保证每个节点有全量数据
dataBr = spark.sparkContext.broadcast(data.collect())


def train5(row):
    X = []
    y = []
    for i in dataBr.value:
        X.append(i["features"])
        y.append(i["label"])
    if row["model"] == "SVC":
        gnb = GaussianNB()
        model = gnb.fit(X, y)
        # 为什么还需要encode一下？
        pickled = codecs.encode(pickle.dumps(model), "base64").decode()
        return [row["model"], pickled]
    if row["model"] == "BAYES":
        svc = SVC()
        model = svc.fit(X, y)
        pickled = codecs.encode(pickle.dumps(model), "base64").decode()
        return [row["model"], pickled]


## 训练完成后，如何在PySpark中使用？
rdd = spark.createDataFrame([["SVC"], ["BAYES"]], ["model"]).rdd.map(train5)
spark.createDataFrame(rdd, schema=StructType([StructField(name="modelType", dataType=StringType()),
                                              StructField(name="modelBinary", dataType=StringType())])).write. \
    format("parquet"). \
    mode("overwrite").save("/tmp/wow")

# 理论上是两条记录
models = spark.read.parquet("/tmp/wow").collect()
svc = [x for x in models if x["modelType"] == "SVC"][0]["modelBinary"]

svcBr = spark.sparkContext.broadcast(svc)


def preidct(items):
    model = pickle.loads(codecs.decode(svcBr.value.encode(), "base64"))
    for item in items:
        yield [model.predict([item["features"]])[0].item(), item["label"]]


spark.createDataFrame(data.rdd.mapPartitions(preidct), ["predict_label", "real_label"]).show()
