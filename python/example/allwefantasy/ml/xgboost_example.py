from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.sql import SparkSession
import logging

from pyspark.sql.types import StructField, StructType, BinaryType, StringType, ArrayType, ByteType
import os
import pickle
import codecs
import xgboost as xgb
import numpy as np

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"
logger = logging.getLogger(__name__)

base_dir = "/Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest"
spark = SparkSession.builder.master("local[*]").appName("example").getOrCreate()

## xgboost 的使用和Sklearn是高度一致的。
data = spark.read.format("libsvm").load(base_dir + "/data/mllib/sample_libsvm_data.txt")

dataBr = spark.sparkContext.broadcast(data.collect())


def train5(row):
    X = []
    y = []
    for i in dataBr.value:
        X.append(i["features"])
        y.append(i["label"])
    if row["model"] == "xgboost":
        xgb_model = xgb.XGBClassifier().fit(np.array(X), np.array(y))
        # xgb_model.save_model('0001.model')
        # # read file
        # with open('0001.model', 'rb') as f:
        #     bgdata = f.read()
        pickled = codecs.encode(pickle.dumps(xgb_model), "base64").decode()
        return [row["model"], pickled]


rdd = spark.createDataFrame([["xgboost"]], ["model"]).rdd.map(train5)
spark.createDataFrame(rdd, schema=StructType([StructField(name="modelType", dataType=StringType()),
                                              StructField(name="modelBinary", dataType=StringType())])).write. \
    format("parquet"). \
    mode("overwrite").save("/tmp/wow")

# 理论上是两条记录
models = spark.read.parquet("/tmp/wow").collect()
svc = [x for x in models if x["modelType"] == "xgboost"][0]["modelBinary"]

svcBr = spark.sparkContext.broadcast(svc)


def preidct(items):
    # with open('0001.model', 'wb') as f:
    #     f.write(codecs.decode(svcBr.value.encode(), "base64"))
    #
    # bst = xgb.Booster()  # init model
    # model = bst.load_model('0001.model')
    model = pickle.loads(codecs.decode(svcBr.value.encode(), "base64"))
    for item in items:
        yield [model.predict(np.array([item["features"]]))[0].item(), item["label"]]


spark.createDataFrame(data.rdd.mapPartitions(preidct), ["predict_label", "real_label"]).show()
