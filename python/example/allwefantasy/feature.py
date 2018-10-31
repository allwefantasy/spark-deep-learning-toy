import logging
from random import Random

import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import ImputerModel, MinMaxScaler, MaxAbsScaler, QuantileDiscretizer
from pyspark.ml.linalg import VectorUDT
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

    def word2vec(self):
        from pyspark.ml.feature import Word2Vec

        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark".split(" "),),
            ("I wish Java could use case classes".split(" "),),
            ("Logistic regression models are neat".split(" "),)
        ], ["text"])

        word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
        model = word2Vec.fit(documentDF)

        # transform 其实只是做了个词向量求平均
        result = model.transform(documentDF)
        for row in result.collect():
            text, vector = row
            print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

        # 如果我希望把向量都拿出，以后在用呢？
        res = dict([(item["word"], item["vector"].toArray()) for item in model.getVectors().collect()])
        print(res["heard"])

    def normalize(self):
        from pyspark.ml.feature import Normalizer
        from pyspark.ml.linalg import Vectors

        df = self.session.createDataFrame([
            (0, [1.0, 0.5, -1.0]),
            (1, [2.0, 1.0, 1.0]),
            (2, [4.0, 10.0, 2.0])
        ], ["id", "features"])

        # Vector概念解释
        @F.udf(returnType=VectorUDT())
        def vectorize_from_array(a):
            return Vectors.dense(a)

        df = df.withColumn("features", vectorize_from_array(F.col("features")))
        # Normalize each Vector using $L^1$ norm.
        normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
        l1NormData = normalizer.transform(df)
        print("Normalized using L^1 norm")
        l1NormData.show()

        # Normalize each Vector using $L^\infty$ norm.
        lInfNormData = normalizer.transform(df, {normalizer.p: float("inf")})
        print("Normalized using L^inf norm")
        lInfNormData.show()

    # PySpark 提供了三种对列标准化的模式
    def standardScaler(self):
        from pyspark.ml.feature import StandardScaler

        dataFrame = self.session.read.format("libsvm").load(self.dataDir + "/data/mllib/sample_libsvm_data.txt")
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                                withStd=True, withMean=False)

        scalerModel = scaler.fit(dataFrame)
        scaledData = scalerModel.transform(dataFrame)
        scaledData.show()

        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

        # Compute summary statistics and generate MinMaxScalerModel
        scalerModel = scaler.fit(dataFrame)

        # rescale each feature to range [min, max].
        scaledData = scalerModel.transform(dataFrame)
        print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
        scaledData.select("features", "scaledFeatures").show()

        scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")

        # Compute summary statistics and generate MaxAbsScalerModel
        scalerModel = scaler.fit(dataFrame)

        # rescale each feature to range [-1, 1].
        scaledData = scalerModel.transform(dataFrame)

        scaledData.select("features", "scaledFeatures").show()

    def discrete(self):
        # Bucketizer
        from pyspark.ml.feature import Bucketizer

        splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]

        data = [(-999.9,), (-0.5,), (-0.3,), (0.0,), (0.2,), (999.9,)]
        dataFrame = self.session.createDataFrame(data, ["features"])

        bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bucketedFeatures")

        # Transform original data into its bucket index.
        bucketedData = bucketizer.transform(dataFrame)

        print("Bucketizer output with %d buckets" % (len(bucketizer.getSplits()) - 1))
        bucketedData.show()

        # QuantileDiscretizer

        data = [(0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2)]
        df = self.createDataFrame(data, ["id", "hour"])

        discretizer = QuantileDiscretizer(numBuckets=3, inputCol="hour", outputCol="result")

        result = discretizer.fit(df).transform(df)
        result.show()

    def onehot(self):
        from pyspark.ml.feature import OneHotEncoderEstimator

        df = self.session.createDataFrame([
            (0.0, 1.0),
            (1.0, 0.0),
            (2.0, 1.0),
            (0.0, 2.0),
            (0.0, 1.0),
            (2.0, 0.0)
        ], ["categoryIndex1", "categoryIndex2"])

        encoder = OneHotEncoderEstimator(inputCols=["categoryIndex1", "categoryIndex2"],
                                         outputCols=["categoryVec1", "categoryVec2"])
        model = encoder.fit(df)
        encoded = model.transform(df)
        encoded.show()

    def vectorCategory(self):
        from pyspark.ml.feature import VectorIndexer

        data = self.session.read.format("libsvm").load(self.dataDir + "/data/mllib/sample_libsvm_data.txt")

        indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=10)
        indexerModel = indexer.fit(data)

        categoricalFeatures = indexerModel.categoryMaps
        print("Chose %d categorical features: %s" %
              (len(categoricalFeatures), ", ".join(str(k) for k in categoricalFeatures.keys())))

        # Create new column "indexed" with categorical values transformed to indices
        indexedData = indexerModel.transform(data)
        indexedData.show()

        ## 问题来了，那我们怎么能够多个字段转化一个vector字段么？
        from pyspark.ml.linalg import Vectors
        from pyspark.ml.feature import VectorAssembler

        dataset = self.session.createDataFrame(
            [(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0)],
            ["id", "hour", "mobile", "userFeatures", "clicked"])

        assembler = VectorAssembler(
            inputCols=["hour", "mobile", "userFeatures"],
            outputCol="features")

        output = assembler.transform(dataset)
        print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
        output.select("features", "clicked").show(truncate=False)

    def stringIndexer(self):
        from pyspark.ml.feature import StringIndexer

        df = self.session.createDataFrame(
            [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
            ["id", "category"])

        indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
        indexed = indexer.fit(df).transform(df)
        indexed.show()

        # 来一个刺激的 我想拿到StringIndexer里完整映射关系怎么办？
        labels = indexer.fit(df)._call_java("labels")
        # sc = SparkContext._active_spark_context
        # java_array = sc._jvm.

        # labelToIndex = indexer.fit(df)._call_java("labelToIndex")
        print(labels)
        # print(labelToIndex)


if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.sql.execution.arrow.enabled", "true")
    FeatureExample.start(conf=conf)
    FeatureExample().stringIndexer()
    FeatureExample.shutdown()
