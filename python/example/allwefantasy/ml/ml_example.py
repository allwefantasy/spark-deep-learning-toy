from pyspark import Row
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor, DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

from example.allwefantasy.base.spark_base import _SparkBase


class MLExample(_SparkBase):
    def dtr(self):
        # Load and parse the data file, converting it to a DataFrame.
        data = self.session.read.format("libsvm").load(self.dataDir + "/data/mllib/sample_libsvm_data.txt")

        # Automatically identify categorical features, and index them.
        # Set maxCategories so features with > 4 distinct values are treated as continuous.
        featureIndexer = \
            VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

        # Split the data into training and test sets (30% held out for testing)
        (trainingData, testData) = data.randomSplit([0.7, 0.3])

        # Train a GBT model.
        drg = DecisionTreeRegressor(featuresCol="indexedFeatures")

        # Chain indexer and GBT in a Pipeline
        pipeline = Pipeline(stages=[featureIndexer, drg])

        # Train model.  This also runs the indexer.
        model = pipeline.fit(trainingData)

        # Make predictions.
        predictions = model.transform(testData)

        # Select example rows to display.
        predictions.select("prediction", "label", "features").show(5)

        # Select (prediction, true label) and compute test error
        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

        gbtModel = model.stages[1]
        print(gbtModel)  # summary only

    def classify(self):
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator

        # Load and parse the data file, converting it to a DataFrame.
        data = self.read.format("libsvm").load(self.dataDir + "/data/mllib/sample_libsvm_data.txt")

        # Index labels, adding metadata to the label column.
        # Fit on whole dataset to include all labels in index.
        labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

        # Automatically identify categorical features, and index them.
        # Set maxCategories so features with > 4 distinct values are treated as continuous.
        featureIndexer = \
            VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

        # Split the data into training and test sets (30% held out for testing)
        (trainingData, testData) = data.randomSplit([0.7, 0.3])

        # Train a RandomForest model.
        rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

        # Convert indexed labels back to original labels.
        labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                       labels=labelIndexer.labels)

        # Chain indexers and forest in a Pipeline
        pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

        # Train model.  This also runs the indexers.
        model = pipeline.fit(trainingData)

        # Make predictions.
        predictions = model.transform(testData)

        # Select example rows to display.
        predictions.select("predictedLabel", "label", "features").show(5)

        # Select (prediction, true label) and compute test error
        evaluator = MulticlassClassificationEvaluator(
            labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Test Error = %g" % (1.0 - accuracy))

        rfModel = model.stages[2]
        print(rfModel)  # summary only

    def cluster(self):
        from pyspark.ml.clustering import KMeans
        from pyspark.ml.evaluation import ClusteringEvaluator

        # Loads data.
        dataset = self.read.format("libsvm").load(self.dataDir + "data/mllib/sample_kmeans_data.txt")

        # Trains a k-means model.
        kmeans = KMeans().setK(2).setSeed(1)
        model = kmeans.fit(dataset)

        # Make predictions
        predictions = model.transform(dataset)

        # Evaluate clustering by computing Silhouette score
        evaluator = ClusteringEvaluator()

        silhouette = evaluator.evaluate(predictions)
        print("Silhouette with squared euclidean distance = " + str(silhouette))

        # Shows the result.
        centers = model.clusterCenters()
        print("Cluster Centers: ")
        for center in centers:
            print(center)



if __name__ == '__main__':
    MLExample.start()
    MLExample().dtr()
    MLExample.shutdown()
