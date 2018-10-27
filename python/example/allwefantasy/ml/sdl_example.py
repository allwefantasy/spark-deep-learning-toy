from pyspark.sql import SparkSession
import logging
import os
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from sparkdl import DeepImageFeaturizer
from sparkdl.image import imageIO as imageIO
from pyspark.ml.image import ImageSchema

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"
logger = logging.getLogger(__name__)

base_dir = "/Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest/data/cifar/test"
spark = SparkSession.builder.master("local[*]").appName("example").getOrCreate()

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# train_images_df = imageIO.readImagesWithCustomFn(base_dir, decode_f=imageIO.PIL_decode)
@F.udf('int')
def extract_label(v):
    return labels.index(v.split("_")[2].split(".")[0])

# 图片非常消耗资源，该怎么办？
train_images_df = ImageSchema.readImages(base_dir, sampleRatio=0.4, numPartitions=4).withColumn("label",
                                                                                                extract_label(F.col(
                                                                                                    "image.origin")))
# train_images_df.show(10)

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])

model = p.fit(train_images_df)  # train_images_df is a dataset of images and labels

# Inspect training error
df = model.transform(train_images_df.limit(10)).select("image", "probability", "uri", "label")
predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
