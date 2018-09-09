from pyspark import Row
from pyspark.sql.types import *
import pyspark.sql.functions as F

from example.allwefantasy.base.spark_base import _SparkBase


class StructuredStreaming(_SparkBase):
    def run(self):
        df = self.session.readStream.format("socket").option("host", "127.0.0.1"). \
            option("port", "9000").load()
        words = df.groupBy("value").agg(F.count("value").alias("word_num")).select("value", "word_num")
        stream = words.writeStream.outputMode("complete").format("console").trigger(processingTime="10 seconds").start()
        return stream


if __name__ == '__main__':
    StructuredStreaming.start()
    query = StructuredStreaming().run()
    query.awaitTermination()
