from pyspark.sql import SparkSession
import logging
import os
# pip install mysqlclient
import MySQLdb
from pyspark.sql.types import *

os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"
logger = logging.getLogger(__name__)

base_dir = "/Users/allwefantasy/CSDNWorkSpace/spark-deep-learning_latest"
session = SparkSession.builder.master("local[*]").appName("example").getOrCreate()

data = session.createDataFrame([
    ("a b c d e spark", 1.0),
    ("b d", 0.0),
    ("spark f g h", 1.0),
    ("hadoop mapreduce", 0.0)
], ["a", "b"])


# 怎么运行的 大家可以想想
# 为什么不用map()而是mapPartitions
# 还有没有更高效的办法
def writePartition(data):
    conn = MySQLdb.connect(host="localhost",
                           user="root",
                           passwd="csdn.net",
                           db="wow")
    x = conn.cursor()
    res = []
    for it in data:
        res.append(it)
    try:
        x.executemany(
            """INSERT INTO testm (a,b)
            VALUES (%s, %s)""",
            res)
        conn.commit()
    except:
        conn.rollback()

    conn.close()
    return iter([])


def createTable():
    conn = MySQLdb.connect(host="localhost",
                           user="root",
                           passwd="csdn.net",
                           db="wow")
    x = conn.cursor()

    try:
        x.execute("""create table if exists testm(a text,b text)""")
        conn.commit()
    except:
        conn.rollback()
    conn.close()


createTable()
data.rdd.mapPartitions(writePartition).count()
