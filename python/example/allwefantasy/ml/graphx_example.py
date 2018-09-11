from functools import reduce
from graphframes import GraphFrame
from pyspark import SparkContext
from pyspark.sql import SparkSession
import os

import pyspark.sql.functions as F

'''
关于环境问题；如何在idea里找到当前运行的spark环境；
如何解决运行时找不到graphframes的问题
'''
os.environ["PYSPARK_PYTHON"] = "/Users/allwefantasy/deepavlovpy3/bin/python3"
sc = SparkContext("local[*]", "graphframe")
sc.setCheckpointDir("/tmp/graphframe")
spark = SparkSession.builder.getOrCreate()
session = spark

# Vertex DataFrame
v = spark.createDataFrame([
    ("a", "Alice", 34),
    ("b", "Bob", 36),
    ("c", "Charlie", 30),
    ("d", "David", 29),
    ("e", "Esther", 32),
    ("f", "Fanny", 36),
    ("g", "Gabby", 60)
], ["id", "name", "age"])
# Edge DataFrame
e = spark.createDataFrame([
    ("a", "b", "friend"),
    ("b", "c", "follow"),
    ("c", "b", "follow"),
    ("f", "c", "follow"),
    ("e", "f", "follow"),
    ("e", "d", "friend"),
    ("d", "a", "friend"),
    ("a", "e", "friend")
], ["src", "dst", "relationship"])
# Create a GraphFrame,本质上就是两个dataframe
g = GraphFrame(v, e)

# 获取两个dataframe
g.vertices
g.edges

## 每个节点的出度 入度
g.inDegrees
g.outDegrees

# motifs 语法
chain4 = g.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[cd]->(d)")
# chain4.show()
# g.find("(c)-[m]->()").show()
# Query on sequence, with state (cnt)
#  (a) Define method for updating state given the next element of the motif.
sumFriends = \
    lambda cnt, relationship: F.when(relationship == "friend", cnt + 1).otherwise(cnt)
#  (b) Use sequence operation to apply method to sequence of elements in motif.
#      In this case, the elements are the 3 edges.
condition = \
    reduce(lambda cnt, e: sumFriends(cnt, F.col(e).relationship), ["ab", "bc", "cd"], F.lit(0))
#  (c) Apply filter to DataFrame.
chainWith2Friends2 = chain4.where(condition >= 2)
# chainWith2Friends2.show()

result = g.connectedComponents()
# 结果含义的解释
# result.show()

# 强连通图和连通图的区别
'''
Connected is usually associated with undirected graphs (two way edges): there is a path between every two nodes.
Strongly connected is usually associated with directed graphs (one way edges): there is a route between every two nodes.
Complete graphs are undirected graphs where there is an edge between every pair of nodes.
'''
result = g.stronglyConnectedComponents(maxIter=10)
# result.orderBy("component").show()

## 社区发现，本质是个聚类算法
result = g.labelPropagation(maxIter=5)
# result.show()

## 一个实例，背景讲解

v = spark.createDataFrame([
    ("a", "Alice", 34),
    ("b", "Bob", 36),
    ("c", "Charlie", 30),
    ("d", "David", 29),
    ("e", "Esther", 32),
    ("f", "Fanny", 36),
    ("g", "Gabby", 60)
], ["id", "name", "age"])
# Edge DataFrame
e = spark.createDataFrame([
    ("a", "b", 0.7),
    ("b", "c", 0.8),
    ("c", "b", 0.1),
    ("f", "c", 0.8),
    ("e", "f", 0.2),
    ("e", "d", 0.2),
    ("d", "a", 0.3),
    ("a", "e", 0.22)
], ["src", "dst", "sim"])
# Create a GraphFrame,本质上就是两个dataframe
g = GraphFrame(v, e)
g = g.filterEdges("sim > 0.6")
result = g.connectedComponents()
# graphframe的好处，可以用我们熟悉的api做后续处理
result.groupby("component").agg(F.collect_list("name").alias("name")).select(F.col("component").alias("group"),
                                                                             F.col("name")).show()
