from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext("local[*]", "Pipeline")
spark = SparkSession.builder.getOrCreate()
session = spark


val relationships = df.rdd.map(f => Edge(f.getAs[Long](rowNumCol), f.getAs[Long](columnNumCol), f.getAs[Double](edgeValueCol)))
val graph = Graph.fromEdges(relationships, 0d)

val vertexCount = Math.max(Math.round(graph.vertices.count() * minCommunityPercent), minCommunitySize)

val validGraph = graph.subgraph(epred = et => {
    et.attr > minSimilarity
}).connectedComponents()


val rdd = validGraph.vertices.map(f => VeterxAndGroup(f._1, f._2)).groupBy(f => f.group).
filter(f => f._2.size > vertexCount).
map(f => GroupVeterxs(f._1, f._2.map(k => k.vertexId).toSeq))