# EPFL CS-449 Project: Personalized Recommender with k-NN
### Authors 
Emna FENDRI : emna.fendri@epfl.ch  
Douglas BOUCHET : douglas.bouchet@epfl.ch  
## Dataset
* [MovieLens 100K movie ratings](https://grouplens.org/datasets/movielens/100k/)
* [MovieLens 1M movie ratings](https://grouplens.org/datasets/movielens/1m/)
* [MovieLens 25M movie ratings](https://grouplens.org/datasets/movielens/25m/)


 
## Description
### M1
This project is composed of two milestones. In the first milestone (M1) we progressively build a recommender system for Movies. We start by implementing a simple baseline prediction for recommendation then distribute it with Spark in Scala. We then compare the quality of its predictions to a second personalized approach based on similarities and k-NN. additionally, we measure the CPU time to develop insights in the system costs of the prediction methods.

### M2
In this milestone, we will parallelize the computation of similarities by leveraging the Breeze linear algebra library for Scala, effectively using more effcient low-level routines. We will also measure how well the Spark implementation scales when adding more executors. We also implementeed a version of approximate k-NN that could be useful in very large datasets. Finally, we compute economic ratios to help us have insight on how to choose the most appropriate infrastructure for our needs.

## Files
```bash

├── M1
├   ├── report.pdf
├   └── src 
├       ├── main/scala
├           ├── distributed
├               └──DistributedBaseline.scala
├           ├── predict
├               ├── Baseline.scala
├               ├── Personalized.scala
├               └──kNN.scala
├           ├── recommend
├               └── Recommender.scala
├           └── shared
├               └──predictions.scala
├       └── test/scala
├    
└── M2
    ├── report.pdf
    └── src 
        ├── main/scala
            ├── distributed
                ├── Approximate.scala
                └── Exact.scala
            ├── economics
                └── Economics.scala
            ├── optimizing
                └── Optimizing.scala
            └── shared
                └──predictions.scala
            └── test/scala
  

```
## References
- Essential sbt: https://www.scalawilliam.com/essential-sbt/  
- Explore Spark Interactively (supports autocompletion with tabs!): https://spark.apache.org/docs/latest/quick-start.html
- Scallop Argument Parsing: https://github.com/scallop/scallop/wiki
- Spark Resilient Distributed Dataset (RDD): https://spark.apache.org/docs/3.0.1/api/scala/org/apache/spark/rdd/RDD.html
