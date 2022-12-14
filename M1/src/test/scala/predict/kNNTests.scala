package test.predict

import org.scalatest._
import funsuite._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import shared.predictions._
import tests.shared.helpers._
import ujson._

class kNNTests extends AnyFunSuite with BeforeAndAfterAll {

  val separator = "\t"
  var spark: org.apache.spark.sql.SparkSession = _

  val train2Path = "data/ml-100k/u2.base"
  val test2Path = "data/ml-100k/u2.test"
  var train2: Array[shared.predictions.Rating] = null
  var test2: Array[shared.predictions.Rating] = null

  var adjustedCosine: Map[Int, Map[Int, Double]] = null

  override def beforeAll {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    spark = SparkSession
      .builder()
      .master("local[1]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    // For these questions, train and test are collected in a scala Array
    // to not depend on Spark
    train2 = load(spark, train2Path, separator).collect()
    test2 = load(spark, test2Path, separator).collect()
  }

  // All the functions definitions for the tests below (and the tests in other suites)
  // should be in a single library, 'src/main/scala/shared/predictions.scala'.

  // Provide tests to show how to call your code to do the following tasks.
  // Ensure you use the same function calls to produce the JSON outputs in
  // src/main/scala/predict/Baseline.scala.
  // Add assertions with the answer you expect from your code, up to the 4th
  // decimal after the (floating) point, on data/ml-100k/u2.base (as loaded above).
  test("kNN predictor with k=10") {

    // we need this in order to compute knn-similarities
    val userArray = train2.map(_.user).distinct
    val cosSimilarity = cosineSimilarity(train2)
    // Create predictor on train2
    val predictorKnn10 = predictorKnn(train2, 10)
    //Similarity between user 1 and itself
    assert(
      within(
        knnSimilarity(1, 1,  userArray, cosSimilarity, 10),
        0.0,
        0.0001
      )
    )

    // Similarity between user 1 and 864
    assert(
      within(
        knnSimilarity(1, 864, userArray, cosSimilarity, 10),
        0.2423,
        0.0001
      )
    )

    // Similarity between user 1 and 886
    assert(
      within(
        knnSimilarity(1, 886, userArray, cosSimilarity, 10),
        0.0,
        0.0001
      )
    )

    // Prediction user 1 and item 1
    assert(within(predictorKnn10(1, 1), 4.3190, 0.0001))

    // MAE on test2
    assert(within(getMAE(predictorKnn10, test2), 0.8287,0.0001))
  }

  test("kNN Mae") {
    // Compute MAE for k around the baseline MAE
    //val baseline_mae = baselineTestError(train2, test2) //0.7604467914538644
    val baseline_mae =
      getMAE(predictorBaseline(train2), test2) //0.7604467914538644

    // Ensure the MAEs are indeed lower/higher than baseline
    val results = List(10, 30, 50, 100, 200, 300, 400, 800, 943)
      .map(k =>
        List(
          k,
          getMAE(predictorKnn(train2, k), test2) // Compute MAE
        )
      )
      .toList

    results.foreach(elem => {
      elem.head match {
        case 10  => assert(elem(1) > baseline_mae)
        case 30  => assert(elem(1) > baseline_mae)
        case 50  => assert(elem(1) > baseline_mae)
        case 100 => assert(elem(1) < baseline_mae)
        case 200 => assert(elem(1) < baseline_mae)
        case 300 => assert(elem(1) < baseline_mae)
        case 400 => assert(elem(1) < baseline_mae)
        case 800 => assert(elem(1) < baseline_mae)
        case 943 => assert(elem(1) < baseline_mae)
      }
    })
  }
}
