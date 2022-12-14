package test.recommend

import org.scalatest._
import funsuite._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import shared.predictions._
import tests.shared.helpers._
import ujson._

class RecommenderTests extends AnyFunSuite with BeforeAndAfterAll {

  val separator = "\t"
  var spark: org.apache.spark.sql.SparkSession = _

  val dataPath = "data/ml-100k/u.data"
  val personalPath = "data/personal.csv"
  var data: Array[shared.predictions.Rating] = null
  var personal: Array[shared.predictions.Rating] = null
  var train: Array[shared.predictions.Rating] = null
  var predictor: (Int, Int) => Double = null

  override def beforeAll {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    spark = SparkSession
      .builder()
      .master("local[1]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    data = load(spark, dataPath, separator).collect()

    println("Loading personal data from: " + personalPath)
    val personalFile = spark.sparkContext.textFile(personalPath)
    personal = personalFile
      .map(l => {
        val cols = l.split(",").map(_.trim)
        if (cols(0) == "id")
          Rating(944, 0, 0.0)
        else if (cols.length < 3)
          Rating(944, cols(0).toInt, 0.0)
        else
          Rating(944, cols(0).toInt, cols(2).toDouble)
      })
      .filter(r => r.rating != 0)
      .collect()

    //predictor trained on data augmented with personal data. For k = 300.
    predictor = predictorKnn(data ++ personal, 300)

  }

  // All the functions definitions for the tests below (and the tests in other suites)
  // should be in a single library, 'src/main/scala/shared/predictions.scala'.
  //
  test("Prediction for user 1 of item 1") {
    assert(within(predictor(1, 1), 4.132180229734752, 0.0001))
  }

  test("Top 3 recommendations for user 944") {
    val recommendations = getTop3RecommandedMovie(
      data.map(_.item).toSet.diff(personal.map(_.item).toSet).toSeq,
      predictor
    )
    assert(recommendations(0)._1 == 119)
    assert(within(recommendations(0)._2, 5.0, 0.0001))
    assert(recommendations(1)._1 == 814)
    assert(within(recommendations(1)._2, 5.0, 0.0001))
    assert(recommendations(2)._1 == 1189)
    assert(within(recommendations(2)._2, 5.0, 0.0001))
  }

}
