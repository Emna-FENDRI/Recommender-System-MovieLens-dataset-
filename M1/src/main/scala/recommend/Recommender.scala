package recommend

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import shared.predictions._

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val data = opt[String](required = true)
  val personal = opt[String](required = true)
  val separator = opt[String](default = Some("\t"))
  val json = opt[String]()
  verify()
}

object Recommender extends App {
  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = SparkSession
    .builder()
    .master("local[1]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  println("")
  println("******************************************************")

  var conf = new Conf(args)
  println("Loading data from: " + conf.data())
  val data = load(spark, conf.data(), conf.separator()).collect()
  //assert(data.length == 100000, "Invalid data")

  println("Loading personal data from: " + conf.personal())
  val personalFile = spark.sparkContext.textFile(conf.personal())
  val personal = personalFile
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
  val movieNames = personalFile
    .map(l => {
      val cols = l.split(",").map(_.trim)
      if (cols(0) == "id") (0, "header")
      else (cols(0).toInt, cols(1).toString)
    })
    .collect()
    .toMap

  // Save answers as JSON
  def printToFile(content: String, location: String = "./answers.json") =
    Some(new java.io.PrintWriter(location)).foreach { f =>
      try {
        f.write(content)
      } finally { f.close }
    }

  //R.1
  //first let's add the new user's data to the dataset:
  val augmentedData = data ++ personal

  //next let's train a knn predictor with k = 300
  val predictor = predictorKnn(augmentedData, 300)

  //R.2
  //let's predict the ratings of the new user for the movies he hasn't seen yet
  val movieToPredict =
    data.map(_.item).toSet.diff(personal.map(_.item).toSet).toSeq

  val best3Movies = getTop3RecommandedMovie(movieToPredict, predictor)

  conf.json.toOption match {
    case None => ;
    case Some(jsonFile) => {
      val answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "data" -> conf.data(),
          "personal" -> conf.personal()
        ),
        "R.1" -> ujson.Obj(
          "PredUser1Item1" -> ujson.Num(
            predictor(1, 1)
          ) // Prediction for user 1 of item 1
        ),
        // IMPORTANT: To break ties and ensure reproducibility of results,
        // please report the top-3 recommendations that have the smallest
        // movie identifier.

        "R.2" -> List(best3Movies(0), best3Movies(1), best3Movies(2)).map(x =>
          ujson.Arr(x._1, movieNames(x._1), x._2)
        )
      )
      val json = write(answers, 4)

      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json, jsonFile)
    }
  }

  println("")
  spark.close()
}
