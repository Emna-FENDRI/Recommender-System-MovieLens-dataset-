package predict

import org.rogach.scallop._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default = Some("\t"))
  val num_measurements = opt[Int](default = Some(0))
  val json = opt[String]()
  verify()
}

object Baseline extends App {
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

  // For these questions, data is collected in a scala Array
  // to not depend on Spark
  println("Loading training data from: " + conf.train())
  val train = load(spark, conf.train(), conf.separator()).collect()
  println("Loading test data from: " + conf.test())
  val test = load(spark, conf.test(), conf.separator()).collect()

  def printToFile(content: String, location: String = "./answers.json") =
    Some(new java.io.PrintWriter(location)).foreach { f =>
      try {
        f.write(content)
      } finally { f.close }
    }

  val globalAvgMeasurements = (1 to conf.num_measurements()).map(x =>
    timingInMs(() => {
      getMAE(predictorGlobAvg(train), test)
    })
  )
  val userBasedMeasurements = (1 to conf.num_measurements()).map(x =>
    timingInMs(() => {
      getMAE(predictorUserAvg(train), test)
    })
  )
  val itemBasedMeasurements = (1 to conf.num_measurements()).map(x =>
    timingInMs(() => {
      getMAE(predictorItemAvg(train), test)
    })
  )
  val baselineMeasurements = (1 to conf.num_measurements()).map(x =>
    timingInMs(() => {
      getMAE(predictorBaseline(train), test)
    })
  )
  val globalAvgTimings =
    globalAvgMeasurements.map(t => t._2)
  val userBasedTimings =
    userBasedMeasurements.map(t => t._2)
  val itemBasedTimings =
    itemBasedMeasurements.map(t => t._2)
  val baselineTimings =
    baselineMeasurements.map(t => t._2)

  //B1
  // When computing the item average for items that do not have ratings in the training set, use the global average
  // When making predictions for items that are not in the training set, use the user average if defined, otherwise the global average.

  println("global average")
  val globalAvg = cmptGlobAvg(train)
  println("user average")
  val usersAvg = computeUserAvg(train)
  val user1Avg = usersAvg.getOrElse(1, globalAvg)

  println("item average")
  val item1Stats = train.filter(_.item == 1).foldLeft((0.0, 0.0)) {
    (acc, curr) => (acc._1 + curr.rating, acc._2 + 1)
  }
  val item1Avg = item1Stats._1 / item1Stats._2

  val itemsAvgDev = computeAvgDevs(train, usersAvg, globalAvg)
  val item1AvgDev = itemsAvgDev.getOrElse(1, globalAvg)

  conf.json.toOption match {
    case None => ;
    case Some(jsonFile) => {
      var answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "B.1" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Num(globalAvg), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(user1Avg), // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(item1Avg), // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(
            item1AvgDev
          ), // Datatype of answer: Double
          "5.PredUser1Item1" -> ujson.Num(
            predictorBaseline(train)(1, 1)
          ) // Datatype of answer: Double
        ),
        "B.2" -> ujson.Obj(
          "1.GlobalAvgMAE" -> ujson.Num(
            getMAE(predictorGlobAvg(train), test)
          ), // Datatype of answer: Double
          "2.UserAvgMAE" -> ujson.Num(
            getMAE(predictorUserAvg(train), test)
          ), // Datatype of answer: Double
          "3.ItemAvgMAE" -> ujson.Num(
            getMAE(predictorItemAvg(train), test)
          ), // Datatype of answer: Double
          "4.BaselineMAE" -> ujson.Num(
            getMAE(predictorBaseline(train), test)
          ) // Datatype of answer: Double
        ),
        "B.3" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(
              mean(globalAvgTimings)
            ), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(
              std(globalAvgTimings)
            ) // Datatype of answer: Double
          ),
          "2.UserAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(
              mean(userBasedTimings)
            ), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(
              std(userBasedTimings)
            ) // Datatype of answer: Double
          ),
          "3.ItemAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(
              mean(itemBasedTimings)
            ), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(
              std(itemBasedTimings)
            ) // Datatype of answer: Double
          ),
          "4.Baseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(
              mean(baselineTimings)
            ), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(
              std(baselineTimings)
            ) // Datatype of answer: Double
          )
        )
      )

      val json = ujson.write(answers, 4)
      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json.toString, jsonFile)
    }
  }

  println("")
  spark.close()
}
