import org.rogach.scallop._
import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import ujson._

package economics {

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val json = opt[String]()
    verify()
  }

  object Economics {
    def main(args: Array[String]) = {
      println("")
      println("******************************************************")

      var conf = new Conf(args)

      // Save answers as JSON
      def printToFile(content: String, location: String = "./answers.json") =
        Some(new java.io.PrintWriter(location)).foreach { f =>
          try {
            f.write(content)
          } finally { f.close }
        }

      // E.1 excluding the energy cost + maintenance, just divide cost of ICC / renting per day
      val icc_cost = 38600

      // E.2
      val cost_ram = 1.6 * math.pow(10, -7) * 3600 * 24 * 32
      val cost_cpu = 1.14 * math.pow(10, -6) * 3600 * 24

      val container_daily_cost = cost_ram + cost_cpu

      val daily_4RPI_idle_nrj_cost = 0.25 * 4 * (0.003) * 24
      val daily_4RPI_cmpt_nrj_cost = 0.25 * 4 * (0.004) * 24

      val rpi_cost = 108.48

      val n_rpi_for_icc = (icc_cost / rpi_cost).floor

      conf.json.toOption match {
        case None => ;
        case Some(jsonFile) => {

          val answers = ujson.Obj(
            "E.1" -> ujson.Obj(
              "MinRentingDays" -> ujson.Num(
                math.ceil(icc_cost / 20.4)
              ) // Datatype of answer: Double
            ),
            "E.2" -> ujson.Obj(
              "ContainerDailyCost" -> ujson.Num(container_daily_cost),
              "4RPisDailyCostIdle" -> ujson.Num(daily_4RPI_idle_nrj_cost),
              "4RPisDailyCostComputing" -> ujson.Num(daily_4RPI_cmpt_nrj_cost),
              "MinRentingDaysIdleRPiPower" -> ujson.Num(
                math.ceil(
                  (rpi_cost * 4) / (container_daily_cost - daily_4RPI_idle_nrj_cost)
                )
              ),
              "MinRentingDaysComputingRPiPower" -> ujson.Num(
                math.ceil(
                  (rpi_cost * 4) / (container_daily_cost - daily_4RPI_cmpt_nrj_cost)
                )
              )
            ),
            "E.3" -> ujson.Obj(
              "NbRPisEqBuyingICCM7" -> ujson.Num(n_rpi_for_icc),
              "RatioRAMRPisVsICCM7" -> ujson.Num(n_rpi_for_icc * 8 / (24 * 64)),
              "RatioComputeRPisVsICCM7" -> ujson.Num(
                n_rpi_for_icc / (2 * 14 * 4)
              ) // denominator: icc has (2) intel core * (14) physical core per intel core * (equivalent to throughput of (4) rpi)
            )
          )

          val json = write(answers, 4)
          println(json)
          println("Saving answers in: " + jsonFile)
          printToFile(json, jsonFile)
        }
      }

      println("")
    }
  }

}
