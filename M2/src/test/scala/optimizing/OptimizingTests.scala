package test.optimizing

import breeze.linalg._
import breeze.numerics._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import shared.predictions._
import test.shared.helpers._

class OptimizingTests extends AnyFunSuite with BeforeAndAfterAll {

  val separator = "\t"
  val train2Path = "data/ml-100k/u2.base"
  val test2Path = "data/ml-100k/u2.test"
  var train2: CSCMatrix[Double] = null
  var test2: CSCMatrix[Double] = null

  override def beforeAll =  {
    // For these questions, train and test are collected in a scala Array
    // to not depend on Spark
    train2 = load(train2Path, separator, 943, 1682)
    test2 = load(test2Path, separator, 943, 1682)
  }

  // I had to redefine them, as they weren't assigned in boforeAll, resulting in null pointer
  train2 = load(train2Path, separator, 943, 1682)
  test2 = load(test2Path, separator, 943, 1682)
  // only run on 100k, that's why we hardcoded nb users and nb movies
  val k = 10
  val nbUsers = 943
  val nbMovies = 1682

  val globalAverage = computeGlobalAverage(train2)
  val usersAvg = computeUsersAvg(nbUsers, train2)
  val normalizedDev = computeNormalizedDev(nbUsers, nbMovies, usersAvg, train2)
  val RUI_denominator = computeDenominator(nbUsers, nbMovies, normalizedDev)
  val RUI = computeRUI(nbUsers, nbMovies, normalizedDev, RUI_denominator)

  val simMatrixk10 = getNeighbours(computeSimCSCMatrix(RUI),k)

  val predictorK10 = predictorKnn(train2, k)

  // Provide tests to show how to call your code to do the following tasks.
  // Ensure you use the same function calls to produce the JSON outputs in
  // the corresponding application.
  // Add assertions with the answer you expect from your code, up to the 4th
  // decimal after the (floating) point, on data/ml-100k/u2.base (as loaded above).
  test("kNN predictor with k=10") {

    // Similarity between user 1 and itself
    assert(
      within(simMatrixk10(0,0), 0.0, 0.0001)
    )

    // Similarity between user 1 and 864
    assert(
      within(
        simMatrixk10(0,863),
        0.2423,
        0.0001
      )
    )

    // Similarity between user 1 and 886
    assert(
      within(simMatrixk10(0,885), 0.0, 0.0001)
    )

    // Prediction user 1 and item 1
    assert(within(predictorK10(1, 1), 4.3190, 0.0001))

    // Prediction user 327 and item 2
    assert(within(predictorK10(327, 2), 2.6994, 0.0001))

    // MAE on test2
    assert(within(getMAE(predictorK10, test2), 0.8287, 0.0001))
  }
}
