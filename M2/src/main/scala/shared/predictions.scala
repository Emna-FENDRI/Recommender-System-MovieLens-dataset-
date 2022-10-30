package shared

import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkContext

package object predictions {
  // ------------------------ For template
  case class Rating(user: Int, item: Int, rating: Double)

  def timingInMs(f: () => Double): (Double, Double) = {
    val start = System.nanoTime()
    val output = f()
    val end = System.nanoTime()
    return (output, (end - start) / 1000000.0)
  }

  def toInt(s: String): Option[Int] = {
    try {
      Some(s.toInt)
    } catch {
      case e: Exception => None
    }
  }

  def mean(s: Seq[Double]): Double =
    if (s.size > 0) s.reduce(_ + _) / s.length else 0.0

  def std(s: Seq[Double]): Double = {
    if (s.size == 0) 0.0
    else {
      val m = mean(s)
      scala.math.sqrt(
        s.map(x => scala.math.pow(m - x, 2)).sum / s.length.toDouble
      )
    }
  }

  def load(
      path: String,
      sep: String,
      nbUsers: Int,
      nbMovies: Int
  ): CSCMatrix[Double] = {
    val file = Source.fromFile(path)
    val builder = new CSCMatrix.Builder[Double](rows = nbUsers, cols = nbMovies)
    for (line <- file.getLines) {
      val cols = line.split(sep).map(_.trim)
      toInt(cols(0)) match {
        case Some(_) =>
          builder.add(cols(0).toInt - 1, cols(1).toInt - 1, cols(2).toDouble)
        case None => None
      }
    }
    file.close
    builder.result()
  }

  def loadSpark(
      sc: org.apache.spark.SparkContext,
      path: String,
      sep: String,
      nbUsers: Int,
      nbMovies: Int
  ): CSCMatrix[Double] = {
    val file = sc.textFile(path)
    val ratings = file
      .map(l => {
        val cols = l.split(sep).map(_.trim)
        toInt(cols(0)) match {
          case Some(_) =>
            Some(
              ((cols(0).toInt - 1, cols(1).toInt - 1), cols(2).toDouble)
            ) // ? -1 as index starts from 0 ?
          case None => None
        }
      })
      .filter({
        case Some(_) => true
        case None    => false
      })
      .map({
        case Some(x) => x
        case None    => ((-1, -1), -1)
      })
      .collect()

    val builder = new CSCMatrix.Builder[Double](rows = nbUsers, cols = nbMovies)
    for ((k, v) <- ratings) {
      v match {
        case d: Double => {
          val u = k._1
          val i = k._2
          builder.add(u, i, d)
        }
      }
    }
    return builder.result
  }

  // Create partitions 
  def partitionUsers(
      nbUsers: Int,
      nbPartitions: Int,
      replication: Int
  ): Seq[Set[Int]] = {
    val r = new scala.util.Random(1337)
    val bins: Map[Int, collection.mutable.ListBuffer[Int]] =
      (0 to (nbPartitions - 1)) 
        .map(p => (p -> collection.mutable.ListBuffer[Int]()))
        .toMap // initialize empty maps for each partition
    (0 to (nbUsers - 1)).foreach(u => {
      val assignedBins = r.shuffle(0 to (nbPartitions - 1)).take(replication)
      for (b <- assignedBins) {
        bins(b) += u
      }
    })
    bins.values.toSeq.map(_.toSet)
  }

  /** @param x
    *   Double
    * @param avg_rating_u
    *   Double : user_u’s average rating
    * @return
    *   Double : Eq(3) = The scale specific to a user’s average rating
    */
  def scale(x: Double, avr_rating_u: Double): Double =
    if (x > avr_rating_u) 5 - avr_rating_u
    else if (x < avr_rating_u) avr_rating_u - 1
    else 1

  /** @param predictor
    *   function that takes a user and an item as arguments and returns the
    *   prediction
    * @param test
    *   test Set
    * @return
    *   MAE returns the Mean Absolute Error
    */
  def getMAE(
      predictor: ((Int, Int) => Double),
      test: CSCMatrix[Double]
  ): Double = {
    var acc = 0.0
    for ((k, v) <- test.activeIterator) {
      acc += (predictor(k._1 + 1, k._2 + 1) - v).abs
    }

    acc / test.activeIterator.size
  }

  /**
  * @param train CSC Matrix training Set
  * @return Double :  Global average from ratings of the given training set
  */
  def computeGlobalAverage(
      train: CSCMatrix[Double]
  ): Double = {
    val nbRatings = train.activeIterator.size
    var sum = 0.0
    for ((k, v) <- train.activeIterator) {
      sum += v
    }
    sum / nbRatings
  }

  
 /**
  * @param nbUsers Total number of users
  * @param train CSC Matrix training Set
  * 
  * @return Dense Vector containing the average rating for each user
  */
  def computeUsersAvg(
      nbUsers: Int,
      train: CSCMatrix[Double]
  ): DenseVector[Double] = {

    val sumUsers = DenseVector.zeros[Double](nbUsers)
    val ratingsPerUser = DenseVector.zeros[Double](nbUsers)

    for ((k, v) <- train.activeIterator) {
      sumUsers(k._1) += v
      ratingsPerUser(k._1) += 1
    }
    // assume at least one rating per user
    sumUsers / ratingsPerUser
  }

  /**
   * @param nbUsers Total number of users
   * @param nbItems Total number of items
   * @param normalizedDev CSCMatrix[Double] Matrix containing the normalized deviation for each rating 
   *                      REMINDER :the normalized deviation of a given rating of item_id by user_id. Eq(2) from Milestone 1
   *                      = (r_u_i - useArvg)/scale(r_u_i, useArvg)
   * 
   * @return Dense Vector containing the value of the denominator needed to compute the preprocessed ratings as explained in part 5.2 of Milestone 1 
   *        This value is specific to each user and independent of the item (of the corresponding rating we want to preprocess)
  */
  def computeDenominator(
      nbUsers: Int,
      nbItems: Int,
      normalizedDev: CSCMatrix[Double]
  ): DenseVector[Double] = {
    val r_u_i_denominator = DenseVector.zeros[Double](nbUsers)
    for (x <- 0 to nbUsers - 1) {
      r_u_i_denominator(x) =
        (sqrt(sum(pow(normalizedDev(x, 0 to nbItems - 1).t, 2))))
    }
    r_u_i_denominator
  }

  /** @param nbUsers Total number of users
    * @param nbItems Total number of items
    * @param usersAvg Dense Vector containing the average rating for each user
    * @param train training Set
    * 
    * @return CSCMatrix[Double] Matrix containing the normalized deviation for each rating
    *         REMINDER :the normalized deviation of a given rating of item_id by user_id. Eq(2) from Milestone 1
    *         = (r_u_i - useArvg)/scale(r_u_i, useArvg)
    */

  def computeNormalizedDev(
      nbUsers: Int,
      nbItems: Int,
      usersAvg: DenseVector[Double],
      train: CSCMatrix[Double]
  ): CSCMatrix[Double] = {

    val normalizedDev = new CSCMatrix.Builder[Double](nbUsers, nbItems)

    for ((k, v) <- train.activeIterator) {
      normalizedDev.add(
        k._1,
        k._2,
        (v - usersAvg(k._1)) / shared.predictions.scale(v, usersAvg(k._1))
      )
    }
    normalizedDev.result
  }


  /** @param nbUsers Total number of users
    * @param nbItems Total number of items
    * @param normalizedDev CSCMatrix[Double] Matrix containing the normalized deviation for each rating
    * @param rui_denominator  Dense Vector containing the value of the denominator needed to compute the preprocessed ratings as explained in part 5.2 of Milestone 1 
    * 
    * @return Matrix containing all preprocessed ratings 
    */
  def computeRUI(
      nbUsers: Int,
      nbItems: Int,
      normalizedDev: CSCMatrix[Double],
      rui_denominator: DenseVector[Double]
  ): CSCMatrix[Double] = {
    val RUI = new CSCMatrix.Builder[Double](nbUsers, nbItems)
    for ((k, v) <- normalizedDev.activeIterator) {

      RUI.add(
        k._1,
        k._2,
        v / rui_denominator(k._1)
      )

    }

    RUI.result
  }


  /***
   *  NOTE  : This technique will cause java heap error if we train on 1m dataSet, but works good on 100k 
   *          We will not use this method in our general implementation.
   * 
   * @param RUI : Matrix of preprocessed ratings 
   * @return Similarity Matrix  
   */
  def computeSimCSCMatrix(
      RUI: CSCMatrix[Double]
  ): CSCMatrix[Double] = {
    RUI * RUI.t
  }

  /***
   * @param RUI : Matrix of preprocessed ratings 
   * @return Similarity Matrix 
   */ 
  def computeSimCSCMatrixLowHeap(
      RUI: CSCMatrix[Double]
  ): CSCMatrix[Double] = {

    val nUser = RUI.rows
    val nCols = RUI.cols
    val simMatrixBuilder = new CSCMatrix.Builder[Double](nUser, nUser)
    val RUI_T = RUI.t
    for (i <- 0 to nUser - 1) {
      val x = RUI * RUI(i, 0 to nCols - 1).t
      for (j <- 0 to nUser - 1) {
        simMatrixBuilder.add(i, j, x(j))
      }
    }
    simMatrixBuilder.result()
  }

  /** @param simMatrix : Similarity Matrix with all similarities between users
    * @param k 
    *
    * @return knn Matrix
    *   a CSCMatrix of shape (nbUsers*nbUsers) : row i stores the similarities of the knn of user i 
    */
  def getNeighbours(simMatrix: CSCMatrix[Double], k: Int): CSCMatrix[Double] = {
    val nbUsers = simMatrix.rows
    val neighboursMatrixBuilder =
      new CSCMatrix.Builder[Double](nbUsers, nbUsers)

    for (i <- 0 to nbUsers - 1) { // for each user get topk
      simMatrix.update(i, i, 0.0)
      val slicedVec = simMatrix(
        i,
        0 to nbUsers - 1
      ) //  i_th row of the simMatrix : i.e. all similarities between user i and all other users
      var neighbours = argtopk(slicedVec.t, k) // returns index of users i.e. index wrt slicedVec (in [0, nbUsers-1])
      val temp = neighbours
        .filter(_ != i)
        .map(x => {
          neighboursMatrixBuilder.add(i, x, simMatrix(i, x)) // Fill in resulting Matrix with topk similarities
        }) 
    }
    neighboursMatrixBuilder.result()
  }

  /** Compute the user weighted sum for a given item and user 
    * @param train
    * @param userId
    * @param itemId
    * @param nUsers
    * @param normalizedDev
    * @param neighboursMatrix knn similarity Matrix : a CSCMatrix of shape (nbUsers*nbUsers) : row i stores the similarities of the knn of user i 
    * 
    * @return
    *   Double : user weighted sum
    */
  def computeUserWeightedSum(
      train: CSCMatrix[Double],
      userId: Int,
      itemId: Int,
      nUsers: Int,
      normalizedDev: CSCMatrix[Double],
      neighboursMatrix: CSCMatrix[Double]
  ): Double = {

    val kNeighbours =
      neighboursMatrix(userId - 1, 0 to nUsers - 1) // k similarities here

    val filteredNeighbours = DenseVector.zeros[Double](nUsers)

    for ((k, v) <- (kNeighbours.t).activeIterator) {
      if (train(k, itemId - 1) != 0.0) {
        filteredNeighbours(k) = kNeighbours(k)
      }
    }

    val denominator = sum(filteredNeighbours.map(_.abs))
    if (denominator == 0) {
      return 0.0
    }

    val sliceNormalizedDev = normalizedDev(0 to nUsers - 1, itemId - 1)
    val numerator = (filteredNeighbours.t) * sliceNormalizedDev
    numerator / denominator
  }

    /** Compute the prediction of a user for a given item
    *
    * @param train
    * @param userId
    *   : the user for which we want to predict the rating
    * @param itemId
    *   : the item for which we want to predict the rating
    * @param usersAvg
    *   : the average rating of the users
    * @param globalAvg
    *   : the global average rating
    * @param normalizedDev
    *   : the map that contains the normalized deviation of the items
    * 
    * @param nUsers
    * @param neighboursMatrix knn Similarity Matrix : a CSCMatrix of shape (nbUsers*nbUsers) : row i stores the similarities of the knn of user i 
    * @return 
    *   the prediction value p_{u,i}
    */
  def prediction(
      train: CSCMatrix[Double],
      userId: Int,
      itemId: Int,
      usersAvg: DenseVector[Double],
      globalAvg: Double,
      normalizedDev: CSCMatrix[Double],
      nUsers: Int,
      neighboursMatrix: CSCMatrix[Double]
  ): Double = {
    val avg = usersAvg(userId - 1)
    if (avg != 0.0) {

      var weightedSum = computeUserWeightedSum(
        train,
        userId,
        itemId,
        nUsers,
        normalizedDev,
        neighboursMatrix
      )

      avg + weightedSum * shared.predictions.scale(avg + weightedSum, avg)
    } else {
      globalAvg
    }
  }


//---------------------------------------------------------------------------------------------------PART 1 OPTIMIZING WITH BREEZE ---------------------------------------------------------------------------------------------------------------------------
  /** @param train
    *   training set
    * @param k
    *   : number of neighbors to consider
    * @return:
    *   (Int, Int) => Double: returns the predictor function that takes a user
    *   and an item as arguments and returns the prediction as explained in
    *   Eq(8) of Milestone 1
    */
  def predictorKnn(
      train: CSCMatrix[Double],
      k: Int
  ): ((Int, Int) => Double) = {

    // Necessary Precomputations
    val nbUsers = train.rows
    val nbItems = train.cols
    val globalAverage = computeGlobalAverage(train)
    val usersAvg = computeUsersAvg(nbUsers, train)
    val normalizedDev = computeNormalizedDev(nbUsers, nbItems, usersAvg, train)
    val RUI_denominator = computeDenominator(
      nbUsers,
      nbItems,
      normalizedDev
    ) // as denom is specific to each user and independent of the currend item : stored in a vector (1 value for each user)
    val RUI = computeRUI(
      nbUsers,
      nbItems,
      normalizedDev,
      RUI_denominator
    ) // there is an r_ui for each given rating : will return  a CSCMatrix[Double] : of shape user*item
    val simMatrix = computeSimCSCMatrixLowHeap(RUI)
    val neighboursMatrix = getNeighbours(simMatrix, k)

    // RETURN
    (userId: Int, itemId: Int) => {
      prediction(
        train,
        userId,
        itemId,
        usersAvg,
        globalAverage,
        normalizedDev,
        nbUsers,
        neighboursMatrix
      )
    }
  }

//--------------------------------------------------------------------------PART 2 PARALLEL (EXACT) K-NN COMPUTATIONS WITH REPLICATED RATINGS -----------------------------------------------------------------------------------------------------------------------------------
 
   /** @param train
    *   training set
    * @param k
    *   : number of neighbors to consider
    * @param sc 
    * @return:
    *   (Int, Int) => Double: returns the predictor function that takes a user and an item as arguments and returns the prediction 
    */
  def parallelPredictorKnn(
      train: CSCMatrix[Double],
      sc: SparkContext,
      k: Int
  ): ((Int, Int) => Double) = {
    val nbUsers = train.rows
    val nbItems = train.cols
    val globalAverage = computeGlobalAverage(train)
    val usersAvg = computeUsersAvg(nbUsers, train)
    val normalizedDev = computeNormalizedDev(nbUsers, nbItems, usersAvg, train)
    
    val neighboursMatrix = buildExactSimMatrix(train,sc,nbUsers,nbItems,k,globalAverage,usersAvg,normalizedDev)

    // RETURN
    (userId: Int, itemId: Int) => {
      prediction(
        train,
        userId,
        itemId,
        usersAvg,
        globalAverage,
        normalizedDev,
        // simMatrix,
        nbUsers,
        neighboursMatrix
      )
    }
  }

 
  /***
   * This is the procedure of the pseudoCode given in Part 4 of Milestone 2
   * @param train
   * @param sc
   * @param k
   * @param nbUsers
   * @param rui Preprocessed ratings 
   * 
   * @return knn Similarity Matrix CSCMatrix[Double] (user*user) 
   */ 
  def parallelKnnComputations(
      train: CSCMatrix[Double],
      sc: SparkContext,
      k: Int,
      nbUsers: Int,
      rui: CSCMatrix[Double]
  ): CSCMatrix[Double] = {
    // We now broadcast the RUIs values to each worker
    val br = sc.broadcast(rui)

    //topk is a procedure that returns the k-NN for a given  user u
    def topk(indexUser_u: Int): (Int, IndexedSeq[(Int, Double)]) = {
      val ruiMatrix = br.value
      val nbItems = ruiMatrix.cols
      val rui_u1 = ruiMatrix(
        indexUser_u,
        0 to nbItems - 1
      ) // rui values for user's index u
      // compute sim for user u with all other users
      val similarities_u1 = (ruiMatrix * (rui_u1.t))

      return (
        indexUser_u,
        argtopk(similarities_u1, k + 1)
          .filter(_ != indexUser_u)
          .map(index_u2 => (index_u2, similarities_u1(index_u2)))
      )

    }


    val topks = sc
      .parallelize(0 to nbUsers - 1)
      .map(u => topk(u))
      .collect

    val builder = new CSCMatrix.Builder[Double](nbUsers, nbUsers)

    topks.map(tuple =>
      tuple._2.map(sim => builder.add(tuple._1, sim._1, sim._2))
    )

    val knnMatrix = builder.result()
    knnMatrix

  }



  /** Build the similarity using the exact method
    *
    * @param train
    * @param sc
    *   : SparkContext
    * @param nUsers
    * @param nMovies
    * @param k
    *   : number of neighbors
    * @param globalAverage
    * @param usersAvg
    * @param normalizedDev
    * 
    * @return
    *   : SCSMatrix[Double] the knn similarity matrix : a CSCMatrix of shape (nbUsers*nbUsers) : row i stores the similarities of the knn of user i 
    */
  def buildExactSimMatrix(
      train: CSCMatrix[Double],
      sc: SparkContext,
      nUsers: Int,
      nMovies: Int,
      k: Int,
      globalAverage : Double,
      usersAvg: DenseVector[Double],
      normalizedDev : CSCMatrix[Double]
  ): CSCMatrix[Double] = {

    val rui_denominator = computeDenominator(
      nUsers,
      nMovies,
      normalizedDev
    ) // as denom is specific to each user and independent of the current item : stored in a vector (1 value for each user)
    val rui = computeRUI(
      nUsers,
      nMovies,
      normalizedDev,
      rui_denominator
    ) // there is an r_ui for each given rating : will return  a CSCMatrix[Double] : of shape user*item

    parallelKnnComputations(train, sc, k, nUsers, rui)

  }

//---------------------------------------------------------------------------PART 3 DISTRIBUTED APPROXIMATE k-NN----------------------------------------------------------------------------------------------------------------------------------
  
  /**
   * NOTE : We perform the partitioning here so we can set the parameters (number of partitions and replication factor) easily when calling this function.
   * @param train
   * @param k
   * @param sc
   * @param conf_users
   * @param conf_movies
   * @param nPartitions
   * @param nRepetitions 
   * 
   * @return (Int, Int) => Double: returns the predictor function that takes a user and an item as arguments and returns the prediction 
  */
  def predictorKnnApproximate(
      train: CSCMatrix[Double],
      k: Int,
      sc: SparkContext,
      conf_users: Int,
      conf_movies: Int,
      nPartitions: Int,
      nRepetitions: Int
  ): ((Int, Int) => Double) = {
    val globalAverage = computeGlobalAverage(train)
    val usersAvg = computeUsersAvg(conf_users, train)
    val normalizedDev =computeNormalizedDev(conf_users, conf_movies, usersAvg, train)


      val knnMatrix = buildApproximateSimMatrix(
      train,
      nPartitions,
      nRepetitions,
      sc,
      conf_users,
      conf_movies,
      k,
      globalAverage,
      usersAvg,
      normalizedDev)

    // Return Pair
    (userId: Int, itemId: Int) => {
      prediction(
        train,
        userId,
        itemId,
        usersAvg,
        globalAverage,
        normalizedDev,
        conf_users,
        knnMatrix
      )
    }
  }

  /**
   * @param localSims an Array of knn similarity matrices computed by each partition
   * @return 1 Matrix with all the similarities
  */
  def mergeLocalSimilarity(
      localSims: Array[CSCMatrix[Double]]
  ): CSCMatrix[Double] = {
    // for each user: (each line of the matrix)
    // put all sim of this user inside big list (userId, toId).
    // then groupBy _._2
    // then pick max

    val builder =
      new CSCMatrix.Builder[Double](localSims.head.rows, localSims.head.cols)
    for (i <- 0 to localSims.head.rows - 1) {
      for (j <- 0 to localSims.head.rows - 1) {
        builder.add(i, j, localSims.map(_(i, j)).max)
      }
    }
    builder.result()
  }



  /** Build the similarity using the approximate method
    *
    * @param train
    * @param nPartitions
    * @param nRepetitions
    * @param sc
    *   : SparkContext
    * @param nUsers
    * @param nMovies
    * @param k 
    * @param globalAverage
    * @param usersAvg
    * @param normalizedDev
    * 
    * @return
    *   : CSCMatrix[Double], the knn similarity matrix
    */
  def buildApproximateSimMatrix(
      train: CSCMatrix[Double],
      nPartitions: Int,
      nRepetitions: Int,
      sc: SparkContext,
      nUsers: Int,
      nMovies: Int,
      k: Int,
      globalAverage : Double,
      usersAvg: DenseVector[Double],
      normalizedDev : CSCMatrix[Double]
  ): CSCMatrix[Double] = {

    val userPartition = partitionUsers(nUsers, nPartitions, nRepetitions)

    // Compute preprocessed ratings

    val rui_denominator =
      computeDenominator(nUsers, nMovies, normalizedDev)
    // compute rui for each rating:  will return  a CSCMatrix[Double] : of shape user*item
    val rui =
      computeRUI(nUsers, nMovies, normalizedDev, rui_denominator)

    // Given the partition of users, build for each partition a matrix filled with processed ratings of size nusers*nusers
    // parallelize Seq[CSCMAtrix]

    val toParallelize = userPartition
      .map(partition => {
        val maskBuilder =
          new CSCMatrix.Builder[Double](nUsers, nUsers)
        partition.toList.map(user_index =>
          maskBuilder.add(user_index, user_index, 1.0)
        )
        val mask = maskBuilder.result()
        val localRatings = mask * rui //

        localRatings
      })
      .toList

    val parallelized =
      sc.parallelize(toParallelize, nPartitions)

    val output = parallelized
      .map(localRuiMatrix => {
        val localSim = computeSimCSCMatrix(localRuiMatrix)

        val localNeighboursMatrix = getNeighbours(localSim, k) // local knn
        localNeighboursMatrix
      })
      .collect()

    var neighboursMatrix_merged = mergeLocalSimilarity(output) // Merge the collected knn Similarity matrices from each partition
    getNeighbours(neighboursMatrix_merged, k) // filter to get top k across partitions
  }

}
