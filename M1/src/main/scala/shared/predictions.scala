package shared

import org.apache.spark.rdd.RDD

package object predictions {
  case class Rating(user: Int, item: Int, rating: Double)

  def timingInMs(f: () => Double): (Double, Double) = {
    val start = System.nanoTime()
    val output = f()
    val end = System.nanoTime()
    return (output, (end - start) / 1000000.0)
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

  def toInt(s: String): Option[Int] = {
    try {
      Some(s.toInt)
    } catch {
      case e: Exception => None
    }
  }

  def load(
      spark: org.apache.spark.sql.SparkSession,
      path: String,
      sep: String
  ): org.apache.spark.rdd.RDD[Rating] = {
    val file = spark.sparkContext.textFile(path)
    return file
      .map(l => {
        val cols = l.split(sep).map(_.trim)
        toInt(cols(0)) match {
          case Some(_) => {
            Some(Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble))
          }
          case None => None
        }
      })
      .filter({
        case Some(_) => true
        case None    => false
      })
      .map({
        case Some(x) => x
        case None    => Rating(-1, -1, -1)
      })
  }


  /** @param trainSet:
    *   Sequence of ratings
    * @return
    *   Global average from ratings of the given training set
    */
  def cmptGlobAvg(trainSet: Seq[Rating]): Double = {
    //compute the average of ratings over trainSet
    val res = trainSet.foldLeft((0.0, 0.0)) { (acc, y) =>
      {
        (acc._1 + y.rating, acc._2 + 1)
      }
    }
    res._1 / res._2
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



  /**
   * @param trainSet
   * @param user_id
   * @return average rating for a given user user_id
   */

  def userAvg(trainSet: Seq[Rating], userId: Int): Double = {
    mean(trainSet.filter(_.user == userId).map(_.rating))
  }

  /** @param trainSet
    * @param itemId
    * @return
    *   average rating for a given item itemId
    */
  def itemAvg(trainSet: Seq[Rating], itemId: Int): Double = {
    mean(trainSet.filter(_.item == itemId).map(_.rating))
  }

    /** @param dataSet
    * @return
    *   Map[Int, Seq[Int]]: a map where the key is user u and the value is a
    *   sequence of items rated by user u
    */
  def computeUserToItems(train: Seq[Rating]): Map[Int, Seq[Int]] = {
    train.groupBy(_.user).mapValues(_.map(_.item).distinct)
  }

    /** @param dataSet
    * @return
    *   Map[Int, Seq[Int]]: a map where the key is item i and the value is a
    *   sequence of the users that rated i
    */
  def computeItemToUsers(train: Seq[Rating]): Map[Int, Seq[Int]] = {
    train.groupBy(_.item).mapValues(_.map(_.user).distinct)
  }

    /** @param dataSet
    * @return
    *   Map[Int, Seq[Rating]]: a map where the key is user u and the value is a
    *   sequence of ratings for which the items where rated by user u
    */
  def getUserToCollection(dataSet: Seq[Rating]): Map[Int, Seq[Rating]] = {
    dataSet.groupBy(_.user)
  }

   //Return a map for collection of rakings wrt item i : Map[(itemi, collection)]
  /** @param dataSet
    * @return
    *   Map[Int, Seq[Rating]]: a map where the key is item i and the value is a
    *   sequence of ratings for which users rated item i
    */
  def getItemToCollection(dataSet: Seq[Rating]): Map[Int, Seq[Rating]] = {
    dataSet.groupBy(_.item)
  }

   /** Compute the average rating for each user, keeping an RDD format
    *
    * @param train
    *   The training data
    * @return
    *   A map of the average rating for each user
    */
  def computeUserAvg(train: Seq[Rating]): Map[Int, Double] = {
    train
      .groupBy(x => x.user)
      .mapValues({ x =>
        {
          val res = x.foldLeft((0.0, 0.0)) { (acc, y) =>
            {
              (acc._1 + y.rating, acc._2 + 1)
            }
          }
          res._1 / res._2
        }
      })
      .toMap
  }

  /** @param trainSet
    * @return
    *   Map[Int, Double] of (k=user, v=itemAvg) for each item of the given
    *   trainingSet
    */
  def computeItemAvg(trainSet: Seq[Rating]): Map[Int, Double] = {
    //group by item: returns a map (k=item, v=list of RatingRows)
    val itemToRatings = trainSet.groupBy(_.item)
    //transform: (k=item, v= itemAvg)
    itemToRatings.transform((item, ratings) => mean(ratings.map(_.rating)))
  }


  /** @param user_id
    * @param item_id
    * @param rating
    *   Double : rating of item_id by user_id
    * @param userAvgs
    *   Map[Int, Double] : map that stores the average rating for each user i.e.
    *   (k= user , v= userAvg)
    * @param globalAvg
    *   Double : global average rating of the underlying training set
    * @return
    *   the normalized deviation of a given rating of item_id by user_id. Eq(2)
    * = (r_u_i - useArvg)/scale(r_u_i, useArvg)
    */
  def cmptNormalizedDeviation(
      user_id: Int,
      item_id: Int,
      rating: Double,
      userAvgs: Map[Int, Double],
      globalAvg: Double
  ): Double = {
    (rating - userAvgs.getOrElse(user_id, globalAvg)) / scale(
      rating,
      userAvgs.getOrElse(user_id, globalAvg)
    )
  }

  /**
   * @param train  
   * @param usersAvg A map of the average rating for each user
   * @param globalAvg The value of the global average rating of the given training set
   * @return 
   */
  def buildNormalizedDevMap(
      train: Seq[Rating],
      usersAvg: Map[Int, Double],
      globalAvg: Double
  ): Map[(Int, Int), Double] = {
    train
      .map(elem =>
        ((elem.user, elem.item) ->
          cmptNormalizedDeviation(
            elem.user,
            elem.item,
            elem.rating,
            usersAvg,
            globalAvg
          ))
      )
      .toMap
  }


  /** Compute the avgDev for each item in the DataSet, and store it in a map
    * that the function will return
    * @param dataSet
    *   Seq[Rating]: Data Set on which we perform the computation
    * @param userAvgs
    *   Map[Int, Double] :
    * @param globalAvg
    *   Global average rating of the given data set
    * @return
    *   Map[Int, Double] : (k= item , v= avgDev)
    */
  def computeAvgDevs(
      dataSet: Seq[Rating],
      userAvgs: Map[Int, Double],
      globalAvg: Double
  ): Map[Int, Double] = {
    dataSet
      .groupBy(x => x.item)
      .mapValues({ x =>
        {
          val res = x.foldLeft((0.0, 0.0)) { (acc, y) =>
            {
              val avg = userAvgs.getOrElse(y.user, globalAvg)
              (acc._1 + (y.rating - avg) / scale(y.rating, avg), acc._2 + 1)
            }
          }
          res._1 / res._2
        }
      })
      .toMap
  }

  /** @param predictor
    *   function that takes a user and an item as arguments and returns the
    *   prediction
    * @param test
    *   test Set
    * @return
    *   MAE returns the Mean Absolute Error Eq(1)
    */
  def getMAE(
      predictor: ((Int, Int) => Double),
      test: Seq[shared.predictions.Rating]
  ): Double = {
    val acc = test.foldLeft((0.0, 0.0)) { (acc, x) =>
      (acc._1 + (predictor(x.user, x.item) - x.rating).abs, acc._2 + 1)
    }
    acc._1 / acc._2
  }


  //---------------Distributed Methods-----------------------------
  /** Compute the average rating for each user, keeping an RDD format
    *
    * @param train
    *   The training data
    * @return
    *   A map of the average rating for each user
    */
  def computeUserAvgsRDD(train: RDD[Rating]): Map[Int, Double] = {
    train
      .map(x => (x.user, (x.rating, 1)))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
      .mapValues(x => (x._1 / x._2))
      .collect
      .toMap
  }

  /** @param train:
    *   Sequence of ratings
    * @return
    *   Global average from ratings of the given training set
    */
  def cmptGlobAvgRDD(train: RDD[Rating]): Double = {
    val tup = train.aggregate((0.0, 0.0))(
      (acc, x) => (acc._1 + x.rating, acc._2 + 1),
      (y, z) => (y._1 + z._1, y._2 + z._2)
    )
    tup._1 / tup._2
  }

  /** @param predictor
    *   function that takes a user and an item as arguments and returns the
    *   prediction
    * @param test
    *   test Set
    * @return
    *   MAE returns the Mean Absolute Error Eq(1)
    */
  def getMaeRDD(
      predictor: ((Int, Int) => Double),
      test: RDD[Rating]
  ): Double = {
    val res = test.aggregate((0.0, 0.0))(
      (acc, x) =>
        (acc._1 + math.abs(predictor(x.user, x.item) - x.rating), acc._2 + 1),
      (y, z) => (y._1 + z._1, y._2 + z._2)
    )
    res._1 / res._2
  }

  /** @param trainSet
    * @param user_id
    * @return
    *   average rating for a given user user_id
    */
  def userAvgRDD(trainSet: RDD[Rating], user_id: Int): Double =
    trainSet.filter(_.user == user_id).map(_.rating).mean()

  /**
   * @param trainSet
   * @param item_id
   * @return average rating for a given item item_id
   */
  def itemAvgRDD(trainSet: RDD[Rating], item_id: Int): Double =
    trainSet.filter(_.item == item_id).map(_.rating).mean()



  /** Compute the avgDev for each item in the DataSet, and store it in a map
    * that the function will return
    * @param dataSet
    *   Seq[Rating]: Data Set on which we perform the computation
    * @param userAvgs
    *   Map[Int, Double] :
    * @param globalAvg
    *   Global average rating of the given data set
    * @return
    *   Map[Int, Double] : (k= item , v= avgDev)
    */
  def computeAvgDevsRDD(
      dataSet: RDD[Rating],
      // itemToCollection: Map[Int, Seq[Rating]],
      userAvgs: Map[Int, Double],
      globalAvg: Double
  ): Map[Int, Double] = {
    // val res = scala.collection.mutable.Map[Int, Double]()
    dataSet
      .groupBy(x => x.item)
      .mapValues({ x =>
        {
          val res = x.foldLeft((0.0, 0.0)) { (acc, y) =>
            {
              val avg = userAvgs.getOrElse(y.user, globalAvg)
              (acc._1 + (y.rating - avg) / scale(y.rating, avg), acc._2 + 1)
            }
          }
          res._1 / res._2
        }
      })
      .collect
      .toMap
  }

  //--------------- KNN -------------------------------------


  /** Compute the KNN similarity between two users
    *
    * @note
    *   if userId == otherUser, then this function always returns 0.0
    *
    * @param userId
    *   user id
    * @param otherUser
    *   user id
    * @param from
    *   :Seq[Int] : list of users
    * @param cosineSimilarity
    *   : Map[(Int, Int), Double] : cosine similarity
    * @param k
    *   : Int : number of neighbors
    * @return
    *   the similarity between the two users
    */
  def knnSimilarity(
      userId: Int,
      otherUser: Int,
      from: Seq[Int],
      cosineSimilarity: (Int, Int) => Double,
      k: Int
  ): Double = { 
    val candidates = from
      .filter(_ != userId) // no self similarity
      .map(user => (user, cosineSimilarity(userId, user)))
      .sortBy(-_._2)
      .take(k) // pick k closest users by decreasing similarity
      .map(_._1) // get back the ID of the users
    if (candidates.contains(otherUser)) {
      cosineSimilarity(userId, otherUser)
    } else {
      0.0
    }
  }

  /** Compute the user weighted sum for a given item and user eq(7)
    * @param neighbours Top k neighbours with their similarities the specific user
    * @param itemId
    * @param normalizedDev
    *   : Map[(Int, Int), Double] : (k= (user, item), v= normalized deviation)
    * @return
    *   Double : user weighted sum
    */
  def computeUserWeightedSum(
      neighbours: Seq[(Int, Double)],
      itemId: Int,
      normalizedDev: Map[(Int, Int), Double]
  ): Double = {
    val acc = neighbours.foldLeft((0.0,0.0)){
      (acc, x)=>{
        if(normalizedDev.contains((x._1, itemId))){
          (acc._1+ x._2 * normalizedDev.getOrElse((x._1, itemId),0.0), math.abs(x._2)+acc._2)
        }else{
          acc
        }
      }
    }
    if(acc._2==0.0){
      0.0
    }else{
      acc._1/acc._2
    }
  }

  /** Compute the prediction of a user for a given item
    *
    * @param neighbours Top k neighbours with their similarities the specific user
    * @param userId
    *   : the user for which we want to predict the rating
    * @param itemId
    *   : the item for which we want to predict the rating 
    * @param k
    *   : the number of neighbors to consider
    * @param usersAvg
    *   : the average rating of the users
    * @param globalAvg
    *   : the global average rating
    * @param normalizedDev
    *   : the map that contains the normalized deviation of the items
    * @return
    */
  def prediction(
      neighbours: Seq[(Int, Double)],
      userId: Int,
      itemId: Int,
      k: Int,
      usersAvg: Map[Int, Double],
      globalAvg: Double,
      normalizedDev: Map[(Int, Int), Double]
  ): Double = {
    usersAvg.get(userId) match {
      case Some(avg) =>
        var r_u_i = computeUserWeightedSum(
          neighbours,
          itemId,
          normalizedDev
        )
        return avg + r_u_i * scale(avg + r_u_i, avg)
      case None =>
        return globalAvg
    }
  }

  /** Compute the top 3 recommanded items from a set of movie
    *
    * @param fromMovies
    *   : Set[Int] : set of movies to recommend
    * @param predictor
    *   : (Int, Int) => Double : function to compute prediction for a user and
    *   an item
    * @return
    */
  def getTop3RecommandedMovie(
      fromMovies: Seq[Int],
      predictor: ((Int, Int) => Double)
  ): Seq[(Int, Double)] = {
    val predictions = fromMovies.map(m => (m, predictor(944, m)))

    val sort = {
      (x:(Int, Double),y:(Int, Double))=>{
        if(y._2==x._2){
          x._1<y._1
        }else{
          x._2>y._2
        }
      }
    }

    //sort predictions by rating
    predictions.sortWith(sort).take(3)
  }

  //--------------------Similarity functions ---------------------------------------

  /** @param train
    *   training set
    * @return:
    *   (Int, Int)=>Double : returns a function that computes the similarity
    *   between 2 users. This function will return a 1 similarity between any 2
    *   user.
    */
  def unitSimilarity(train: Seq[Rating]): (Int, Int) => Double = {
    (user_u: Int, user_v: Int) => 1.0
  }

  /** @param train
    *   training set
    * @return:
    *   (Int, Int)=>Double : returns a function that computes the similarity
    *   between 2 users. This function will return the jaccard similarity
    *   between any 2 user. This similarity consists in computing the ratio:
    *   (number of items rated by boths users user1 AND user2)/(number of items
    *   rated by either user1 OR user2)
    */
  def jaccardSimilarity(train: Seq[Rating]): (Int, Int) => Double = {
    val userToCollection = getUserToCollection(train)
    // RETURN
    (user_u: Int, user_v: Int) => {
      // AinterB divided by AunionB
      val itemsRatedBy_u =
        userToCollection.getOrElse(user_u, Nil).map(_.item).toSet
      val itemsRatedBy_v =
        userToCollection.getOrElse(user_v, Nil).map(_.item).toSet
      val intersectionItems_u_v = itemsRatedBy_u.intersect(itemsRatedBy_v)
      val intersectionSize =intersectionItems_u_v.size.toDouble
      val denom =(itemsRatedBy_v.size+itemsRatedBy_u.size-intersectionSize)
      if (denom != 0) {
        intersectionSize / denom
      } else {
        0.0
      }
    }
  }

  /** @param train
    *   training set
    * @return:
    *   (Int, Int)=>Double : returns a function that computes the similarity
    *   between 2 users. This function will return the cosine similarity between
    *   any 2 user as explained in Eq(6)
    */
  def cosineSimilarity(
      train: Seq[Rating]): (Int, Int) => Double = {
    val globalAvg = cmptGlobAvg(train)
    val userAvgs = computeUserAvg(train)
    val userToCollection = getUserToCollection(train)
    val normalizedDev = buildNormalizedDevMap(train, userAvgs, globalAvg)
    val collectionOfUserNorms = train
      .foldLeft(Map[Int, Double]()) { (acc, el) =>
        {
          val cur: Double = acc.getOrElse(el.user, 0.0)
          acc + (el.user -> (cur + math.pow(
            normalizedDev.getOrElse((el.user, el.item), 0.0),
            2
          )))
        }
      }
      .mapValues { x =>
        math.sqrt(x)
      }
    //val mapForRatings = train.map(x => (x.user, x.item) -> x.rating).toMap

    //RETURN
    (user_u: Int, user_v: Int) => {
      if (user_u == user_v) {
        1.0
      } else {
        val itemsRatedBy_u =
          userToCollection.getOrElse(user_u, Nil).map(_.item).toSet
        val itemsRatedBy_v =
          userToCollection.getOrElse(user_v, Nil).map(_.item).toSet
        val userNorm_u = collectionOfUserNorms.getOrElse(user_u, 0.0)
        val userNorm_v = collectionOfUserNorms.getOrElse(user_v, 0.0)

        val denominator = userNorm_u * userNorm_v
        if (itemsRatedBy_u.union(itemsRatedBy_v) == Nil || denominator == 0) {
          0.0
        } else {
          val intersectionItems_u_v = itemsRatedBy_u.intersect(itemsRatedBy_v)
          val numerator =
            intersectionItems_u_v.foldLeft(0: Double)((acc, item) => {
              val product =
                normalizedDev.getOrElse((user_u, item), 0.0) * normalizedDev(
                  (user_v, item)
                )
              acc + product
            })
          numerator / denominator
        }
      }
    }
  }

  //-----------------------Predictors-------------------------------------------------

  /** @param train
    *   training set
    * @return:
    *   (Int, Int) => Double: returns the predictor function that takes a user
    *   and an item as arguments and returns the prediction This function uses
    *   the baseline method as described in Eq(5)
    */
  def predictorBaselineRDD(train: RDD[Rating]): ((Int, Int) => Double) = {
    val globalAvg = cmptGlobAvgRDD(train)
    val userAvgs = computeUserAvgsRDD(train)
    val itemsAvgDev = computeAvgDevsRDD(train, userAvgs, globalAvg)

    val tempMap = scala.collection.mutable.Map[Int, Double]()
    var itemAvgDev = 0.0
    (userId: Int, itemId: Int) => {
      if (tempMap.contains(itemId)) {
        itemAvgDev = tempMap(itemId)
      } else {
        itemAvgDev = itemsAvgDev.getOrElse(itemId, 0)
        tempMap += (itemId -> itemAvgDev)
      }
      val userAvg: Double = userAvgs.getOrElse(userId, globalAvg)
      if (userAvg == globalAvg) {
        globalAvg
      } else {
        userAvg + itemAvgDev * scale((userAvg + itemAvgDev), userAvg)
      }
    }
  }

  /** @param train
    *   training set
    * @return:
    *   (Int, Int) => Double: returns the predictor function that takes a user
    *   and an item as arguments and returns the prediction This function uses
    *   the baseline method as described in Eq(5)
    */
  def predictorBaseline(
      train: Seq[shared.predictions.Rating]
  ): ((Int, Int) => Double) = {
    val globalAvg = cmptGlobAvg(train)

    val userAvgs = computeUserAvg(train)
    val itemsAvgDev = computeAvgDevs(train, userAvgs, globalAvg)

    val tempMap = scala.collection.mutable.Map[Int, Double]()
    var itemAvgDev = 0.0
    (userId: Int, itemId: Int) => {
      if (tempMap.contains(itemId)) {
        itemAvgDev = tempMap(itemId)
      } else {
        itemAvgDev = itemsAvgDev.getOrElse(itemId, 0)
        tempMap += (itemId -> itemAvgDev)
      }

      val userAvg = userAvgs.getOrElse(userId, globalAvg)
      if (userAvg == globalAvg) {
        globalAvg
      } else {
        userAvg + itemAvgDev * scale((userAvg + itemAvgDev), userAvg)
      }
    }
  }

  /** @param train
    *   training set
    * @return:
    *   (Int, Int) => Double: returns the predictor function that takes a user
    *   and an item as arguments and returns the prediction For a given user u,
    *   the predictor returns the value of the userAvg as a prediction for any
    *   item i. If u has no rating in the training set, we use the global
    *   average as prediction.
    */
  def predictorUserAvg(
      train: Seq[shared.predictions.Rating]
  ): (Int, Int) => Double = {
    //COMPUTE global average of training set
    val globalAvg = cmptGlobAvg(train)
    //COMPUTE map of (user, avgUser)
    val userAvgs = computeUserAvg(train)
    //Return prediction (u,i)=>Double
    (userId: Int, itemId: Int) => {
      userAvgs.getOrElse(userId, globalAvg)
    }

  }

  /** @param train
    *   training set
    * @return:
    *   (Int, Int) => Double: returns the predictor function that takes a user
    *   and an item as arguments and returns the prediction For a given item i,
    *   the predictor returns the value of the itemAvg as a prediction for any
    *   user u. If i has no rating in the training set, we use the global
    *   average as prediction.
    */
  def predictorItemAvg(
      train: Seq[shared.predictions.Rating]
  ): ((Int, Int) => Double) = {
    //COMPUTE global average of training set
    val globalAvg = cmptGlobAvg(train)
    //COMPUTE map of (user, avgUser)
    val itemAvgs = computeItemAvg(train)
    //Return prediction (u,i)=>Double
    (userId: Int, itemId: Int) => {
      itemAvgs.getOrElse(itemId, globalAvg)
    }
  }

  /** @param train
    *   training set
    * @return:
    *   (Int, Int) => Double: returns the predictor function that takes a user
    *   and an item as arguments and returns the prediction The predictor
    *   returns the value of the global Avg of the given training set as a
    *   prediction for any item i and user u.
    */
  def predictorGlobAvg(
      train: Seq[shared.predictions.Rating]
  ): ((Int, Int) => Double) = {
    //COMPUTE global average of training set
    val globalAvg = cmptGlobAvg(train)
    return (userId: Int, itemId: Int) => {
      globalAvg
    }
  }

  /** @param train
    *   training set
    * @param computeSimilarity
    *   : corresponds to the similarity metric we wish to use. the function
    *   takes 2 users and outputs the similarity
    * @return:
    *   (Int, Int) => Double: returns the predictor function that takes a user
    *   and an item as arguments and returns the prediction as explained in
    *   Eq(8)
    */
  def predictorPersonalized(
      train: Seq[Rating],
      computeSimilarity: (Int, Int) => Double
  ): ((Int, Int) => Double) = {

    val globalAvg = cmptGlobAvg(train)
    val userAvgs = computeUserAvg(train)
    val itemToCollection = getItemToCollection(train)

    //Map that stores similarities. Will be updated during the prediction process.
    val storesSimilarities = scala.collection.mutable.Map[(Int, Int), Double]()

    //RETURN prediction (u,i)=>Double
    (userId: Int, itemId: Int) => {
      val userAvg = userAvgs.getOrElse(userId, globalAvg)
      if (userAvg == globalAvg) {
        globalAvg
      } else {
        // Computing specificItemAvgDev
        var specificItemAvgDev = 0.0
        val setU_i_ratings = itemToCollection.getOrElse(itemId, Nil)

        val temp = setU_i_ratings.map(x => {
          var similarity_u_v = storesSimilarities.getOrElseUpdate((userId, x.user),-1.0)
          //Check if the similarity of (userId, x.user) is in the map, otherwise compute it and store it.
          if (similarity_u_v< 0) {
            similarity_u_v = computeSimilarity(userId, x.user)
            storesSimilarities += (userId, x.user) -> similarity_u_v
            storesSimilarities += (x.user, userId) -> similarity_u_v
          }
          // For each user v in the set U_i, we generate the tuple ((r_hat_v_i * similarity_u_v), similarity_u_v)
          (
            cmptNormalizedDeviation(
              x.user,
              x.item,
              x.rating,
              userAvgs,
              globalAvg
            ) * similarity_u_v,
            similarity_u_v
          )
        })
        // Compute the user-specific weighted-sum deviation for given item i.e. Eq(7)
        val acc = temp.foldLeft((0.0,0.0)) { case (a, (r, s)) =>
          ( a._1 + r, a._2 + math.abs(s))
        }
        //val weightedSum = temp.foldLeft(0.0) { case (a, (r, s)) => a + r }
        if (acc._2 != 0) {
          specificItemAvgDev = acc._1 / acc._2
        }
        // Return prediction as in Eq(8)
        userAvg + specificItemAvgDev * scale(
          (userAvg + specificItemAvgDev),
          userAvg
        )
      }
    }
  }

  /** @param train
    *   training set
    * @param k
    *   : number of neighbors to consider
    * @return:
    *   (Int, Int) => Double: returns the predictor function that takes a user
    *   and an item as arguments and returns the prediction as explained in
    *   Eq(8)
    */
  def predictorKnn(
      train: Seq[shared.predictions.Rating],
      k: Int
  ): ((Int, Int) => Double) = {
    val userArray = train.map(_.user).distinct
    val globalAvg = cmptGlobAvg(train)
    val usersAvg = computeUserAvg(train)
    val normalizedDev = buildNormalizedDevMap(train, usersAvg, globalAvg)
    val storesSimilarities = scala.collection.mutable.Map[(Int, Int), Double]()
    val cosSimilarity = cosineSimilarity(train)

    var neighbours = scala.collection.mutable.Map[Int, Seq[(Int, Double)]]()


    (userId: Int, itemId: Int) => {
      //By convention we assume that if the similarity contains user(i) with itself, then the
      //similarity of i with all the others
      if (!storesSimilarities.contains(userId, userId)) {
        var sim = 0.0
        for (elem <- userArray) {
          if (!storesSimilarities.contains(userId, elem)) {
            sim = cosSimilarity(userId, elem)
            storesSimilarities += ((userId, elem) -> sim)
            storesSimilarities += ((elem, userId) -> sim)
          }
        }
      }
        
      val nn = neighbours.getOrElseUpdate(userId, {
        userArray
          .filter(_ != userId)
          .map(user => (user, storesSimilarities((userId, user))))
          .sortBy(-_._2)
          .take(k)
      })

      prediction(
        nn,
        userId,
        itemId,
        k,
        usersAvg,
        globalAvg,
        normalizedDev
      )
    }
  }

}
