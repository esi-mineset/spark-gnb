package org.apache.spark.mllib.feature

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.format.DateTimeFormat

/**
  * Loads various test datasets
  */
object TestHelper {

  final val SPARK_CTX = createSparkContext()
  final val FILE_PREFIX = "src/test/resources/data/"
  final val ISO_DATE_FORMAT = DateTimeFormat.forPattern("yyyy-MM-dd'T'HH:mm:ss")
  final val NULL_VALUE = "?"

  // This value is used to represent nulls in string columns
  final val MISSING = "__MISSING_VALUE__"
  final val CLEAN_SUFFIX: String = "_CLEAN"
  final val INDEX_SUFFIX: String = "_IDX"


  def createSparkContext() = {
    // the [n] corresponds to the number of worker threads and should correspond ot the number of cores available.
    val conf = new SparkConf().setAppName("test-spark").setMaster("local[4]")
    // Changing the default parallelism to 4 hurt performance a lot for a big dataset.
    // When maxByPart was 10000, it wend from 39 min to 4.5 hours.
    //conf.set("spark.default.parallelism", "4")
    val sc = new SparkContext(conf)
    LogManager.getRootLogger.setLevel(Level.WARN)
    sc
  }

}
