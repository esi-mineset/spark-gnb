package org.apache.spark.ml.classification

import org.scalatest.FunSuite
import org.apache.spark.{SparkException}

class SimpleSuite extends FunSuite {
  test("foo") {
    assertResult(true) { 5 == 5 }
  }
}