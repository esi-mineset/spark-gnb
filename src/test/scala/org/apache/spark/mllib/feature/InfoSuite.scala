package org.apache.spark.mllib.feature

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite}


/**
  * Test entropy calculation.
  *
  * @author Barry Becker
  */
@RunWith(classOf[JUnitRunner])
class InfoSuite extends FunSuite {

  test("Test Info construction") {

    val info = new Info(IndexedSeq(2L, 5L))

    assertResult(7L) {
      info.s
    }
  }

}