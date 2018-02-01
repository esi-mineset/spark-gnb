name := "spark-gnb"
version := "0.1-SPARK-2.1.1"
organization := "org.apache.spark"
scalaVersion := "2.11.11"
spName := "apache/spark-gnb"
//spIgnoreProvided := true,
sparkVersion := "2.1.1"
sparkComponents ++= Seq("core", "sql", "mllib")
publishTo := Some("Artifactory Realm" at "http://esi-components.esi-group.com/artifactory/snapshot")
credentials += Credentials(Path.userHome / ".m2" / ".credentials")
publishMavenStyle := true
licenses += "Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html")

libraryDependencies ++= Seq(
  "joda-time" % "joda-time" % "2.9.4",
  // "org.apache.spark" % "spark-core_2.11" % "2.1.1",
  // "org.apache.spark" % "spark-sql_2.11" % "2.1.1",
  // "org.apache.spark" % "spark-mllib_2.11" % "2.1.1",
  // dependencies for unit tests
  //"org.scalactic" %% "scalactic" % "3.0.1" % "test",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test",  // was 2.2.4
  //"junit" % "junit" % "4.12" % "test",
  "org.apache.commons" % "commons-lang3" % "3.4" % "test",
  "com.holdenkarau" %% "spark-testing-base" % "2.2.0_0.8.0" % "test"

  //"org.apache.spark" %% "spark-core" % "2.1.1" % "test" force(),
  ///"org.apache.spark" %% "spark-sql" % "2.1.1" % "test" force(),
  //"org.apache.spark" %% "spark-mllib" % "2.1.1" % "test" force()
)

// Skip tests during assembly
test in assembly := {}