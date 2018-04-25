name := "spark-gnb"
version := "1.2-SPARK-2.3.0-SNAPSHOT"
organization := "org.apache.spark"
scalaVersion := "2.11.11"
spName := "apache/spark-gnb"
//spIgnoreProvided := true,
sparkVersion := "2.3.0"
sparkComponents ++= Seq("core", "sql", "mllib", "hive") //, "mllib-local")
publishTo := Some("Artifactory Realm" at "http://esi-components.esi-group.com/artifactory/snapshot")
credentials += Credentials(Path.userHome / ".m2" / ".credentials")
publishMavenStyle := true
licenses += "Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html")
//resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  // dependencies for unit tests
  //"org.scalactic" %% "scalactic" % "3.0.1" % "test",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test",  // was 2.2.4
  //"junit" % "junit" % "4.12" % "test",
  "org.apache.commons" % "commons-lang3" % "3.4" % "test",
  "com.holdenkarau" %% "spark-testing-base" % "2.2.0_0.8.0" % "test"
)

// Skip tests during assembly
test in assembly := {}