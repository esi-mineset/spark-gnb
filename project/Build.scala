import sbt.{Credentials, _}
import sbt.Keys.{credentials, _}
import sbtsparkpackage.SparkPackagePlugin.autoImport._

object ProjectBuild extends Build {

  lazy val project = Project(
    id = "root",
    base = file("."),
    settings = Defaults.coreDefaultSettings ++ Seq(
      name := "spark-gnb",
      version := "0.1-SPARK-2.1.1",
      organization := "org.apache.spark",
      scalaVersion := "2.11.6",
      spName := "apache/spark-gnb",
      sparkVersion := "2.1.1",
      sparkComponents += "mllib",
      publishTo := Some("Artifactory Realm" at "http://esi-components.esi-group.com/artifactory/snapshot"),
      credentials += Credentials(Path.userHome / ".m2" / ".credentials"),
      publishMavenStyle := true,
      licenses += "Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html"),

      libraryDependencies ++= Seq(
        "joda-time" % "joda-time" % "2.9.4",
        // dependencies for unit tests
        "org.scalatest" %% "scalatest" % "2.2.4" % "test",
        "junit" % "junit" % "4.12" % "test",
        "org.apache.commons" % "commons-lang3" % "3.4" % "test"
        )
      )
  )

}

