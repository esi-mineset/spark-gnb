# spark-general-naive-bayes
A more general implementation of a Naive Bayes classifier than the one provided natively in Spark.
I can be applied to any sort of data, not just documents.

Originally we had added this code to our fork of spark, but we now want to get off the fork so that we can use official versions.
One step toward that goal is to make this classifier a separate project.

To build, do "sbt assembly test"

For more information see my [Spark Summit 2017 presentation](https://www.youtube.com/watch?v=Y_rckbjA9sE)