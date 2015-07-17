package com.inkenkun.x1.spark.examples.ml

import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{Row, SQLContext, SaveMode}

case class LabeledDocument(id: Long, text: String, label: Double)
case class Document(id: Long, text: String)

object JapaneseTopic {

  import com.inkenkun.x1.spark.examples._
  lazy val sparkConfig = ConfigFactory.load.getConfig( "spark" ).toSparkConf

  def main( args: Array[String] ) {

    val train  = args(0)
    val pred   = args(1)
    val output = args(2)

    val sc = new SparkContext( sparkConfig )
    val sqlContext = new SQLContext( sc )
    import sqlContext.implicits._

    // training
    val training = sqlContext.read.format( "com.databricks.spark.csv" ).option( "header", "true" ).load( train )
      .map( r => LabeledDocument(
        id = r.getString(0).toLong,
        label = r.getString(2).toDouble,
        text = r.getString(1) )
      )

    val tokenizer = new JapaneseTokenizer()
      .setInputCol( "text" )
      .setOutputCol( "words" )

    val hashingTF = new HashingTF()
      .setNumFeatures( 1000 )
      .setInputCol( tokenizer.getOutputCol )
      .setOutputCol( "features" )

    val lr = new LogisticRegression()
      .setMaxIter( 10 )
      .setRegParam( 0.001 )

    val ovr = new OneVsRest().setClassifier( lr )

    val pipeline = new Pipeline()
      .setStages( Array(tokenizer, hashingTF, ovr) )

    val model = pipeline.fit( training.toDF() )

    val test = sqlContext.read.format( "com.databricks.spark.csv" ).option( "header", "true" ).load( pred )
      .map( r => Document(
        id = r.getString(0).toInt,
        text = r.getString(1) )
      )

    val predict = model.transform( test.toDF() )
    predict.select( "id", "prediction" ).write.format( "json" ).mode( SaveMode.Overwrite ).save( output )

    // Cross Validation
    val crossval = new CrossValidator()
      .setEstimator( pipeline )
      .setEvaluator( new RegressionEvaluator )

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid( lr.regParam, Array(0.1, 0.001) )
      .build()

    crossval.setEstimatorParamMaps( paramGrid )
    crossval.setNumFolds( 3 ) // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = crossval.fit( training.toDF() )

    cvModel.transform( test.toDF() )
      .select("id", "text", "features", "prediction")
      .collect()
      .foreach { case Row(id: Long, text: String, features: Vector, prediction: Double) =>
        println(s"($id, $text) --> features=$features, prediction=$prediction")
      }

    sc.stop()

  }
}
