package com.inkenkun.x1.spark.examples.ml

import com.typesafe.config.ConfigFactory
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext, SaveMode}

object HashingTrickLR {

  import com.inkenkun.x1.spark.examples._
  lazy val sparkConfig = ConfigFactory.load.getConfig( "spark" ).toSparkConf

  def main( args: Array[String] ) {

    val sc = new SparkContext( sparkConfig )
    val sqlContext = new SQLContext( sc )
    import sqlContext.implicits._


    val input = args(0)
    val data = sqlContext.read.json( input )


    val f = udf { (v:String, prefix:Int ) => s"$prefix:$v" }
    val features = data.select(
      $"id",
      $"clicks".cast( DoubleType ).as( "label" ),
      array(
        f( $"d_rank", lit(1) ),
        f( $"viewability", lit(2) ),
        f( $"hour", lit(3) ),
        f( $"weekday", lit(4) ),
        f( $"sponsor_id", lit(5) ),
        f( $"prediction", lit(6) ),
        f( $"os", lit(7) ),
        f( $"recency", lit(8) ) ).as( "text" )
    )

    val hashingTF = new HashingTF().setNumFeatures( 100 ).setInputCol( "text" ).setOutputCol( "features" )
    val lr = new LogisticRegression().setMaxIter( 10 ).setRegParam( 0.001 )

    val pipeline = new Pipeline().setStages( Array( hashingTF, lr ) )
    val model = pipeline.fit( features.repartition(8) )


    model.transform( features ).select("*").schema.printTreeString
    model.transform( features ).select("features", "label", "probability", "prediction").take(10).foreach {
      case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        println(s"($features, $label) -> prob=$prob, prediction=$prediction")
    }

    val v2w = udf { (v:Vector ) =>
      //    if ( p == 0d ) v(1)
      //    else v(0)
      1.0 / (1.0 + math.exp(v(0)))
    }
    val v2w2 = udf { (v:Vector ) =>
      //    if ( p == 0d ) v(1)
      //    else v(0)
      1.0 / (1.0 + math.exp(-v(0)))
    }
    val v0 = udf { v:Vector =>
      v(0)
    }
    val v2 = udf { v:Vector =>
      v(1)
    }
    val sigmoid = udf { x: Double =>
      val r = math.tanh(x/2)
      if ( r.isNaN ) 0
      else (r+1)/2
    }
    val logistic = udf { x: Double =>
      1.0 / (1.0 + math.exp(-x))
    }

    model.transform( features ).select(
      $"text",
      v2( $"probability" ).as( "probability" ),
      v0( $"probability" ).as( "probability0" ),
      v2w( $"probability" ).as( "weight" ),
      v2w2( $"probability" ).as( "weight2" ),
      v0( $"rawPrediction" ).as( "r3" ),
      v2( $"rawPrediction" ).as( "r4" )
    ).groupBy("text").agg(
      avg( $"probability" ).as( "probability" ),
      avg( $"probability0" ).as( "probability0" ),
      avg( $"weight" ).as( "weight" ),
      avg( $"weight2" ).as( "weight2" ),
      avg( $"r3" ).as( "r3" ),
      avg( $"r4" ).as( "r4" )
    ).select(
      $"text",
      $"probability",
      $"probability0",
      $"weight",
      $"weight2",
      sigmoid($"r4"),
      logistic($"r4")
    ).orderBy(
      $"probability".desc
    ).take(100).foreach {
      case Row( text: Seq[_], p: Double, r:Double, w: Double, w2: Double, r3: Double, r4: Double ) =>
        println(s"$text, ${p*10000}, $r, $w, $w2, $r3, $r4")
    }

    val tested = model.transform( features ).select(
      v0( $"probability" ),
      $"label"
    )
    val scoreAndLabels = tested.map {
      case Row( prob: Double, label: Double ) =>
        (prob, label)
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area Under ROC is " + auROC)

    // Cross Validation
    val crossval =
      new CrossValidator().setEstimator( pipeline ).setEvaluator( new RegressionEvaluator )

    val paramGrid =
      new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(10, 100, 1000)).addGrid( lr.regParam, Array(0.1, 0.01, 0.001) ).build()

    crossval.setEstimatorParamMaps( paramGrid )
    crossval.setNumFolds( 3 ) // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = crossval.fit( features )
    val parent = cvModel.bestModel.parent.asInstanceOf[Pipeline]
    val m = parent.getStages(1).asInstanceOf[LogisticRegression]
    println(s"regParam: ${m.getRegParam}")
    println(s"maxIter: ${m.getMaxIter}")

    sc.stop()
  }


}
