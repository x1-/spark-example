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


    val trainCSV = args(0)
    val testCSV  = args(1)

    // UDF(=UserDefinedFunction)の定義
    val f    = udf { (v:String, prefix:Int ) => s"$prefix:$v" }
    val hour = udf { (v:String ) => v slice( 6, 8 ) }

    /** ---------------------------------------------------------------------------------------------------------*/
    // 訓練データの準備
    val trainRaw = sqlContext.read.format( "com.databricks.spark.csv" ).option( "header", "false" ).load( trainCSV )

    val train = trainRaw.select(
      $"C0" as "id",
      $"C1".cast( DoubleType ).as( "label" ),
      array(
        f( hour( $"C2" ), lit(2) ),
        f( $"C4",  lit(4) ),
        f( $"C5",  lit(5) ),
        f( $"C6",  lit(6) ),
        f( $"C7",  lit(7) ),
        f( $"C9",  lit(8) ),
        f( $"C10", lit(10) ),
        f( $"C13", lit(13) ),
        f( $"C14", lit(14) )
      ).as( "text" )
    )
    val splits = train.randomSplit( Array(0.7, 0.3) )
    val ( trainingData, testData ) = ( splits(0), splits(1) )


    /** ---------------------------------------------------------------------------------------------------------*/
    // HashingTrickで次元削減
    val hashingTF = new HashingTF().setNumFeatures( 1000 ).setInputCol( "text" ).setOutputCol( "features" )
    // ロジスティック回帰で確立を予測
    val lr = new LogisticRegression().setMaxIter( 10 ).setRegParam( 0.1 )

    /** 学習 */
    val pipeline = new Pipeline().setStages( Array( hashingTF, lr ) )
    val model = pipeline.fit( train )


    // スキーマ
    model.transform( testData ).select("*").schema.printTreeString


    /** ---------------------------------------------------------------------------------------------------------*/
    // またUDF(=UserDefinedFunction)の定義
    val v0 = udf { v:Vector => v(0) }
    val v1 = udf { v:Vector => v(1) }

    val predictPrint: Row => Unit = {
      case Row( label: Double, p: Double, prob0: Double, prob1: Double ) =>
        println(s"label:$label, predict:${p}, v(0):$prob1, v(1):$prob0")
    }

    // スコア<=0.5の場合prediction=0.0と推定している
    model.transform( testData ).select(
      $"label",
      $"prediction",
      v0( $"probability" ).as( "probability0" ),
      v1( $"probability" ).as( "probability1" )
    ).filter(
      $"prediction" === 0d
    ).take(10).foreach ( predictPrint )
    /**
      label:0.0, predict:0.0, v(0):0.20898102282246844, v(1):0.7910189771775317
      label:0.0, predict:0.0, v(0):0.1522678351592042,  v(1):0.8477321648407957
      label:0.0, predict:0.0, v(0):0.266760428477267,   v(1):0.733239571522733
      label:0.0, predict:0.0, v(0):0.26334004677227635, v(1):0.7366599532277236
      label:0.0, predict:0.0, v(0):0.20688312364751776, v(1):0.7931168763524822
      label:0.0, predict:0.0, v(0):0.10161121247773879, v(1):0.8983887875222611
      label:0.0, predict:0.0, v(0):0.09582750300794549, v(1):0.9041724969920544
      label:0.0, predict:0.0, v(0):0.24387704556739082, v(1):0.7561229544326091
      label:0.0, predict:0.0, v(0):0.29437941433414977, v(1):0.7056205856658503
      label:0.0, predict:0.0, v(0):0.14794267607123565, v(1):0.8520573239287644
     */

    // スコア>0.5の場合prediction=1.0と推定している
    model.transform( testData ).select(
      $"label",
      $"prediction",
      v0( $"probability" ).as( "probability0" ),
      v1( $"probability" ).as( "probability1" )
    ).filter(
      $"prediction" === 1d
    ).take(10).foreach ( predictPrint )
    /**
      label:0.0, predict:1.0, v(0):0.5253865685466333, v(1):0.4746134314533666
      label:1.0, predict:1.0, v(0):0.5253865685466333, v(1):0.4746134314533666
      label:0.0, predict:1.0, v(0):0.5210522907695819, v(1):0.4789477092304181
      label:0.0, predict:1.0, v(0):0.5253865685466333, v(1):0.4746134314533666
      label:1.0, predict:1.0, v(0):0.6451633295672089, v(1):0.3548366704327911
      label:0.0, predict:1.0, v(0):0.5253865685466333, v(1):0.4746134314533666
      label:0.0, predict:1.0, v(0):0.5253865685466333, v(1):0.4746134314533666
      label:1.0, predict:1.0, v(0):0.5206920194365575, v(1):0.4793079805634424
      label:1.0, predict:1.0, v(0):0.5027381151771576, v(1):0.49726188482284245
      label:0.0, predict:1.0, v(0):0.5253865685466333, v(1):0.4746134314533666
     */

    model.transform( testData ).select(
      $"text",
      v0( $"probability" ).as( "probability0" ),
      v1( $"probability" ).as( "probability1" )
    ).groupBy("text").agg(
      avg( $"probability0" ).as( "probability0" ),
      avg( $"probability1" ).as( "probability1" )
    ).select(
      $"text",
      $"probability0",
      $"probability1"
    ).orderBy(
      $"probability0".desc
    ).take(20).foreach {
      case Row( text: Seq[_], p0: Double, p1:Double ) =>
        println(s"$text, $p0, $p1")
    }


    /** ---------------------------------------------------------------------------------------------------------*/
    // ROC curve
    val tested = model.transform( testData ).select(
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


    /** ---------------------------------------------------------------------------------------------------------*/
    // Cross Validation
    val crossval =
      new CrossValidator().setEstimator( pipeline ).setEvaluator( new RegressionEvaluator )

    val paramGrid =
      new ParamGridBuilder().addGrid(
        hashingTF.numFeatures, Array(10, 1000)
      ).addGrid(
          lr.regParam, Array(0.1, 0.001)
        ).addGrid(
          lr.maxIter, Array(10, 100)
        ).build()

    crossval.setEstimatorParamMaps( paramGrid )
    crossval.setNumFolds( 3 ) // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = crossval.fit( train )
    val parent = cvModel.bestModel.parent.asInstanceOf[Pipeline]
    val bestHT = parent.getStages(0).asInstanceOf[HashingTF]
    val bestLR = parent.getStages(1).asInstanceOf[LogisticRegression]
    println(s"numFeatures: ${bestHT.getNumFeatures}")
    println(s"regParam: ${bestLR.getRegParam}")
    println(s"maxIter: ${bestLR.getMaxIter}")



    /** ---------------------------------------------------------------------------------------------------------*/
    // テストデータの準備
    val testRaw = sqlContext.read.format( "com.databricks.spark.csv" ).option( "header", "false" ).load( testCSV )

    val test = trainRaw.select(
      $"C0" as "id",
      array(
        f( hour( $"C1" ), lit(2) ),
        f( $"C3",  lit(4) ),
        f( $"C4",  lit(5) ),
        f( $"C5",  lit(6) ),
        f( $"C6",  lit(7) ),
        f( $"C8",  lit(8) ),
        f( $"C9",  lit(10) ),
        f( $"C12", lit(13) ),
        f( $"C13", lit(14) )
      ).as( "text" )
    )


    sc.stop()
  }


}
