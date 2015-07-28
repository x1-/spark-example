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

/**
 * このジョブはHashingTrickとLogisticRegressionを使ってCTR推定を行うジョブです。
 * KaggleのClick-Through Rate Predictionで用意されているデータを使用することを前提としています。
 *
 * https://www.kaggle.com/c/avazu-ctr-prediction/data
 */
object HashingTrickLR {

  import com.inkenkun.x1.spark.examples._
  lazy val sparkConfig = ConfigFactory.load.getConfig( "spark" ).toSparkConf

  def main( args: Array[String] ) {

    val sc = new SparkContext( sparkConfig )
    val sqlContext = new SQLContext( sc )
    import sqlContext.implicits._


    val csv = args(0)  // kaggleデータのパスが渡されることを想定


    /** ---------------------------------------------------------------------------------------------------------*/
    // CSV(訓練データ)の読み込み
    val rawCsv = sqlContext.read.format( "com.databricks.spark.csv" ).option( "header", "false" ).load( csv )


    /** ---------------------------------------------------------------------------------------------------------*/
    // UDF(=UserDefinedFunction)の定義
    val f    = udf { (v:String, prefix:String ) => s"$prefix:$v" }
    val hour = udf { (v:String ) => v slice( 6, 8 ) }


    /** ---------------------------------------------------------------------------------------------------------*/
    // 訓練データの準備
    val data = rawCsv.select(
      $"id",
      $"click".cast( DoubleType ).as( "label" ),
      array(
        f( hour( $"hour" ),     lit( "hour" ),
        f( $"C1",               lit( "C1" ),
        f( $"banner_pos",       lit( "banner_pos" ),
        f( $"site_id",          lit( "site_id" ),
        f( $"site_domain",      lit( "site_domain" ),
        f( $"site_category",    lit( "site_category" ),
        f( $"app_id",           lit( "app_id" ),
        f( $"app_domain",       lit( "app_domain" ),
        f( $"app_category",     lit( "app_category" ),
        f( $"device_id",        lit( "device_id" ),
        f( $"device_ip",        lit( "device_ip" ),
        f( $"device_model",     lit( "device_model" ),
        f( $"device_type",      lit( "device_type" ),
        f( $"device_conn_type", lit( "device_conn_type"  ),
        f( $"C14",              lit( "C14" ) ),
        f( $"C15",              lit( "C15" ) ),
        f( $"C16",              lit( "C16" ) ),
        f( $"C17",              lit( "C17" ) ),
        f( $"C18",              lit( "C18" ) ),
        f( $"C19",              lit( "C19" ) ),
        f( $"C20",              lit( "C20" ) ),
        f( $"C21",              lit( "C21" ) )
      ).as( "text" )
)

    /** ---------------------------------------------------------------------------------------------------------*/
    // 訓練データと検証データにわける
    val splits = data.randomSplit( Array(0.7, 0.3) )
    val ( train, test ) = ( splits(0), splits(1) )


    /** ---------------------------------------------------------------------------------------------------------*/
    // HashingTrickで次元削減
    val hashingTF = new HashingTF().setNumFeatures( 1000 ).setInputCol( "text" ).setOutputCol( "features" )
    // ロジスティック回帰でクリック確率を予測
    val lr = new LogisticRegression().setMaxIter( 10 ).setRegParam( 0.1 )

    /** 学習 */
    val pipeline = new Pipeline().setStages( Array( hashingTF, lr ) )
    val model = pipeline.fit( train )


    // スキーマ
    model.transform( test ).select("*").schema.printTreeString


    /** ---------------------------------------------------------------------------------------------------------*/
    // またUDF(=UserDefinedFunction)の定義
    val v0 = udf { v:Vector => v(0) }
    val v1 = udf { v:Vector => v(1) }

    val predictPrint: Row => Unit = {
      case Row( label: Double, p: Double, prob0: Double, prob1: Double ) =>
        println(s"label:$label, predict:${p}, v(0):$prob0, v(1):$prob1")
    }

    // スコア( predict(1) )<=0.5の場合prediction=0.0と推定している
    model.transform( test ).select(
      $"label",
      $"prediction",
      v0( $"probability" ).as( "probability0" ),
      v1( $"probability" ).as( "probability1" )
    ).filter(
      $"prediction" === 0d
    ).take(10).foreach ( predictPrint )
    /**
      label:0.0, predict:0.0, v(0):0.7910189771775317, v(1):0.20898102282246844
      label:0.0, predict:0.0, v(0):0.8477321648407957, v(1):0.1522678351592042
      label:0.0, predict:0.0, v(0):0.733239571522733,  v(1):0.266760428477267
      label:0.0, predict:0.0, v(0):0.7366599532277236, v(1):0.26334004677227635
      label:0.0, predict:0.0, v(0):0.7931168763524822, v(1):0.20688312364751776
      label:0.0, predict:0.0, v(0):0.8983887875222611, v(1):0.10161121247773879
      label:0.0, predict:0.0, v(0):0.9041724969920544, v(1):0.09582750300794549
      label:0.0, predict:0.0, v(0):0.7561229544326091, v(1):0.24387704556739082
      label:0.0, predict:0.0, v(0):0.7056205856658503, v(1):0.29437941433414977
      label:0.0, predict:0.0, v(0):0.8520573239287644, v(1):0.14794267607123565
     */

    // スコア( predict(1) )>0.5の場合prediction=1.0と推定している
    model.transform( test ).select(
      $"label",
      $"prediction",
      v0( $"probability" ).as( "probability0" ),
      v1( $"probability" ).as( "probability1" )
    ).filter(
      $"prediction" === 1d
    ).take(10).foreach ( predictPrint )
    /**
      label:0.0, predict:1.0, v(1):0.4746134314533666,  v(0):0.5253865685466333
      label:1.0, predict:1.0, v(1):0.4746134314533666,  v(0):0.5253865685466333
      label:0.0, predict:1.0, v(1):0.4789477092304181,  v(0):0.5210522907695819
      label:0.0, predict:1.0, v(1):0.4746134314533666,  v(0):0.5253865685466333
      label:1.0, predict:1.0, v(1):0.3548366704327911,  v(0):0.6451633295672089
      label:0.0, predict:1.0, v(1):0.4746134314533666,  v(0):0.5253865685466333
      label:0.0, predict:1.0, v(1):0.4746134314533666,  v(0):0.5253865685466333
      label:1.0, predict:1.0, v(1):0.4793079805634424,  v(0):0.5206920194365575
      label:1.0, predict:1.0, v(1):0.49726188482284245, v(0):0.5027381151771576
      label:0.0, predict:1.0, v(1):0.4746134314533666,  v(0):0.5253865685466333
     */

    model.transform( test ).select(
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
    val tested = model.transform( test ).select(
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
    // 交差検証
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
    val cvModel = crossval.fit( data )
    /**
     *
     Array({
        logreg_9d3153f29ceb-maxIter: 10,
        hashingTF_2cab2d172d76-numFeatures: 10,
        logreg_9d3153f29ceb-regParam: 0.1
      }, {
        logreg_9d3153f29ceb-maxIter: 10,
        hashingTF_2cab2d172d76-numFeatures: 10,
        logreg_9d3153f29ceb-regParam: 0.001
      }, {
        logreg_9d3153f29ceb-maxIter: 10,
        hashingTF_2cab2d172d76-numFeatures: 1000,
        logreg_9d3153f29ceb-regParam: 0.1
      }, {
        logreg_9d3153f29ceb-maxIter: 10,
        hashingTF_2cab2d172d76-numFeatures: 1000,
        logreg_9d3153f29ceb-regParam: 0.001
      }, {
        logreg_9d3153f29ceb-maxIter: 100,
        hashingTF_2cab2d172d76-numFeatures: 10,
        logreg_9d3153f29ceb-regParam: 0.1
      }, {
        logreg_9d3153f29ceb-maxIter: 100,
        hashingTF_2cab2d172d76-numFeatures: 10,
        logreg_9d3153f29ceb-regParam: 0.001
        :
     *
     */


    val parent = cvModel.bestModel.parent.asInstanceOf[Pipeline]
    val bestHT = parent.getStages(0).asInstanceOf[HashingTF]
    val bestLR = parent.getStages(1).asInstanceOf[LogisticRegression]
    println(s"numFeatures: ${bestHT.getNumFeatures}")
    println(s"regParam: ${bestLR.getRegParam}")
    println(s"maxIter: ${bestLR.getMaxIter}")


    sc.stop()
  }


}
