package com.inkenkun.x1.spark

import scala.collection.JavaConverters._

import com.typesafe.config.Config
import org.apache.spark.SparkConf

package object examples {
  
  implicit class convertSparkConf( config: Config ) {
    def toSparkConf: SparkConf =
      config.entrySet.asScala.foldLeft( new SparkConf() ) { (spark, entry) =>
        spark.set( s"spark.${entry.getKey}", config.getString(entry.getKey) )
      }
  }
}
