package com.inkenkun.x1.spark.examples.ml

import java.util.UUID
import scala.collection.JavaConverters._

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.sql.types.{StringType, ArrayType, DataType}
import org.atilika.kuromoji.Tokenizer

class JapaneseTokenizer( override val uid: String ) extends UnaryTransformer[String, Seq[String], JapaneseTokenizer] {

  lazy val tokenizer = Tokenizer.builder().build()

  def this() = this( s"stok_${UUID.randomUUID().toString.takeRight(12)}" )

  override protected def createTransformFunc: String => Seq[String] = {
    tokenizer.tokenize( _ ).asScala.map( t => t.getSurfaceForm )
  }

  override protected def validateInputType( inputType: DataType ): Unit = {
    require(inputType == StringType, s"Input type must be string type but got $inputType.")
  }

  override protected def outputDataType: DataType = new ArrayType(StringType, false)
}