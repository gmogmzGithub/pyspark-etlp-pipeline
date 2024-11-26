from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import logging
import os
from datetime import datetime

class EarthquakeETLPipeline:
    def __init__(self, mongo_host="mongodb", mongo_port=27017):
        self.logger = self._setup_logger()
        self.spark = self._create_spark_session(mongo_host, mongo_port)
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('EarthquakeETL')

    def _create_spark_session(self, mongo_host, mongo_port):
        self.logger.info("Initializing Spark session...")
        return SparkSession.builder \
            .master('local[4]') \
            .appName('earthquake_analysis') \
            .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1') \
            .config('spark.mongodb.input.uri', f'mongodb://admin:password123@{mongo_host}:{mongo_port}/admin?authSource=admin') \
            .config('spark.mongodb.output.uri', f'mongodb://admin:password123@{mongo_host}:{mongo_port}/admin?authSource=admin') \
            .config('spark.driver.memory', '4g') \
            .config('spark.executor.memory', '4g') \
            .getOrCreate()

    def extract(self, historical_data_path, query_data_path):
        """Extract data from CSV files"""
        self.logger.info("Extracting data from CSV files...")
        
        # Load historical data
        self.historical_df = self.spark.read.csv(
            historical_data_path, 
            header=True
        )
        
        # Load query data
        self.query_df = self.spark.read.csv(
            query_data_path, 
            header=True
        )
        
        return self.historical_df, self.query_df

    def transform_historical_data(self):
        """Transform historical earthquake data"""
        self.logger.info("Transforming historical data...")
        
        # Drop unnecessary columns
        drop_columns = [
            'Depth Error', 'Time', 'Depth Seismic Stations', 
            'Magnitude Error', 'Magnitude Seismic Stations', 
            'Azimuthal Gap', 'Root Mean Square', 'Source', 
            'Location Source', 'Magnitude Source', 'Status'
        ]
        df = self.historical_df.drop(*drop_columns)
        
        # Add year column and convert data types
        df = df.withColumn('Year', year(to_timestamp('Date', 'dd/MM/yyyy')))
        df = df.withColumn('Latitude', df['Latitude'].cast(DoubleType())) \
               .withColumn('Longitude', df['Longitude'].cast(DoubleType())) \
               .withColumn('Depth', df['Depth'].cast(DoubleType())) \
               .withColumn('Magnitude', df['Magnitude'].cast(DoubleType()))
        
        # Calculate aggregations
        df_quake_freq = df.groupBy('Year').count().withColumnRenamed('count', "Count")
        df_max = df.groupBy('Year').max('Magnitude').withColumnRenamed('max(Magnitude)', 'MAX Magnitude')
        df_avg = df.groupBy('Year').avg('Magnitude').withColumnRenamed('avg(Magnitude)', 'AVG Magnitude')
        
        # Join aggregations
        self.freq_df = df_quake_freq.join(df_avg, ['Year']).join(df_max, ['Year'])
        
        # Clean data
        self.transformed_historical_df = df.dropna()
        self.freq_df = self.freq_df.dropna()
        
        return self.transformed_historical_df, self.freq_df

    def transform_query_data(self):
        """Transform query data for predictions"""
        self.logger.info("Transforming query data...")
        
        # Select and rename columns
        df = self.query_df.select('time', 'latitude', 'longitude', 'depth', 'mag')
        df = df.withColumnRenamed('time', 'Date') \
               .withColumnRenamed('latitude', 'Latitude') \
               .withColumnRenamed('mag', 'Magnitude') \
               .withColumnRenamed('longitude', 'Longitude') \
               .withColumnRenamed('depth', 'Depth')
        
        # Convert data types
        df = df.withColumn('Latitude', df['Latitude'].cast(DoubleType())) \
               .withColumn('Longitude', df['Longitude'].cast(DoubleType())) \
               .withColumn('Depth', df['Depth'].cast(DoubleType())) \
               .withColumn('Magnitude', df['Magnitude'].cast(DoubleType()))
        
        self.transformed_query_df = df.dropna()
        return self.transformed_query_df

    def train_model(self):
        """Train RandomForest model for magnitude prediction"""
        self.logger.info("Training prediction model...")
        
        # Prepare features
        feature_cols = ['Latitude', 'Longitude', 'Depth']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        
        # Create and train model
        rf = RandomForestRegressor(featuresCol='features', labelCol='Magnitude')
        pipeline = Pipeline(stages=[assembler, rf])
        
        # Train model
        self.model = pipeline.fit(self.transformed_historical_df)
        return self.model

    def make_predictions(self):
        """Make predictions using trained model"""
        self.logger.info("Making predictions...")
        
        predictions = self.model.transform(self.transformed_query_df)
        
        # Evaluate model
        evaluator = RegressionEvaluator(
            labelCol='Magnitude', 
            predictionCol='prediction', 
            metricName='rmse'
        )
        rmse = evaluator.evaluate(predictions)
        self.logger.info(f'Model RMSE: {rmse}')
        
        # Prepare prediction results
        self.pred_df = predictions.select(
            'Latitude', 'Longitude', 'prediction'
        ).withColumnRenamed('prediction', 'Prediction Magnitude')
        
        self.pred_df = self.pred_df.withColumn('Year', lit(2024)) \
                                  .withColumn('RMSE', lit(rmse))
        
        return self.pred_df

    def load_to_mongodb(self):
        """Load transformed and predicted data to MongoDB"""
        self.logger.info("Loading data to MongoDB...")
        
        # Load historical data
        self.transformed_historical_df.write \
            .format("com.mongodb.spark.sql.DefaultSource") \
            .mode("overwrite") \
            .option("spark.mongodb.output.uri", 
                   "mongodb://admin:password123@mongodb:27017/Quake.quakes?authSource=admin") \
            .save()
        
        # Load frequency data
        self.freq_df.write \
            .format("com.mongodb.spark.sql.DefaultSource") \
            .mode("overwrite") \
            .option("spark.mongodb.output.uri", 
                   "mongodb://admin:password123@mongodb:27017/Quake.quakes_freq?authSource=admin") \
            .save()
        
        # Load predictions
        self.pred_df.write \
            .format("com.mongodb.spark.sql.DefaultSource") \
            .mode("overwrite") \
            .option("spark.mongodb.output.uri", 
                   "mongodb://admin:password123@mongodb:27017/Quake.prediction_results?authSource=admin") \
            .save()

    def run_pipeline(self, historical_data_path, query_data_path):
        """Run the complete ETL pipeline"""
        try:
            self.logger.info("Starting ETL pipeline...")
            
            # Extract
            self.extract(historical_data_path, query_data_path)
            
            # Transform
            self.transform_historical_data()
            self.transform_query_data()
            
            # Train and Predict
            self.train_model()
            self.make_predictions()
            
            # Load
            self.load_to_mongodb()
            
            self.logger.info("ETL pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        
        finally:
            self.spark.stop()

if __name__ == "__main__":
    # Example usage
    pipeline = EarthquakeETLPipeline()
    pipeline.run_pipeline(
        historical_data_path="../data/database.csv",
        query_data_path="../data/query.csv"
    )