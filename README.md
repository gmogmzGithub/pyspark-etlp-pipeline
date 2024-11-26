# Earthquake Data Analysis Pipeline

A big data pipeline for analyzing earthquake data using PySpark, MongoDB, and Bokeh visualizations. This project implements an ETLP (Extract, Transform, Load, Predict) pipeline to process historical earthquake data and make predictions using machine learning.

## Features

- **Data Processing Pipeline** using PySpark
- **Machine Learning Model** for earthquake magnitude prediction
- **Interactive Visualizations** using Bokeh
- **MongoDB Integration** for data storage
- **Docker Containerization** for easy deployment

## Architecture

The project consists of three main components:

1. **ETL Pipeline**
   - Data extraction from CSV files
   - Data transformation using PySpark
   - Loading data into MongoDB
   - Machine learning predictions using Spark MLlib

2. **Data Storage**
   - MongoDB for storing processed data
   - Collections for historical data, frequency analysis, and predictions

3. **Visualization Layer**
   - Interactive map showing historical and predicted earthquakes
   - Magnitude trend analysis
   - Depth distribution heatmap
   - Complete dashboard combining all visualizations

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Apache Spark
- MongoDB

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd earthquake-analysis
```

2. Create necessary directories:
```bash
mkdir -p notebooks/data
```

3. Download the data:
```bash
# Download database.csv
curl -o notebooks/data/database.csv https://github.com/EBISYS/WaterWatch/blob/master/database.csv
```

4. Start the Docker containers:
```bash
docker-compose up -d
```

### Configuration

The `docker-compose.yml` file contains the following services:

- MongoDB (`mongodb`) - Database server
- Mongo Express (`mongo-express`) - Web-based MongoDB admin interface
- Jupyter (`jupyter`) - Notebook server with PySpark

Default credentials:
- MongoDB:
  - Username: admin
  - Password: password123
- Mongo Express:
  - Username: admin
  - Password: password123

## 📊 Usage

### Running the Pipeline

1. As a Python script:
```python
from earthquake_pipeline import EarthquakeETLPipeline

pipeline = EarthquakeETLPipeline()
pipeline.run_pipeline(
    historical_data_path="data/database.csv",
    query_data_path="data/query.csv"
)
```

2. Creating visualizations:
```python
from earthquake_visualization import EarthquakeVisualization

viz = EarthquakeVisualization()
viz.load_data()
dashboard = viz.create_dashboard("earthquake_dashboard.html")
```

### Accessing Services

- Jupyter Notebook: http://localhost:8888
- Mongo Express: http://localhost:8081
- MongoDB: localhost:27017

## 📁 Project Structure

```
earthquake-analysis/
├── docker-compose.yml
├── notebooks/
│   ├── data/
│   │   ├── database.csv
│   │   └── query.csv
│   └── work/
├── src/
│   ├── earthquake_pipeline.py
│   └── earthquake_visualization.py
└── README.md
```

## Pipeline Components

### ETL Pipeline (`pipeline.py`)

The ETL pipeline handles:
- Data extraction from CSV files
- Data cleaning and transformation
- Feature engineering
- Machine learning model training
- Predictions generation
- Data loading into MongoDB

### Visualization Module (`visualization.py`)

The visualization module provides:
- Interactive earthquake map
- Magnitude trend analysis
- Depth distribution heatmap
- Comprehensive dashboard generation

## 📚 Data Sources

The pipeline uses two main data sources:

1. Historical Earthquake Data (`database.csv`):
   - Source: [WaterWatch Repository](https://github.com/EBISYS/WaterWatch/blob/master/database.csv)
   - Contains historical earthquake records with location, magnitude, and depth information

2. Query Data (`query.csv`):
   - Source: [WaterWatch Repository](https://github.com/EBISYS/WaterWatch/blob/master/query.csv)
   - Contains recent earthquake data for prediction validation
   - Similar structure to historical data

## 🙏 Acknowledgments

- Data provided by [WaterWatch](https://github.com/EBISYS/WaterWatch)
- Built with Apache Spark, MongoDB, and Bokeh