import pandas as pd
import numpy as np
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import column, row, gridplot
from bokeh.models import HoverTool, ColorBar, LinearColorMapper
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.palettes import Viridis256, RdYlBu11
from bokeh.transform import linear_cmap
import math
from pymongo import MongoClient
import logging

class EarthquakeVisualization:
    def __init__(self, mongo_host='mongodb', mongo_port=27017, 
                 username='admin', password='password123'):
        self.logger = self._setup_logger()
        self.mongo_config = {
            'host': mongo_host,
            'port': mongo_port,
            'username': username,
            'password': password
        }
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('EarthquakeViz')

    def _connect_mongodb(self, db='Quake'):
        """Establish MongoDB connection"""
        try:
            conn_string = f"mongodb://{self.mongo_config['username']}:{self.mongo_config['password']}@" \
                         f"{self.mongo_config['host']}:{self.mongo_config['port']}/{db}?authSource=admin"
            
            client = MongoClient(conn_string)
            return client[db]
        except Exception as e:
            self.logger.error(f"MongoDB connection failed: {str(e)}")
            raise

    def load_data(self):
        """Load data from MongoDB"""
        self.logger.info("Loading data from MongoDB...")
        
        db = self._connect_mongodb()
        
        # Load all required collections
        self.df_quakes = pd.DataFrame(list(db.quakes.find()))
        self.df_freq = pd.DataFrame(list(db.quakes_freq.find()))
        self.df_pred = pd.DataFrame(list(db.prediction_results.find()))
        
        # Clean up MongoDB _id column
        for df in [self.df_quakes, self.df_freq, self.df_pred]:
            if '_id' in df.columns:
                df.drop('_id', axis=1, inplace=True)

    def _mercator_coordinates(self, lat, lon):
        """Convert lat/lon to mercator coordinates"""
        r_major = 6378137.000
        x = r_major * math.radians(lon)
        scale = x/lon if lon != 0 else x
        y = math.log(math.tan(math.pi/4 + math.radians(lat)/2)) * scale
        return x, y

    def create_earthquake_map(self, width=2300, height=700):
        """Create interactive earthquake map"""
        self.logger.info("Creating earthquake map visualization...")
        
        # Process coordinates for both actual and predicted earthquakes
        def prepare_data(df, is_prediction=False):
            coords = [self._mercator_coordinates(lat, lon) 
                     for lat, lon in zip(df['Latitude'], df['Longitude'])]
            
            mag_col = 'Prediction Magnitude' if is_prediction else 'Magnitude'
            
            return ColumnDataSource({
                'lat': [c[1] for c in coords],
                'lon': [c[0] for c in coords],
                'mag': df[mag_col],
                'year': df['Year'],
                'mag_size': df[mag_col] * 4
            })

        actual_source = prepare_data(self.df_quakes)
        pred_source = prepare_data(self.df_pred, is_prediction=True)

        # Create figure
        p = figure(
            title="Earthquake Map",
            width=width,
            height=height,
            x_range=(-2000000, 6000000),
            y_range=(-1000000, 7000000),
            tooltips=[
                ("Year", "@year"),
                ("Magnitude", "@mag")
            ]
        )

        # Add map tile
        p.add_tile(CARTODBPOSITRON)

        # Plot actual earthquakes
        p.circle(
            x='lon', y='lat',
            size='mag_size',
            fill_color='red',
            fill_alpha=0.7,
            source=actual_source,
            legend_label='Historical Earthquakes'
        )

        # Plot predicted earthquakes
        p.circle(
            x='lon', y='lat',
            size='mag_size',
            fill_color='#CCFF33',
            fill_alpha=0.7,
            source=pred_source,
            legend_label='Predicted Earthquakes'
        )

        # Style the plot
        self._style_plot(p)
        return p

    def create_magnitude_trend(self, width=800, height=400):
        """Create magnitude trend visualization"""
        self.logger.info("Creating magnitude trend visualization...")
        
        source = ColumnDataSource(self.df_freq)
        
        p = figure(
            title="Earthquake Magnitude Trends Over Time",
            width=width,
            height=height,
            x_axis_label='Year',
            y_axis_label='Magnitude'
        )

        # Plot average magnitude
        p.line(
            x='Year', y='AVG Magnitude',
            line_color='blue',
            line_width=2,
            legend_label='Average Magnitude',
            source=source
        )

        # Plot maximum magnitude
        p.line(
            x='Year', y='MAX Magnitude',
            line_color='red',
            line_width=2,
            legend_label='Maximum Magnitude',
            source=source
        )

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ('Year', '@Year'),
                ('Average Magnitude', '@{AVG Magnitude}{0.00}'),
                ('Maximum Magnitude', '@{MAX Magnitude}{0.00}')
            ]
        )
        p.add_tools(hover)

        self._style_plot(p)
        return p

    def create_depth_heatmap(self, width=800, height=400):
        """Create depth distribution heatmap"""
        self.logger.info("Creating depth heatmap visualization...")
        
        # Create depth bins
        self.df_quakes['depth_bin'] = pd.cut(self.df_quakes['Depth'], 
                                            bins=20, 
                                            labels=False)
        
        # Calculate average magnitude for each depth bin
        heatmap_data = self.df_quakes.groupby('depth_bin').agg({
            'Depth': 'mean',
            'Magnitude': 'mean',
            'Year': 'count'
        }).reset_index()

        source = ColumnDataSource(heatmap_data)

        # Create color mapper
        mapper = LinearColorMapper(
            palette=RdYlBu11,
            low=heatmap_data['Magnitude'].min(),
            high=heatmap_data['Magnitude'].max()
        )

        p = figure(
            title="Earthquake Depth vs Magnitude Distribution",
            width=width,
            height=height,
            x_axis_label='Depth (km)',
            y_axis_label='Average Magnitude'
        )

        # Create circles with size based on count
        p.circle(
            x='Depth',
            y='Magnitude',
            size='Year',
            fill_color={'field': 'Magnitude', 'transform': mapper},
            fill_alpha=0.7,
            source=source
        )

        # Add color bar
        color_bar = ColorBar(
            color_mapper=mapper,
            title="Magnitude",
            location=(0, 0)
        )
        p.add_layout(color_bar, 'right')

        self._style_plot(p)
        return p

    def _style_plot(self, p):
        """Apply consistent styling to plots"""
        p.title.align = 'center'
        p.title.text_font_size = '16pt'
        p.title.text_font = 'serif'

        p.axis.axis_label_text_font_size = '12pt'
        p.axis.axis_label_text_font_style = 'bold'
        p.axis.major_label_text_font_size = '10pt'

        p.legend.location = 'top_right'
        p.legend.click_policy = 'hide'
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_alpha = 0.1

    def create_dashboard(self, output_file_path=None):
        """Create complete dashboard with all visualizations"""
        self.logger.info("Creating complete dashboard...")
        
        # Load data if not already loaded
        if not hasattr(self, 'df_quakes'):
            self.load_data()

        # Create individual plots
        map_plot = self.create_earthquake_map()
        trend_plot = self.create_magnitude_trend()
        depth_plot = self.create_depth_heatmap()

        # Arrange plots in a grid
        grid = gridplot([
            [map_plot],
            [trend_plot, depth_plot]
        ])

        # Save to file if path provided
        if output_file_path:
            output_file(output_file_path)
        
        return grid

def main():
    # Example usage
    viz = EarthquakeVisualization()
    viz.load_data()
    dashboard = viz.create_dashboard("earthquake_dashboard.html")
    show(dashboard)

if __name__ == "__main__":
    main()