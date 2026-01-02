"""
Logger for tracking training metrics and writing to TensorBoard and CSV.
"""

import os
import csv
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any


class Logger:
    """
    Logger class for tracking training metrics.
    
    Writes to both TensorBoard and CSV files for comprehensive logging.
    
    Args:
        log_dir: Directory to save logs
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        
        # CSV file for metrics
        self.csv_path = os.path.join(log_dir, 'metrics.csv')
        self.csv_file = None
        self.csv_writer = None
        self.csv_columns = None
        
        # Track logged data
        self.episode_count = 0
    
    def log(self, metrics: Dict[str, Any]):
        """
        Log metrics to TensorBoard and CSV.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        episode = metrics.get('episode', self.episode_count)
        
        # Write to TensorBoard
        for key, value in metrics.items():
            if key != 'episode' and isinstance(value, (int, float)):
                self.tb_writer.add_scalar(key, value, episode)
        
        # Write to CSV
        if self.csv_file is None:
            # Initialize CSV file with headers
            self.csv_columns = list(metrics.keys())
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_columns)
            self.csv_writer.writeheader()
        
        # Write row
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()  # Ensure data is written immediately
        
        self.episode_count += 1
    
    def close(self):
        """Close logger and flush remaining data."""
        if self.csv_file is not None:
            self.csv_file.close()
        self.tb_writer.close()
    
    def __del__(self):
        """Cleanup when logger is deleted."""
        self.close()
