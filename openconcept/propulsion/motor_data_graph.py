import pandas as pd
import numpy as np

class MotorDataGraphEff:
    """Global data store for motor training data from H3X_HPDM_2300_eff.xlsx - loads once, accessible everywhere"""
    rpm_data = None
    torque_data = None
    eff_data = None
    
    @classmethod
    def load_data(cls, motor_filename='openconcept/propulsion/empirical_data/H3X_HPDM_2300_eff.xlsx'):
        """Load data once and store it in the class variables"""
        if cls.voltage_data is None:  # Only load if not already loaded
            # Read the Excel file (assumes headers are in the first row)
            df = pd.read_excel(motor_filename)
            
            # Drop any rows with NaN values
            print(f"Original dataframe shape: {df.shape}")
            df = df.dropna()
            print(f"Dataframe shape after dropping NaNs: {df.shape}")
            
            # Assign data from columns (adjust column indices based on your Excel structure)
            cls.rpm_data = df['RPM'].values
            cls.torque_data = df['Torque(Nm)'].values
            cls.eff_data = df['Efficiency'].values
            
            # Define your input columns
            input_cols = ['RPM','Torque(Nm)','Efficiency']  # replace with your actual input columns
            
            # Find all rows where the input combination is duplicated (singular points)
            dupe_input_mask = df.duplicated(subset=input_cols, keep='first')
            
            # Show all rows with duplicate input points
            print(df[dupe_input_mask])
            print(f"Number of singular (duplicate input) points: {dupe_input_mask.sum()}")
            
            print("Motor data from H3X_HPDM_2300_eff.xlsx loaded successfully!")
    
    @classmethod
    def get_data(cls):
        """Ensure data is loaded and return all data as a tuple"""
        if cls.voltage_data is None:
            cls.load_data()
        return cls


class MotorDataGraphVolts:
    """Global data store for motor training data from H3X_HPDM_2300_volts.xlsx - loads once, accessible everywhere"""
    voltage_data = None
    power_data = None
    rpm_data = None
    
    @classmethod
    def load_data(cls, motor_filename='openconcept/propulsion/empirical_data/H3X_HPDM_2300_volts.xlsx'):
        """Load data once and store it in the class variables"""
        if cls.voltage_data is None:  # Only load if not already loaded
            # Read the Excel file (assumes headers are in the first row)
            df = pd.read_excel(motor_filename)
            
            # Drop any rows with NaN values
            print(f"Original dataframe shape: {df.shape}")
            df = df.dropna()
            print(f"Dataframe shape after dropping NaNs: {df.shape}")

            # Assign data from columns (adjust column indices based on your Excel structure)
            cls.voltage_data = df['Voltage'].values
            cls.power_data = df['Power(kW)'].values
            cls.rpm_data = df['RPM'].values

            # Define your input columns
            input_cols = ['RPM','Power(kW)','Voltage']  # replace with your actual input columns
            
            # Find all rows where the input combination is duplicated (singular points)
            dupe_input_mask = df.duplicated(subset=input_cols, keep='first')
            
            # Show all rows with duplicate input points
            print(df[dupe_input_mask])
            print(f"Number of singular (duplicate input) points: {dupe_input_mask.sum()}")
            
            print("Motor data from H3X_HPDM_2300_volts.xlsx loaded successfully!")
    
    @classmethod
    def get_data(cls):
        """Ensure data is loaded and return all data as a tuple"""
        if cls.voltage_data is None:
            cls.load_data()
        return cls
