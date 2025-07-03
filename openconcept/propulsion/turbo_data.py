import pandas as pd
import numpy as np

class TurboData:
    """Global data store for turbo training data - loads once, accessible everywhere"""

    # Static data
    stat_disa_data__degC = None
    stat_alt_data__ft = None
    stat_alt_data__m = None
    stat_mach_data = None
    stat_frac_data = None
    stat_power_data__kW = None
    stat_power_data__W = None
    stat_fuel_flow_data__kgph = None

    # Dynamic data
    dyn_disa_data__degC = None
    dyn_alt_data__ft = None
    dyn_alt_data__m = None
    dyn_mach_data = None
    dyn_frac_data = None
    dyn_power_data__kW = None
    dyn_power_data__W = None
    dyn_fuel_flow_data__kgph = None

    # Idle Data
    idle_alt_data__m = None
    idle_fuel_flow_data__kgph = None
    idle_power_data__kW = None


    
    @classmethod
    def load_data(cls, turbo_filename, sheet_name='CRZ'):
        """Load data once and store it in the class variables"""
        if cls.dyn_disa_data__degC is None:  # Only load if not already loaded
            # Read the Excel file
            df = pd.read_excel(turbo_filename, sheet_name=sheet_name)
            
            # Rename columns
            df.columns = [
                'DISA', 'Altitude_ft', 'Mach', 'FRAC', 'power_SHP', 'FuelFlow_pph', 'JetThrust_lbf',
                'Altitude', 'power_kW', 'FuelFlow_kgph', 'JetThrust_N'
            ]

            # Define your input columns
            input_cols = ['DISA','Altitude','Mach','FRAC','power_kW','FuelFlow_kgph']  # replace with your actual input columns
            
            # Find all rows where the input combination is duplicated (singular points)
            df.drop_duplicates(subset=input_cols, keep='first')

            # Drop any rows with NaN values
            df = df.dropna()

            # Get idle data. Most sensitive to altitude. Ignore changes in mach no. 
            df_idle = df[df['FRAC'] == 0]
            
            # Drop values at 0 throttle 
            df = df[df['FRAC'] != 0]

            # Split data into static and dynamic conditions
            stat_inds = df['Mach'] == 0
            dyn_inds = df['Mach'] != 0
            
            # Assign staticdata from columns
            cls.stat_disa_data__degC = df['DISA'].values[stat_inds]
            #cls.stat_alt_data__ft = df['Altitude_ft'].values[stat_inds]
            #cls.stat_mach_data = df['Mach'].values[stat_inds]
            cls.stat_frac_data = df['FRAC'].values[stat_inds]
            cls.stat_alt_data__m = df['Altitude'].values[stat_inds]
            cls.stat_power_data__kW = df['power_kW'].values[stat_inds]
            cls.stat_fuel_flow_data__kgph = df['FuelFlow_kgph'].values[stat_inds]

            # Assign dynamic data from columns
            cls.dyn_disa_data__degC = df['DISA'].values[dyn_inds]
            #cls.dyn_alt_data__ft = df['Altitude_ft'].values[dyn_inds]
            cls.dyn_mach_data = df['Mach'].values[dyn_inds]
            cls.dyn_frac_data = df['FRAC'].values[dyn_inds]
            cls.dyn_alt_data__m = df['Altitude'].values[dyn_inds]
            cls.dyn_power_data__kW = df['power_kW'].values[dyn_inds]
            cls.dyn_fuel_flow_data__kgph = df['FuelFlow_kgph'].values[dyn_inds]

            # Assign idle data from columns
            
            # Group by altitude and take the mean of fuel flow and power values. Ignore changes in delta ISA and Mach. Negligible
            idle_grouped = df_idle.groupby('Altitude').agg({
                'FuelFlow_kgph': 'mean',
                'power_kW': 'mean'
            }).reset_index()
            cls.idle_alt_data__m = idle_grouped['Altitude'].values
            cls.idle_fuel_flow_data__kgph = idle_grouped['FuelFlow_kgph'].values
            cls.idle_power_data__kW = idle_grouped['power_kW'].values

            # Convert to SI units
            cls.stat_power_data__W = cls.stat_power_data__kW * 1000
            cls.dyn_power_data__W = cls.dyn_power_data__kW * 1000

            print("Turbine data loaded successfully!")
    
    @classmethod
    def get_data(cls, turbo_filename, sheet_name='CRZ'):
        """Ensure data is loaded and return all data as a tuple"""
        if cls.dyn_disa_data__degC is None:
            cls.load_data(turbo_filename, sheet_name)
        return cls 