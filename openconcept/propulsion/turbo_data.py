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

    # Test data
    test_alt_data__m = None
    test_mach_data = None
    test_disa_data__degC = None
    test_throttle_data = None
    test_power_data__kW = None
    test_fuel_flow_data__kgph = None
    test_condition_type = None  # 'idle', 'static', or 'dynamic'


    
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

    @classmethod
    def load_test_data(cls, turbo_filename, sheet_name='CRZ', n_points=100, seed=42):
        """
        Load clean test data by combining static, dynamic, and idle conditions.
        Removes NaNs and duplicates, then samples n random points.
        
        Parameters
        ----------
        n_points : int
            Number of test points to sample
        seed : int
            Random seed for reproducible sampling
            
        Returns
        -------
        dict
            Dictionary containing test data arrays and condition types
        """


        df_test = pd.read_excel(turbo_filename, sheet_name=sheet_name)
        
        # Rename columns
        df_test.columns = [
            'DISA', 'Altitude_ft', 'Mach', 'FRAC', 'power_SHP', 'FuelFlow_pph', 'JetThrust_lbf',
            'Altitude', 'power_kW', 'FuelFlow_kgph', 'JetThrust_N'
        ]

        # Define your input columns
        input_cols = ['DISA','Altitude','Mach','FRAC','power_kW','FuelFlow_kgph']  # replace with your actual input columns
        
        # Find all rows where the input combination is duplicated (singular points)
        df_test.drop_duplicates(subset=input_cols, keep='first')

        # Drop any rows with NaN values
        df_test = df_test.dropna()
        
        
        # Sample n_points randomly
        if len(df_test) > n_points:
            np.random.seed(seed)
            sample_indices = np.random.choice(len(df_test), n_points, replace=False)
            df_test = df_test.iloc[sample_indices].reset_index(drop=True)
        else:
            print(f"Warning: Only {len(df_test)} points available, using all of them")
        
        # Store test data in class variables
        cls.test_alt_data__m = df_test['Altitude'].values
        cls.test_mach_data = df_test['Mach'].values
        cls.test_disa_data__degC = df_test['DISA'].values
        cls.test_throttle_data = df_test['FRAC'].values
        cls.test_power_data__kW = df_test['power_kW'].values
        cls.test_fuel_flow_data__kgph = df_test['FuelFlow_kgph'].values

        # Add simple condition mask using vectorized operations
        idle_mask = df_test['FRAC'] == 0
        static_mask = (df_test['Mach'] == 0) & (df_test['FRAC'] != 0)
        dynamic_mask = (df_test['Mach'] != 0) & (df_test['FRAC'] != 0)
        
        cls.test_condition_type = np.where(idle_mask, 'idle', 
                                          np.where(static_mask, 'static', 'dynamic'))

        print(f"Test data loaded: {len(df_test)} points ({np.sum(idle_mask)} idle, "
              f"{np.sum(static_mask)} static, "
              f"{np.sum(dynamic_mask)} dynamic)")

        return {
            'Altitude': cls.test_alt_data__m,
            'Mach': cls.test_mach_data,
            'DISA': cls.test_disa_data__degC,
            'FRAC': cls.test_throttle_data,
            'power_kW': cls.test_power_data__kW,
            'FuelFlow_kgph': cls.test_fuel_flow_data__kgph,
            'condition_type': cls.test_condition_type
        } 