import pandas as pd
import numpy as np

class BatteryData:
    """Global data store for battery training data - loads once, accessible everywhere"""
    soc_data = None
    c_rate_data = None
    ir0_data = None
    ocv_data = None
    line_voltage_data = None
    m_cell = None
    Cp_cell = None
    cell_Ah_capacity = None
    limit_min_temp = None
    limit_max_temp = None
    
    # Meshgrid data for 2D interpolation
    soc_mesh_flat = None
    c_rate_mesh_flat = None
    line_voltage_flat = None
    ir0_flat = None
    
    @classmethod
    def load_data(cls,bat_filename, cell_sheetname, config_sheetname):
        """Load data once and store it in the class variables"""
        if cls.soc_data is None:  # Only load if not already loaded
            df_cell = pd.read_excel(bat_filename, sheet_name=cell_sheetname)
            df_batconfig = pd.read_excel(bat_filename, sheet_name=config_sheetname)
            
            cls.soc_data = df_cell.SOC.to_numpy()
            # Round any 0 SOC values up to 1e-6
            cls.soc_data[cls.soc_data == 0] = 1e-6
            
            cls.c_rate_data = df_cell.Crate_value.dropna().to_numpy()
            cls.ir0_data =df_cell[df_cell.Crate_col.dropna()].to_numpy()/1000
            cls.ocv_data = df_cell.OCV.to_numpy()
            cls.line_voltage_data = cls.ocv_data.reshape((-1,1)) - np.multiply(cls.ir0_data, cls.c_rate_data)

            cls.m_cell = df_cell.mass_cell.iloc[0]
            cls.cp_cell = df_cell.Cp_cell.iloc[0]
            cls.cell_Ah_capacity = df_cell.capacity_Ah.iloc[0]
            cls.limit_min_temp = df_cell.min_temp.iloc[0]
            cls.limit_max_temp = df_cell.max_temp.iloc[0]

            cls.n_str = df_batconfig.Group.iloc[-1]
            cls.n_parallel_per_str = df_batconfig.nparallels.iloc[0]
            cls.n_series_per_str = df_batconfig.nseries.iloc[0]
            cls.n_motors = df_batconfig.motor_id.iloc[0][-1]

            cls.soc_init = cls.soc_data[0]

            # Generate meshgrid data for 2D interpolation
            cls._generate_meshgrid_data()
            
            print("Battery data loaded successfully!")
            print(f"SOC range: {cls.soc_data.min():.2e} to {cls.soc_data.max():.2f}")
    
    @classmethod
    def _generate_meshgrid_data(cls):
        """Generate flattened meshgrid data for 2D interpolation"""
        # Create meshgrid for training data
        soc_mesh, c_rate_mesh = np.meshgrid(cls.soc_data, cls.c_rate_data, indexing='ij')
        cls.soc_mesh_flat = soc_mesh.flatten()
        cls.c_rate_mesh_flat = c_rate_mesh.flatten()
        
        # Reshape the 2D data to match the meshgrid
        line_voltage_mesh = cls.line_voltage_data.reshape(soc_mesh.shape)
        ir0_mesh = cls.ir0_data.reshape(soc_mesh.shape)
        cls.line_voltage_flat = line_voltage_mesh.flatten()
        cls.ir0_flat = ir0_mesh.flatten()
        
        print(f"Generated meshgrid data: {len(cls.soc_mesh_flat)} points ({len(cls.soc_data)} SOC × {len(cls.c_rate_data)} C-rate)")
    
    @classmethod
    def get_data(cls):
        """Ensure data is loaded and return all data as a tuple"""
        if cls.soc_data is None:
            cls.load_data()
        return cls