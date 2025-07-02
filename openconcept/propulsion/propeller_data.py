import pandas as pd
import numpy as np

class PropellerData:
    """Global data store for propeller training data - loads once, accessible everywhere"""
    
    # Static data (zero airspeed)
    stat_disa_data__K = None
    stat_dens_alt_data__m = None
    stat_tas_data__mps = None
    stat_rpm_data__rpm = None
    stat_rpm_data__radps = None
    stat_power_data__hp = None
    stat_power_data__W = None
    stat_J_data = None
    stat_tip_mac_data = None
    stat_Cp_data = None
    stat_Ct_data = None
    stat_eta_data = None
    stat_thrust_data__lbf = None
    stat_thrust_data__N = None
    stat_ktas_data__kts = None
    
    # Dynamic data (non-zero airspeed)
    dyn_disa_data__K = None
    dyn_dens_alt_data__m = None
    dyn_tas_data__mps = None
    dyn_rpm_data__rpm = None
    dyn_rpm_data__radps = None
    dyn_power_data__hp = None
    dyn_power_data__W = None
    dyn_J_data = None
    dyn_tip_mac_data = None
    dyn_Cp_data = None
    dyn_Ct_data = None
    dyn_eta_data = None
    dyn_thrust_data__lbf = None
    dyn_thrust_data__N = None
    dyn_ktas_data__kts = None
    
    @classmethod
    def load_data(cls, prop_filename, sheet_name='data'):
        """Load data once and store it in the class variables"""
        if cls.dyn_rpm_data__rpm is None:  # Only load if not already loaded
            # Read the Excel file
            df = pd.read_excel(prop_filename, sheet_name=sheet_name)
            
            # Drop any rows with NaN values
            print(f"Original dataframe shape: {df.shape}")
            df = df.dropna()
            print(f"Dataframe shape after dropping NaNs: {df.shape}")
            
            # Define your input columns
            #df['SHP'] = df['SHP'].astype(float)
            #df['densAlt_ft'] = df['densAlt_ft'].astype(float)
            
            #input_cols = ['SHP','RPM','KEAS','densAlt_ft','thrust_lbf', 'advJ']  # replace with your actual input columns
            input_cols = ['densAlt_ft','Ct','SHP', 'advJ']  # replace with your actual input columns

            # Find all rows where the input combination is duplicated (singular points)
            df.drop_duplicates(subset=input_cols, keep='first')
            print(f"Dataframe shape after dropping duplicates: {df.shape}")
            
            print(df)
            
            # Drop values at 0 throttle 
            df = df[df['SHP'] != 0]
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            # Split data into static and dynamic conditions
            stat_inds = df['KEAS'] == 0
            dyn_inds = df['KEAS'] != 0
            
            # Assign static data from columns
            disa_data__degC = df['disaDegC'].values
            dens_alt_data__ft = df['densAlt_ft'].values
            ktas_data__kts = df['KEAS'].values
            rpm_data__rpm = df['RPM'].values
            power_data__hp = df['SHP'].values
            J_data = df['advJ'].values
            tip_mac_data = df['tipMach'].values
            Cp_data = df['Cp'].values
            Ct_data = df['Ct'].values
            eta_data = df['eta'].values * 100
            thrust_data__lbf = df['thrust_lbf'].values
            
            # Convert to SI units
            disa_data__K = disa_data__degC + 273.15
            dens_alt_data__m = dens_alt_data__ft * 0.3048
            tas_data__mps = ktas_data__kts * 0.514444
            rpm_data__radps = rpm_data__rpm * 2 * np.pi / 60
            power_data__W = power_data__hp * 745.7  
            thrust_data__N = thrust_data__lbf * 4.44822
            
            # Split into static and dynamic
            cls.stat_disa_data__K = disa_data__K[stat_inds]
            cls.stat_dens_alt_data__m = dens_alt_data__m[stat_inds]
            cls.stat_tas_data__mps = tas_data__mps[stat_inds]
            cls.stat_rpm_data__rpm = rpm_data__rpm[stat_inds]
            cls.stat_rpm_data__radps = rpm_data__radps[stat_inds]
            cls.stat_power_data__hp = power_data__hp[stat_inds]
            cls.stat_power_data__W = power_data__W[stat_inds]
            cls.stat_J_data = J_data[stat_inds]
            cls.stat_tip_mac_data = tip_mac_data[stat_inds]
            cls.stat_Cp_data = Cp_data[stat_inds]
            cls.stat_Ct_data = Ct_data[stat_inds]
            cls.stat_eta_data = eta_data[stat_inds]
            cls.stat_thrust_data__lbf = thrust_data__lbf[stat_inds]
            cls.stat_thrust_data__N = thrust_data__N[stat_inds]
            cls.stat_ktas_data__kts = ktas_data__kts[stat_inds]
            
            # Assign dynamic data from columns
            cls.dyn_disa_data__K = disa_data__K[dyn_inds]
            cls.dyn_dens_alt_data__m = dens_alt_data__m[dyn_inds]
            cls.dyn_tas_data__mps = tas_data__mps[dyn_inds]
            cls.dyn_rpm_data__rpm = rpm_data__rpm[dyn_inds]
            cls.dyn_rpm_data__radps = rpm_data__radps[dyn_inds]
            cls.dyn_power_data__hp = power_data__hp[dyn_inds]
            cls.dyn_power_data__W = power_data__W[dyn_inds]
            cls.dyn_J_data = J_data[dyn_inds]
            cls.dyn_tip_mac_data = tip_mac_data[dyn_inds]
            cls.dyn_Cp_data = Cp_data[dyn_inds]
            cls.dyn_Ct_data = Ct_data[dyn_inds]
            cls.dyn_eta_data = eta_data[dyn_inds]
            cls.dyn_thrust_data__lbf = thrust_data__lbf[dyn_inds]
            cls.dyn_thrust_data__N = thrust_data__N[dyn_inds]
            cls.dyn_ktas_data__kts = ktas_data__kts[dyn_inds]
            
            print(f"Propeller data loaded successfully!")
            print(f"Static data points: {len(cls.stat_thrust_data__N)}")
            print(f"Dynamic data points: {len(cls.dyn_thrust_data__N)}")
    
    @classmethod
    def get_data(cls, prop_filename, sheet_name='data'):
        """Ensure data is loaded and return all data as a tuple"""
        if cls.dyn_rpm_data__rpm is None:
            cls.load_data(prop_filename, sheet_name)
        return cls 