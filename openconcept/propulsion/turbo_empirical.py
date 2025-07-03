import numpy as np
import openmdao.api as om

import numpy as np
import cProfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

import json

import os, sys

# Import the global data store
from openconcept.propulsion.turbo_data import TurboData



def test_interpolation_accuracy():
    """
    Test interpolation accuracy by comparing interpolated vs actual values
    """
    print("Testing interpolation accuracy...")
    
    # Get the data
    turbo_data = TurboData.get_data(turbo_filename='openconcept/propulsion/empirical_data/PT6E-67XP-4_EngData.xlsx', sheet_name='CRZ')
    
    # Randomly sample 100 points from dynamic conditions
    n_test_points = 100
    n_total_points = len(turbo_data.dyn_fuel_flow_data__kgph)
    
    # Generate random indices
    np.random.seed(42)  # For reproducible results
    test_indices = np.random.choice(n_total_points, n_test_points, replace=False)
    
    print(f"Testing {n_test_points} randomly sampled points from {n_total_points} total dynamic data points...")
    
    # Set up the OpenMDAO model for testing
    num_nodes = n_test_points
    
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables with the randomly sampled data points
    ivc.add_output('fltcond|h', turbo_data.dyn_alt_data__m[test_indices], units='m', desc='Altitude in meters')
    ivc.add_output('fltcond|M', turbo_data.dyn_mach_data[test_indices], desc='Mach number')
    ivc.add_output('fltcond|disa', turbo_data.dyn_disa_data__degC[test_indices], desc='DISA in degrees Celsius')
    #ivc.add_output('throttle', turbo_data.dyn_frac_data[test_indices], desc='Throttle fraction')
    ivc.add_output('power', turbo_data.dyn_power_data__kW[test_indices], units='kW', desc='Power output')

    model.add_subsystem('ivc', ivc, promotes=['*'])
    #model.add_subsystem('power_solver', TurboFuelFlowFromPower(num_nodes=num_nodes), promotes=["*"])
    model.add_subsystem('fuel_flow_group', TurboFuelFlowFromPowerMetaModel(num_nodes=num_nodes), promotes=["*"])

    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Run the model with the randomly sampled data points
    prob.run_model()
    
    # Get interpolated values
    interpolated_values = prob.get_val('fuel_flow_kgph')
    actual_values = turbo_data.dyn_fuel_flow_data__kgph[test_indices]
    
    # Calculate statistics
    errors = np.abs(interpolated_values - actual_values)
    relative_errors = errors / actual_values * 100
    
    print(f"Mean absolute error: {np.mean(errors):.2f} kg/h")
    print(f"Max absolute error: {np.max(errors):.2f} kg/h")
    print(f"Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"Max relative error: {np.max(relative_errors):.2f}%")
    
    # Create the comparison plot
    plt.figure(figsize=(10, 8))
    
    # Plot actual vs interpolated
    plt.subplot(2, 2, 1)
    plt.scatter(actual_values, interpolated_values, alpha=0.6, s=20)
    plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Fuel Flow (kg/h)')
    plt.ylabel('Interpolated Fuel Flow (kg/h)')
    plt.title('Dynamic Conditions: Actual vs Interpolated (100 Random Points)')
    plt.grid(True, alpha=0.3)
    
    # Plot error distribution
    plt.subplot(2, 2, 2)
    plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error (kg/h)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot relative error distribution
    plt.subplot(2, 2, 3)
    plt.hist(relative_errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    plt.title('Relative Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot error vs actual value
    plt.subplot(2, 2, 4)
    plt.scatter(actual_values, relative_errors, alpha=0.6, s=20)
    plt.xlabel('Actual Fuel Flow (kg/h)')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs Actual Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return interpolated_values, actual_values, errors, relative_errors


class TurboFuelFlowRBF(om.ExplicitComponent):
    """
    Explicit component that determines fuel flow as a function of flight conditions and power.
    
    This component uses RBF interpolation to find fuel flow given:
    - Flight conditions (altitude, Mach, DISA)
    - Power output
    
    Parameters
    ----------
    num_nodes : int
        Number of analysis points (default: 1)
    
    Inputs
    ------
    fltcond|h : array_like
        Altitude [m]
    fltcond|M : array_like
        Mach number
    fltcond|disa : array_like
        Temperature deviation from ISA [degC]
    power : array_like
        Power output [kW]
    
    Outputs
    -------
    fuel_flow_kgph : array_like
        Fuel flow rate [kg/h]
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('throttle_set', default=True, desc='Set throttle or power for gas turbine')
        # Load data immediately when component is initialized
        TurboData.load_data(turbo_filename='openconcept/propulsion/empirical_data/PT6E-67XP-4_EngData.xlsx', sheet_name='CRZ')

    
    def setup(self):
        nn = self.options['num_nodes']
        
        # Inputs
        self.add_input('fltcond|h', val=10000.0, units='m', desc='Altitude', shape=(nn,))
        self.add_input('fltcond|M', val=0.5, desc='Mach number', shape=(nn,))
        self.add_input('fltcond|disa', val=0.0, desc='Temperature deviation from ISA', shape=(nn,))

        if self.options['throttle_set']:
            self.add_input('throttle', val=0.6, units=None, desc='Throttle fraction', shape=(nn,))
        else:
            self.add_input('power', val=1000.0, units='kW', desc='Power output', shape=(nn,))
        # end
        # Outputs
        self.add_output('fuel_flow_kgph', val=200.0, units='kg/h', desc='Fuel flow rate', shape=(nn,))
        
        # Declare partials
        self.declare_partials('*', '*', method='fd')
        
        # Build interpolator in setup
        self._build_dyn_interpolator()
        self._build_idle_interpolator_fuel_flow()
        self._build_idle_interpolator_power()
        self._build_stat_interpolator()

    def _build_stat_interpolator(self):
        """Build RBF interpolator for fuel flow based on static (Mach = 0, throttle > 0) data"""
        # Get training data
        alt_data = TurboData.stat_alt_data__m
        disa_data = TurboData.stat_disa_data__degC

        if self.options['throttle_set']:
            frac_data = TurboData.stat_frac_data
            X_train = np.column_stack([alt_data, disa_data, frac_data])
        else:
            power_data_kW = TurboData.stat_power_data__kW
            X_train = np.column_stack([alt_data, disa_data, power_data_kW])
        # end

        fuel_flow_data = TurboData.stat_fuel_flow_data__kgph
        
        # Create RBF interpolator
        self.stat_fuel_flow_interpolator = RBFInterpolator(X_train, fuel_flow_data, kernel='thin_plate_spline')

    def _build_idle_interpolator_fuel_flow(self):
        """Build interpolator for fuel flow based on idle (throttle = 0) data"""
        # Get training data
        alt_data = TurboData.idle_alt_data__m
        fuel_flow_data = TurboData.idle_fuel_flow_data__kgph

        # Data is already cleaned in TurboData.load_data()
        self.idle_fuel_flow_interpolator = UnivariateSpline(alt_data, fuel_flow_data, s=0)

    def _build_idle_interpolator_power(self):
        """Build interpolator for power based on idle (throttle = 0) data"""
        # Get training data
        alt_data = TurboData.idle_alt_data__m
        power_data_kW = TurboData.idle_power_data__kW

        # Data is already cleaned in TurboData.load_data()
        self.idle_power_interpolator = UnivariateSpline(alt_data, power_data_kW, s=0)

    def _build_dyn_interpolator(self):
        """Build RBF interpolator for fuel flow based on dynamic (Mach > 0, throttle > 0) data"""
        # Get training data
        alt_data = TurboData.dyn_alt_data__m
        mach_data = TurboData.dyn_mach_data
        disa_data = TurboData.dyn_disa_data__degC
        fuel_flow_data = TurboData.dyn_fuel_flow_data__kgph

        if self.options['throttle_set']:
            throttle_data = TurboData.dyn_frac_data
            X_train = np.column_stack([alt_data, mach_data, disa_data, throttle_data])

        else:
            power_data_kW = TurboData.dyn_power_data__kW
            X_train = np.column_stack([alt_data, mach_data, disa_data, power_data_kW])
        # end

        # Create RBF interpolator
        self.dyn_fuel_flow_interpolator = RBFInterpolator(X_train, fuel_flow_data, kernel='thin_plate_spline')
    
    def compute(self, inputs, outputs):
        throttle_set = self.options['throttle_set']
        
        altitude = inputs['fltcond|h']
        mach = inputs['fltcond|M']
        disa = inputs['fltcond|disa']
        
        if throttle_set:
            throttle = inputs['throttle']
            # Define conditions
            idle_condition = (throttle == 0)
            static_condition = (mach == 0)
            
            # Get indices for each 
            idle_indices = np.where(idle_condition)[0]
            static_indices = np.where(static_condition)[0]
            dynamic_indices = np.where(~idle_condition & ~static_condition)[0]
            
            # Initialize output array
            fuel_flow = np.zeros_like(altitude)
            
            # Handle idle points
            if len(idle_indices) > 0:
                fuel_flow_idle = self.idle_fuel_flow_interpolator(altitude[idle_indices])
                fuel_flow[idle_indices] = fuel_flow_idle

            # Handle static points
            if len(static_indices) > 0:
                X_query_static = np.column_stack([
                    altitude[static_indices], 
                    disa[static_indices], 
                    throttle[static_indices]
                ])
                fuel_flow_static = self.stat_fuel_flow_interpolator(X_query_static)
                fuel_flow[static_indices] = fuel_flow_static
            
            # Handle dynamic points
            if len(dynamic_indices) > 0:
                X_query_dynamic = np.column_stack([
                    altitude[dynamic_indices], 
                    mach[dynamic_indices], 
                    disa[dynamic_indices], 
                    throttle[dynamic_indices]
                ])
                fuel_flow_dynamic = self.dyn_fuel_flow_interpolator(X_query_dynamic)
                fuel_flow[dynamic_indices] = fuel_flow_dynamic
        
        else:
            power = inputs['power']
            # Define conditions
            static_condition = (mach == 0)
            
            # Get indices for each condition
            static_indices = np.where(static_condition)[0]
            dynamic_indices = np.where(~static_condition)[0]
            
            # Initialize output array
            fuel_flow = np.zeros_like(altitude)
            
            # Handle static points
            if len(static_indices) > 0:
                X_query_static = np.column_stack([
                    altitude[static_indices], 
                    disa[static_indices], 
                    power[static_indices]
                ])
                fuel_flow_static = self.stat_fuel_flow_interpolator(X_query_static)
                fuel_flow[static_indices] = fuel_flow_static
            
            # Handle dynamic points
            if len(dynamic_indices) > 0:
                X_query_dynamic = np.column_stack([
                    altitude[dynamic_indices], 
                    mach[dynamic_indices], 
                    disa[dynamic_indices], 
                    power[dynamic_indices]
                ])
                fuel_flow_dynamic = self.dyn_fuel_flow_interpolator(X_query_dynamic)
                fuel_flow[dynamic_indices] = fuel_flow_dynamic
        
        outputs['fuel_flow_kgph'] = fuel_flow


class TurboFuelFlowFromPowerMetaModel(om.Group):
    """
    Group containing the turbo fuel flow metamodel component with vectorization support.
    
    This group uses MetaModelUnStructuredComp to interpolate fuel flow as a function of:
    - Flight conditions (altitude, Mach, DISA)
    - Power output
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('throttle_set', default=True, desc='Set throttle or power for gas turbine')
        # Load data immediately when module is imported - this ensures it's available for all components
        TurboData.load_data(turbo_filename='openconcept/propulsion/empirical_data/PT6E-67XP-4_EngData.xlsx', sheet_name='CRZ')

    
    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']
        throttle_set = self.options['throttle_set']
        
        # Create unstructured metamodel interpolator instance with vectorization
        fuel_flow_metamodel = om.MetaModelUnStructuredComp(vec_size=num_nodes, default_surrogate=om.ResponseSurface())
        
        # set up inputs and outputs with vectorization using global data
        fuel_flow_metamodel.add_input('fltcond|h', 10000.0, training_data=TurboData.dyn_alt_data__m, units="m", shape=(num_nodes,))
        fuel_flow_metamodel.add_input('fltcond|M', 0.5, training_data=TurboData.dyn_mach_data, units=None, shape=(num_nodes,))
        fuel_flow_metamodel.add_input('fltcond|disa', 0.0, training_data=TurboData.dyn_disa_data__degC, units=None, shape=(num_nodes,))

        if throttle_set:
            fuel_flow_metamodel.add_input('throttle', 0.6, training_data=TurboData.dyn_frac_data, units=None, shape=(num_nodes,))
        else:
            fuel_flow_metamodel.add_input('power', 1000.0, training_data=TurboData.dyn_power_data__kW, units="kW", shape=(num_nodes,))
        # end

        fuel_flow_metamodel.add_output('fuel_flow_kgph', 200.0, training_data=TurboData.dyn_fuel_flow_data__kgph, units="kg/h", shape=(num_nodes,))
        
        self.add_subsystem('fuel_flow_metamodel', fuel_flow_metamodel, promotes=["*"])


if __name__ == "__main__":
    


    num_nodes = 10
    
    # Set up the OpenMDAO model
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables with vectorization
    ivc.add_output('fltcond|h', 10000*0.3048 * np.ones(num_nodes), units='m', desc='Altitude in meters')
    ivc.add_output('fltcond|M', 0.01 * np.ones(num_nodes), desc='Mach number')
    ivc.add_output('fltcond|disa', 20 * np.ones(num_nodes), desc='DISA in degrees Celsius')
    ivc.add_output('throttle', 0.6 * np.ones(num_nodes), desc='Throttle fraction')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('dyn_turbo_group', TurboFuelFlowRBF(num_nodes=num_nodes), promotes=["*"])
    #model.add_subsystem('stat_turbo_group', EmpiricalStaticTurbo(num_nodes=num_nodes), promotes=["*"])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Now test out a 'fuzzy' XOR

    
    prob.run_model()
    
    print(prob.get_val('fuel_flow_kgph'))
    
    # we can verify all gradients by checking against finite-difference
    #prob.check_partials(compact_print=True)
    
    # Grid interpolation
    f_prop_thrust_interp = NearestNDInterpolator((TurboData.dyn_alt_data__m, TurboData.dyn_mach_data, TurboData.dyn_disa_data__degC, TurboData.dyn_frac_data), TurboData.dyn_fuel_flow_data__kgph)
    print(f"Fuel Flow grid interpolation  = {f_prop_thrust_interp((10000*0.3048, 0.01, 20, 0.6))}")
    
    # Test interpolation accuracy
    test_interpolation_accuracy()
    
