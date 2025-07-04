import numpy as np
import openmdao.api as om

import numpy as np
import cProfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
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
    
    # Get clean test data across all conditions (idle, static, dynamic)
    test_data = TurboData.load_test_data(turbo_filename='openconcept/propulsion/empirical_data/PT6E-67XP-4_EngData.xlsx', sheet_name='CRZ', n_points=100, seed=42)
    
    n_test_points = len(test_data['Altitude'])
    print(f"Testing {n_test_points} points across all conditions...")
    
    # Set up the OpenMDAO model for testing
    num_nodes = n_test_points
    
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables with the test data points
    ivc.add_output('fltcond|h', test_data['Altitude'], units='m', desc='Altitude in meters')
    ivc.add_output('fltcond|M', test_data['Mach'], desc='Mach number')
    ivc.add_output('fltcond|disa', test_data['DISA'], desc='DISA in degrees Celsius')
    ivc.add_output('throttle', test_data['FRAC'], desc='Throttle fraction')

    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('fuel_flow_group', TurboFuelFlowRBF(num_nodes=num_nodes, throttle_set=True), promotes=["*"])

    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Run the model with the test data points
    prob.run_model()
    
    # Get interpolated values
    interpolated_values = prob.get_val('fuel_flow_kgph')
    actual_values = test_data['FuelFlow_kgph']
    
    # Calculate statistics
    errors = np.abs(interpolated_values - actual_values)
    relative_errors = errors / actual_values * 100
    
    print(f"Mean absolute error: {np.mean(errors):.2f} kg/h")
    print(f"Max absolute error: {np.max(errors):.2f} kg/h")
    print(f"Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"Max relative error: {np.max(relative_errors):.2f}%")
    
    # Calculate statistics by condition type
    condition_types = test_data['condition_type']
    for condition in ['idle', 'static', 'dynamic']:
        condition_mask = [c == condition for c in condition_types]
        if any(condition_mask):
            condition_errors = errors[condition_mask]
            condition_relative_errors = relative_errors[condition_mask]
            print(f"\n{condition.capitalize()} conditions:")
            print(f"  Mean absolute error: {np.mean(condition_errors):.2f} kg/h")
            print(f"  Mean relative error: {np.mean(condition_relative_errors):.2f}%")
            print(f"  Number of points: {sum(condition_mask)}")
    
    # Create the comparison plot
    plt.figure(figsize=(10, 8))
    
    # Plot actual vs interpolated
    plt.subplot(2, 2, 1)
    plt.scatter(actual_values, interpolated_values, alpha=0.6, s=20)
    plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Fuel Flow (kg/h)')
    plt.ylabel('Interpolated Fuel Flow (kg/h)')
    plt.title('All Conditions: Actual vs Interpolated')
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
    
    # Class-level interpolators (shared across all instances)
    _dyn_fuel_flow_interpolator = None
    _dyn_power_interpolator = None
    _idle_fuel_flow_interpolator = None
    _idle_power_interpolator = None
    _stat_fuel_flow_interpolator = None
    _stat_power_interpolator = None
    _interpolators_built = False
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('throttle_set', default=True, desc='Set throttle or power for gas turbine')
        # Load data immediately when component is initialized
        TurboData.load_data(turbo_filename='openconcept/propulsion/empirical_data/PT6E-67XP-4_EngData.xlsx', sheet_name='CRZ')
        
        # Build shared interpolators once when class is first initialized
        self._build_shared_interpolators()

    @classmethod
    def _build_shared_interpolators(cls):
        """Build interpolators once for all instances"""
        if cls._interpolators_built:
            return
            
        print("Building shared turbo interpolators...")
        
        # Build dynamic interpolator
        alt_data = TurboData.dyn_alt_data__m
        mach_data = TurboData.dyn_mach_data
        disa_data = TurboData.dyn_disa_data__degC
        fuel_flow_data = TurboData.dyn_fuel_flow_data__kgph
        power_data_kW = TurboData.dyn_power_data__kW
        throttle_data = TurboData.dyn_frac_data
        
        X_train = np.column_stack([alt_data, mach_data, disa_data, throttle_data])
        cls._dyn_fuel_flow_interpolator = RBFInterpolator(X_train, fuel_flow_data, kernel='thin_plate_spline')
        #cls._dyn_fuel_flow_interpolator = LinearNDInterpolator(X_train, fuel_flow_data)

        print("Dynamic fuel flow interpolator built successfully!")
        
        cls._dyn_power_interpolator = RBFInterpolator(X_train, power_data_kW, kernel='thin_plate_spline')
        #cls._dyn_power_interpolator = LinearNDInterpolator(X_train, power_data_kW)
        print("Dynamic power interpolator built successfully!")
        # Build idle interpolators
        idle_alt_data = TurboData.idle_alt_data__m
        idle_fuel_flow_data = TurboData.idle_fuel_flow_data__kgph
        idle_power_data = TurboData.idle_power_data__kW
        
        cls._idle_fuel_flow_interpolator = UnivariateSpline(idle_alt_data, idle_fuel_flow_data, s=0)
        print("Idle fuel flow interpolator built successfully!")
        cls._idle_power_interpolator = UnivariateSpline(idle_alt_data, idle_power_data, s=0)
        print("Idle power interpolator built successfully!")
        
        # Build static interpolators
        stat_alt_data = TurboData.stat_alt_data__m
        stat_disa_data = TurboData.stat_disa_data__degC
        stat_power_data_kW = TurboData.stat_power_data__kW
        stat_frac_data = TurboData.stat_frac_data
        stat_fuel_flow_data = TurboData.stat_fuel_flow_data__kgph
        
        X_train_static = np.column_stack([stat_alt_data, stat_disa_data, stat_frac_data])
        cls._stat_fuel_flow_interpolator = RBFInterpolator(X_train_static, stat_fuel_flow_data, kernel='thin_plate_spline')
        print("Static fuel flow interpolator built successfully!")
        cls._stat_power_interpolator = RBFInterpolator(X_train_static, stat_power_data_kW, kernel='thin_plate_spline')
        print("Static power interpolator built successfully!")
        
        cls._interpolators_built = True
        print("Shared turbo interpolators built successfully!")
    
    def setup(self):
        print("Setting up TurboFuelFlowRBF...")
        nn = self.options['num_nodes']
        
        # Inputs
        self.add_input('fltcond|h', val=10000.0, units='m', desc='Altitude', shape=(nn,))
        self.add_input('fltcond|M', val=0.5, desc='Mach number', shape=(nn,))
        self.add_input('fltcond|disa', val=0.0, desc='Temperature deviation from ISA', shape=(nn,))

        if self.options['throttle_set']:
            self.add_input('throttle', val=0.6, units=None, desc='Throttle fraction', shape=(nn,))
            self.add_output('power', val=1000.0, units='kW', desc='Power output', shape=(nn,))
        else:
            self.add_input('power', val=1000.0, units='kW', desc='Power output', shape=(nn,))
        # end
        # Outputs
        self.add_output('fuel_flow_kgph', val=200.0, units='kg/h', desc='Fuel flow rate', shape=(nn,))
        
        # Declare partials
        self.declare_partials('*', '*', method='fd')
        


    def compute(self, inputs, outputs):
        print("Computing Fuel Flow...")
        throttle_set = self.options['throttle_set']
        
        altitude = inputs['fltcond|h']
        mach = inputs['fltcond|M']
        disa = inputs['fltcond|disa']
        
        if throttle_set:
            throttle = inputs['throttle']
            # Define conditions
            idle_condition = (throttle == 0)
            static_condition = np.logical_and(mach == 0, throttle != 0)
            
            # Get indices for each 
            idle_indices = np.where(idle_condition)[0]
            static_indices = np.where(static_condition)[0]
            dynamic_indices = np.where(~idle_condition & ~static_condition)[0]
            
            # Initialize output array
            fuel_flow = np.zeros_like(altitude)
            power = np.zeros_like(altitude)
            
            # Handle idle points
            if len(idle_indices) > 0:
                fuel_flow_idle = self._idle_fuel_flow_interpolator(altitude[idle_indices])
                fuel_flow[idle_indices] = fuel_flow_idle
                power[idle_indices] = self._idle_power_interpolator(altitude[idle_indices])

            # Handle static points
            if len(static_indices) > 0:
                X_query_static = np.column_stack([
                    altitude[static_indices], 
                    disa[static_indices], 
                    throttle[static_indices]
                ])
                fuel_flow_static = self._stat_fuel_flow_interpolator(X_query_static)
                fuel_flow[static_indices] = fuel_flow_static
                power[static_indices] = self._stat_power_interpolator(X_query_static)
            
            # Handle dynamic points
            if len(dynamic_indices) > 0:
                X_query_dynamic = np.column_stack([
                    altitude[dynamic_indices], 
                    mach[dynamic_indices], 
                    disa[dynamic_indices], 
                    throttle[dynamic_indices]
                ])
                fuel_flow_dynamic = self._dyn_fuel_flow_interpolator(X_query_dynamic)
                fuel_flow[dynamic_indices] = fuel_flow_dynamic
                power[dynamic_indices] = self._dyn_power_interpolator(X_query_dynamic)

            outputs['power'] = power
        
        else:
            power = inputs['power']
            # Define conditions
            static_condition = np.logical_and(mach == 0, power > 0)
            
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
                fuel_flow_static = self._stat_fuel_flow_interpolator(X_query_static)
                fuel_flow[static_indices] = fuel_flow_static
            
            # Handle dynamic points
            if len(dynamic_indices) > 0:
                X_query_dynamic = np.column_stack([
                    altitude[dynamic_indices], 
                    mach[dynamic_indices], 
                    disa[dynamic_indices], 
                    power[dynamic_indices]
                ])
                fuel_flow_dynamic = self._dyn_fuel_flow_interpolator(X_query_dynamic)
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


def test_rbf_component():

    num_nodes = 30
    
    # Set up the OpenMDAO model
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables with vectorization
    ivc.add_output('fltcond|h', 10000*0.3048 * np.ones(num_nodes), units='m', desc='Altitude in meters')
    ivc.add_output('fltcond|M', np.linspace(0, 0.4, num_nodes), desc='Mach number')
    ivc.add_output('fltcond|disa', 20 * np.ones(num_nodes), desc='DISA in degrees Celsius')
    ivc.add_output('throttle', np.linspace(0, 1, num_nodes), desc='Throttle fraction')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('dyn_turbo_group', TurboFuelFlowRBF(num_nodes=num_nodes), promotes=["*"])
    #model.add_subsystem('stat_turbo_group', EmpiricalStaticTurbo(num_nodes=num_nodes), promotes=["*"])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Now test out a 'fuzzy' XOR

    
    prob.run_model()
    print("Fuel Flow [kg/h]:")
    print(prob.get_val('fuel_flow_kgph'))
    print("Power [kW]:")
    print(prob.get_val('power'))
    
    

if __name__ == "__main__":
    


    
    #test_rbf_component()
    # Test interpolation accuracy
    test_interpolation_accuracy()
    
