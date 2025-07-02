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

import json

import os, sys

# Import the global data store
from openconcept.propulsion.turbo_data import TurboData



class EmpiricalDynamicTurbo(om.Group):
    """
    Group containing the turbo interpolation component with vectorization support.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        # Load data immediately when module is imported - this ensures it's available for all components
        TurboData.load_data(turbo_filename='openconcept/propulsion/empirical_data/PT6E-67XP-4_EngData.xlsx', sheet_name='CRZ')

    
    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']
        
        # Create regular grid interpolator instance with vectorization
        dyn_turbo_interp = om.MetaModelUnStructuredComp(vec_size=num_nodes, default_surrogate=om.ResponseSurface())
        
        # set up inputs and outputs with vectorization using global data
        dyn_turbo_interp.add_input('fltcond|disa', 1.0, training_data=TurboData.dyn_disa_data__degC, units=None, shape=(num_nodes,))
        dyn_turbo_interp.add_input('fltcond|h', 1.0, training_data=TurboData.dyn_alt_data__m, units="m", shape=(num_nodes,))
        dyn_turbo_interp.add_input('fltcond|M', 1.0, training_data=TurboData.dyn_mach_data, units=None, shape=(num_nodes,))
        dyn_turbo_interp.add_input('frac', 1.0, training_data=TurboData.dyn_frac_data, units=None, shape=(num_nodes,))
        dyn_turbo_interp.add_output('fuel_flow_kgph', 1.0, training_data=TurboData.dyn_fuel_flow_data__kgph, units="kg/h", shape=(num_nodes,))
        
        self.add_subsystem('dyn_turbo_interp', dyn_turbo_interp, promotes=["*"])

class EmpiricalStaticTurbo(om.Group):
    """
    Group containing the turbo interpolation component with vectorization support.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        # Load data immediately when module is imported - this ensures it's available for all components
        TurboData.load_data(turbo_filename='openconcept/propulsion/empirical_data/PT6E-67XP-4_EngData.xlsx', sheet_name='CRZ')

    
    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']
        
        # Create regular grid interpolator instance with vectorization
        stat_turbo_interp = om.MetaModelUnStructuredComp(vec_size=num_nodes, default_surrogate=om.ResponseSurface())
        
        # set up inputs and outputs with vectorization using global data
        stat_turbo_interp.add_input('fltcond|disa', 1.0, training_data=TurboData.stat_disa_data__degC, units=None, shape=(num_nodes,))
        stat_turbo_interp.add_input('fltcond|h', 1.0, training_data=TurboData.stat_alt_data__m, units="m", shape=(num_nodes,))
        stat_turbo_interp.add_input('fltcond|M', 1.0, training_data=TurboData.stat_mach_data, units=None, shape=(num_nodes,))
        stat_turbo_interp.add_input('frac', 1.0, training_data=TurboData.stat_frac_data, units=None, shape=(num_nodes,))
        stat_turbo_interp.add_output('fuel_flow_kgph', 1.0, training_data=TurboData.stat_fuel_flow_data__kgph, units="kg/h", shape=(num_nodes,))
        
        self.add_subsystem('stat_turbo_interp', stat_turbo_interp, promotes=["*"])


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
    ivc.add_output('frac', turbo_data.dyn_frac_data[test_indices], desc='Throttle fraction')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('dyn_turbo_group', EmpiricalDynamicTurbo(num_nodes=num_nodes), promotes=["*"])
    
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


if __name__ == "__main__":
    
    # Number of nodes to evaluate
    num_nodes = 10
    
    # Set up the OpenMDAO model
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables with vectorization
    ivc.add_output('fltcond|h', 10000*0.3048 * np.ones(num_nodes), units='m', desc='Altitude in meters')
    ivc.add_output('fltcond|M', 0.01 * np.ones(num_nodes), desc='Mach number')
    ivc.add_output('fltcond|disa', 20 * np.ones(num_nodes), desc='DISA in degrees Celsius')
    ivc.add_output('frac', 0.6 * np.ones(num_nodes), desc='Throttle fraction')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('dyn_turbo_group', EmpiricalDynamicTurbo(num_nodes=num_nodes), promotes=["*"])
    #model.add_subsystem('stat_turbo_group', EmpiricalStaticTurbo(num_nodes=num_nodes), promotes=["*"])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Now test out a 'fuzzy' XOR
    prob.set_val('fltcond|h', 10000*0.3048 * np.ones(num_nodes))
    prob.set_val('fltcond|M', 0.01 * np.ones(num_nodes))
    prob.set_val('fltcond|disa', 20 * np.ones(num_nodes))
    prob.set_val('frac', 0.6 * np.ones(num_nodes))
    
    prob.run_model()
    
    print(prob.get_val('fuel_flow_kgph'))
    
    # we can verify all gradients by checking against finite-difference
    prob.check_partials(compact_print=True)
    
    # Grid interpolation
    f_prop_thrust_interp = NearestNDInterpolator((TurboData.dyn_alt_data__m, TurboData.dyn_mach_data, TurboData.dyn_disa_data__degC, TurboData.dyn_frac_data), TurboData.dyn_fuel_flow_data__kgph)
    print(f"Fuel Flow grid interpolation  = {f_prop_thrust_interp((10000*0.3048, 0.5, 20, 0.6))}")
    
    # Test interpolation accuracy
    test_interpolation_accuracy()

