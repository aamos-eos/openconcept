import numpy as np
import openmdao.api as om

import numpy as np
import cProfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import json

import os, sys

# Import the global data store
from openconcept.propulsion.motor_data_tbl import MotorData

# Load data immediately when module is imported - this ensures it's available for all components
MotorData.load_data(motor_filename='openconcept/propulsion/empirical_data/H3X_HPDM-XXXX_1MW.xlsx')




class EmpiricalMotor(om.Group):
    """
    Group containing the motor interpolation component with vectorization support.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
    
    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']
        
        # Create regular grid interpolator instance with vectorization
        motor_interp = om.MetaModelUnStructuredComp(vec_size=num_nodes, default_surrogate=om.ResponseSurface())
        
        # set up inputs and outputs with vectorization using global data
        motor_interp.add_input('voltage', 1.0, training_data=MotorData.voltage_data, units=None, shape=(num_nodes,))
        motor_interp.add_input('power', 1.0, training_data=MotorData.power_data__W, units="W", shape=(num_nodes,))
        motor_interp.add_output('eff', 1.0, training_data=MotorData.eff_data, units=None, shape=(num_nodes,))
        
        self.add_subsystem('motor_interp', motor_interp, promotes=["*"])


def test_interpolation_accuracy():
    """
    Test interpolation accuracy by comparing interpolated vs actual values
    """
    print("Testing motor interpolation accuracy...")
    
    # Get the data
    motor_data = MotorData.get_data()
    
    # Not manum_nodesy data points
    n_test_points = len(motor_data.eff_data)
    n_total_points = len(motor_data.eff_data)
    
    # Generate random indices
    np.random.seed(42)  # For reproducible results
    test_indices = np.random.choice(n_total_points, n_test_points, replace=False)
    
    print(f"Testing {n_test_points} randomly sampled points from {n_total_points} total motor data points...")
    
    # Set up the OpenMDAO model for testing
    num_nodes = n_test_points
    
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables with the randomly sampled data points
    ivc.add_output('voltage', motor_data.voltage_data[test_indices], units=None, desc='Motor voltage')
    ivc.add_output('power', motor_data.power_data__W[test_indices], units='W', desc='Motor power')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('motor_group', EmpiricalMotor(num_nodes=num_nodes), promotes=["*"])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Run the model with the randomly sampled data points
    prob.run_model()
    
    # Get interpolated values
    interpolated_values = prob.get_val('eff')
    actual_values = motor_data.eff_data[test_indices]
    
    # Calculate statistics
    errors = np.abs(interpolated_values - actual_values)
    relative_errors = errors / actual_values * 100
    
    print(f"Mean absolute error: {np.mean(errors):.4f}")
    print(f"Max absolute error: {np.max(errors):.4f}")
    print(f"Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"Max relative error: {np.max(relative_errors):.2f}%")
    
    # Create the comparison plot
    plt.figure(figsize=(10, 8))
    
    # Plot actual vs interpolated
    plt.subplot(2, 2, 1)
    plt.scatter(actual_values, interpolated_values, alpha=0.6, s=20)
    plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Efficiency')
    plt.ylabel('Interpolated Efficiency')
    plt.title('Motor: Actual vs Interpolated Efficiency (100 Random Points)')
    plt.grid(True, alpha=0.3)
    
    # Plot error distribution
    plt.subplot(2, 2, 2)
    plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error')
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
    plt.xlabel('Actual Efficiency')
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
    ivc.add_output('voltage', 700 * np.ones(num_nodes), units=None, desc='Motor voltage')
    ivc.add_output('power', 125 * 1000 * np.ones(num_nodes), units='W', desc='Motor power')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('motor_group', EmpiricalMotor(num_nodes=num_nodes), promotes=["*"])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Now test out a 'fuzzy' XOR
    prob.set_val('power', 125*1000 * np.ones(num_nodes))
    prob.set_val('voltage', 700 * np.ones(num_nodes))
    
    prob.run_model()
    
    print(prob.get_val('eff'))
    
    # we can verify all gradients by checking against finite-difference
    prob.check_partials(compact_print=True)
    
    # Test interpolation accuracy
    test_interpolation_accuracy()