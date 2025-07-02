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
from scipy.optimize import minimize

import json

import os, sys

# Import the global data store
from openconcept.propulsion.turbo_data import TurboData


class TurboPowerSolver(om.ExplicitComponent):
    """
    Component that solves for throttle fraction (frac) to achieve desired power output.
    
    This component uses optimization to find the correct throttle setting given:
    - Flight conditions (altitude, Mach, DISA)
    - Desired power output
    
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
    desired_power : array_like
        Desired power output [W]
    
    Outputs
    -------
    frac : array_like
        Throttle fraction (0-1)
    fuel_flow_kgph : array_like
        Fuel flow rate [kg/h]
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        # Load data immediately when component is initialized
        TurboData.load_data(turbo_filename='openconcept/propulsion/empirical_data/PT6E-67XP-4_EngData.xlsx', sheet_name='CRZ')

    
    def setup(self):
        nn = self.options['num_nodes']
        
        # Inputs
        self.add_input('fltcond|h', val=10000.0, units='m', desc='Altitude', shape=(nn,))
        self.add_input('fltcond|M', val=0.5, desc='Mach number', shape=(nn,))
        self.add_input('fltcond|disa', val=0.0, desc='Temperature deviation from ISA', shape=(nn,))
        self.add_input('desired_power', val=1000.0, units='kW', desc='Desired power output', shape=(nn,))
        
        # Outputs
        self.add_output('frac', val=0.8, desc='Throttle fraction', shape=(nn,))
        
        # Use finite difference for partials since optimization is involved
        self.declare_partials('*', '*', method='fd')
    
    def _objective_function_vectorized(self, frac_vector, altitude, mach, disa, desired_power):
        """
        Vectorized objective function: minimize |actual_power - desired_power| for all nodes at once
        
        Parameters
        ----------
        frac_vector : array_like
            Throttle fraction vector to test
        altitude : array_like
            Altitude vector [m]
        mach : array_like
            Mach number vector
        disa : array_like
            Temperature deviation vector [degC]
        desired_power : array_like
            Desired power output vector [W]
        
        Returns
        -------
        array_like
            Absolute difference between actual and desired power for all nodes
        """
        # Use the existing interpolator from TurboData
        # Create query points for interpolation
        X_query = np.column_stack([altitude, mach, disa, frac_vector])
        
        # Use the same interpolator approach as in the main turbo component
        # Get training data
        alt_data = TurboData.dyn_alt_data__m
        mach_data = TurboData.dyn_mach_data
        disa_data = TurboData.dyn_disa_data__degC
        frac_data = TurboData.dyn_frac_data
        power_data_kW = TurboData.dyn_power_data__kW
        
        # Create training matrix
        X_train = np.column_stack([alt_data, mach_data, disa_data, frac_data])
        
        # Create RBF interpolator and interpolate fuel flow
        power_interpolator = RBFInterpolator(X_train, power_data_kW, kernel='thin_plate_spline')
        power_kW = power_interpolator(X_query)
        
        # Convert fuel flow to power (simplified relationship)
        actual_power = power_kW * 1000
        
        return np.abs(actual_power - desired_power)
    
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        
        altitude = inputs['fltcond|h']
        mach = inputs['fltcond|M']
        disa = inputs['fltcond|disa']
        desired_power = inputs['desired_power']
        
        # Initial guess for all nodes
        frac_initial = 0.8 * np.ones(nn)
        
        # Set bounds for throttle fraction (0.1 to 1.0) for all nodes
        bounds = [(0.1, 1.0)] * nn
        
        # Vectorized optimization
        result = minimize(
            lambda x: np.sum(self._objective_function_vectorized(x, altitude, mach, disa, desired_power)),
            frac_initial,
            bounds=bounds,
            method='Powell',
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if result.success:
            frac_optimal = result.x
        else:
            # If optimization fails, use reasonable defaults
            frac_optimal = 0.8 * np.ones(nn)
        
        # Get fuel flow for optimal throttle (vectorized)
        # Use the same interpolator approach as in the main turbo component
        alt_data = TurboData.dyn_alt_data__m
        mach_data = TurboData.dyn_mach_data
        disa_data = TurboData.dyn_disa_data__degC
        frac_data = TurboData.dyn_frac_data
        power_data_kW = TurboData.dyn_power_data__kW
        
        X_train = np.column_stack([alt_data, mach_data, disa_data, frac_data])
        X_query = np.column_stack([altitude, mach, disa, frac_optimal])
        
        power_interpolator = RBFInterpolator(X_train, power_data_kW, kernel='thin_plate_spline')
        power_kW_optimal = power_interpolator(X_query)
        
        outputs['frac'] = frac_optimal
        outputs['power_kW'] = power_kW_optimal


class TurboPowerImplicitSolver(om.ImplicitComponent):
    """
    Implicit component that solves for throttle fraction (frac) to achieve desired power output.
    
    This component uses the existing EmpiricalDynamicTurbo component and solves
    the implicit equation: actual_power - desired_power = 0
    
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
    desired_power : array_like
        Desired power output [W]
    
    Outputs
    -------
    frac : array_like
        Throttle fraction (0-1)
    fuel_flow_kgph : array_like
        Fuel flow rate [kg/h]
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        # Load data immediately when component is initialized
        TurboData.load_data(turbo_filename='openconcept/propulsion/empirical_data/PT6E-67XP-4_EngData.xlsx', sheet_name='CRZ')
    
    def setup(self):
        nn = self.options['num_nodes']
        
        # Inputs
        self.add_input('fltcond|h', val=10000.0, units='m', desc='Altitude', shape=(nn,))
        self.add_input('fltcond|M', val=0.5, desc='Mach number', shape=(nn,))
        self.add_input('fltcond|disa', val=0.0, desc='Temperature deviation from ISA', shape=(nn,))
        self.add_input('desired_power', val=1000.0, units='kW', desc='Desired power output', shape=(nn,))
        
        # Outputs
        self.add_output('frac', val=0.8, desc='Throttle fraction', shape=(nn,))
        
        # Residual
        self.add_residual('power_residual', val=np.zeros(nn), desc='Power residual (actual - desired)')
        
        # Declare partials
        self.declare_partials('*', '*', method='fd')
    
    def apply_nonlinear(self, inputs, outputs, residuals):
        nn = self.options['num_nodes']
        
        # Get current values
        altitude = inputs['fltcond|h']
        mach = inputs['fltcond|M']
        disa = inputs['fltcond|disa']
        desired_power = inputs['desired_power']
        frac = outputs['frac']
        
        # Use the existing turbo data to get fuel flow for current frac
        # Get training data
        alt_data = TurboData.dyn_alt_data__m
        mach_data = TurboData.dyn_mach_data
        disa_data = TurboData.dyn_disa_data__degC
        frac_data = TurboData.dyn_frac_data
        power_data_kW = TurboData.dyn_power_data__kW
        
        # Create training matrix
        X_train = np.column_stack([alt_data, mach_data, disa_data, frac_data])
        
        # Create query points
        X_query = np.column_stack([altitude, mach, disa, frac])
        
        # Interpolate fuel flow
        power_interpolator = RBFInterpolator(X_train, power_data_kW, kernel='thin_plate_spline')
        power_kW = power_interpolator(X_query)
        
        # Set residual: actual_power - desired_power = 0
        residuals['power_residual'] = power_kW - desired_power
    


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


def test_turbo_power_solver():
    """
    Test the TurboPowerSolver component
    """
    print("Testing TurboPowerSolver component...")
    
    # Set up the OpenMDAO model
    num_nodes = 5
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables
    ivc.add_output('fltcond|h', [5000, 10000, 15000, 20000, 20000], units='ft', desc='Altitude')
    ivc.add_output('fltcond|M', [0.1, 0.2, 0.3, 0.4, 0.4], desc='Mach number')
    ivc.add_output('fltcond|disa', [0, 5, 10, 15, 20], desc='Temperature deviation from ISA')
    ivc.add_output('desired_power', [700, 700, 700, 700, 700], units='kW', desc='Desired power')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('power_solver', TurboPowerSolver(num_nodes=num_nodes), promotes=['*'])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Run the model
    prob.run_model()
    
    # Get results
    frac_results = prob.get_val('frac')
    fuel_flow_results = prob.get_val('fuel_flow_kgph')
    altitude = prob.get_val('fltcond|h', units='m')
    mach = prob.get_val('fltcond|M')
    disa = prob.get_val('fltcond|disa')
    desired_power = prob.get_val('desired_power', units='W')
    
    print(f"\nTurboPowerSolver Results:")
    print(f"{'Node':<4} {'Altitude (m)':<12} {'Mach':<6} {'DISA (°C)':<10} {'Desired Power (W)':<16} {'Throttle':<8} {'Fuel Flow (kg/h)':<15}")
    print("-" * 85)
    
    for i in range(num_nodes):
        print(f"{i:<4} {altitude[i]:<12.0f} {mach[i]:<6.1f} {disa[i]:<10.0f} {desired_power[i]:<16.0f} {frac_results[i]:<8.3f} {fuel_flow_results[i]:<15.1f}")
    
    # Verify the solution by checking if the calculated power matches desired power
    specific_power = 3500.0  # W/(kg/h) - same as in the component
    calculated_power = fuel_flow_results * specific_power
    power_errors = np.abs(calculated_power - desired_power)
    relative_errors = power_errors / desired_power * 100
    
    print(f"\nPower Verification:")
    print(f"Mean power error: {np.mean(power_errors):.0f} W")
    print(f"Max power error: {np.max(power_errors):.0f} W")
    print(f"Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"Max relative error: {np.max(relative_errors):.2f}%")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot throttle vs desired power
    plt.subplot(2, 2, 1)
    plt.scatter(desired_power/1000, frac_results, alpha=0.7, s=50)
    plt.xlabel('Desired Power (kW)')
    plt.ylabel('Throttle Fraction')
    plt.title('Throttle vs Desired Power')
    plt.grid(True, alpha=0.3)
    
    # Plot fuel flow vs desired power
    plt.subplot(2, 2, 2)
    plt.scatter(desired_power/1000, fuel_flow_results, alpha=0.7, s=50)
    plt.xlabel('Desired Power (kW)')
    plt.ylabel('Fuel Flow (kg/h)')
    plt.title('Fuel Flow vs Desired Power')
    plt.grid(True, alpha=0.3)
    
    # Plot throttle vs altitude
    plt.subplot(2, 2, 3)
    plt.scatter(altitude/1000, frac_results, alpha=0.7, s=50)
    plt.xlabel('Altitude (km)')
    plt.ylabel('Throttle Fraction')
    plt.title('Throttle vs Altitude')
    plt.grid(True, alpha=0.3)
    
    # Plot power error distribution
    plt.subplot(2, 2, 4)
    plt.scatter(desired_power/1000, relative_errors, alpha=0.7, s=50)
    plt.xlabel('Desired Power (kW)')
    plt.ylabel('Power Error (%)')
    plt.title('Power Error vs Desired Power')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    return frac_results, fuel_flow_results, calculated_power, desired_power


def test_turbo_power_implicit_solver():
    """
    Test the TurboPowerImplicitSolver component
    """
    print("Testing TurboPowerImplicitSolver component...")
    
    # Set up the OpenMDAO model
    num_nodes = 5
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables
    ivc.add_output('fltcond|h', [5000, 10000, 15000, 20000, 20000], units='ft', desc='Altitude')
    ivc.add_output('fltcond|M', [0.1, 0.2, 0.3, 0.4, 0.4], desc='Mach number')
    ivc.add_output('fltcond|disa', [0, 5, 10, 15, 20], desc='Temperature deviation from ISA')
    ivc.add_output('desired_power', [700, 700, 700, 700, 700], units='kW', desc='Desired power')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('implicit_power_solver', TurboPowerImplicitSolver(num_nodes=num_nodes), promotes=['*'])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Set up nonlinear solver for the implicit component
    prob.model.implicit_power_solver.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    prob.model.implicit_power_solver.nonlinear_solver.options['iprint'] = 0
    prob.model.implicit_power_solver.nonlinear_solver.options['maxiter'] = 20
    prob.model.implicit_power_solver.nonlinear_solver.options['atol'] = 1e-6
    prob.model.implicit_power_solver.nonlinear_solver.options['rtol'] = 1e-6
    
    # Run the model
    prob.run_model()
    
    # Get results
    frac_results = prob.get_val('frac')
    fuel_flow_results = prob.get_val('fuel_flow_kgph')
    altitude = prob.get_val('fltcond|h', units='m')
    mach = prob.get_val('fltcond|M')
    disa = prob.get_val('fltcond|disa')
    desired_power = prob.get_val('desired_power', units='W')
    
    print(f"\nTurboPowerImplicitSolver Results:")
    print(f"{'Node':<4} {'Altitude (m)':<12} {'Mach':<6} {'DISA (°C)':<10} {'Desired Power (W)':<16} {'Throttle':<8} {'Fuel Flow (kg/h)':<15}")
    print("-" * 85)
    
    for i in range(num_nodes):
        print(f"{i:<4} {altitude[i]:<12.0f} {mach[i]:<6.1f} {disa[i]:<10.0f} {desired_power[i]:<16.0f} {frac_results[i]:<8.3f} {fuel_flow_results[i]:<15.1f}")
    
    # Verify the solution by checking if the calculated power matches desired power
    specific_power = 3500.0  # W/(kg/h) - same as in the component
    calculated_power = fuel_flow_results * specific_power
    power_errors = np.abs(calculated_power - desired_power)
    relative_errors = power_errors / desired_power * 100
    
    print(f"\nPower Verification:")
    print(f"Mean power error: {np.mean(power_errors):.0f} W")
    print(f"Max power error: {np.max(power_errors):.0f} W")
    print(f"Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"Max relative error: {np.max(relative_errors):.2f}%")
    
    # Compare with explicit solver results
    print(f"\nComparing with explicit solver...")
    
    # Set up explicit solver model for comparison
    model_explicit = om.Group()
    ivc_explicit = om.IndepVarComp()
    
    ivc_explicit.add_output('fltcond|h', [5000, 10000, 15000, 20000, 25000], units='m', desc='Altitude')
    ivc_explicit.add_output('fltcond|M', [0.3, 0.4, 0.5, 0.6, 0.7], desc='Mach number')
    ivc_explicit.add_output('fltcond|disa', [0, 5, 10, 15, 20], desc='Temperature deviation from ISA')
    ivc_explicit.add_output('desired_power', [800000, 900000, 1000000, 1100000, 1200000], units='W', desc='Desired power')
    
    model_explicit.add_subsystem('ivc', ivc_explicit, promotes=['*'])
    model_explicit.add_subsystem('explicit_power_solver', TurboPowerSolver(num_nodes=num_nodes), promotes=['*'])
    
    prob_explicit = om.Problem(model_explicit, reports=False)
    prob_explicit.setup()
    prob_explicit.run_model()
    
    frac_explicit = prob_explicit.get_val('frac')
    fuel_flow_explicit = prob_explicit.get_val('fuel_flow_kgph')
    
    # Compare results
    frac_diff = np.abs(frac_results - frac_explicit)
    fuel_flow_diff = np.abs(fuel_flow_results - fuel_flow_explicit)
    
    print(f"Throttle fraction differences:")
    print(f"  Mean difference: {np.mean(frac_diff):.6f}")
    print(f"  Max difference: {np.max(frac_diff):.6f}")
    
    print(f"Fuel flow differences:")
    print(f"  Mean difference: {np.mean(fuel_flow_diff):.2f} kg/h")
    print(f"  Max difference: {np.max(fuel_flow_diff):.2f} kg/h")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot throttle comparison
    plt.subplot(2, 2, 1)
    plt.scatter(desired_power/1000, frac_results, alpha=0.7, s=50, label='Implicit', color='blue')
    plt.scatter(desired_power/1000, frac_explicit, alpha=0.7, s=50, label='Explicit', color='red', marker='s')
    plt.xlabel('Desired Power (kW)')
    plt.ylabel('Throttle Fraction')
    plt.title('Throttle vs Desired Power - Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot fuel flow comparison
    plt.subplot(2, 2, 2)
    plt.scatter(desired_power/1000, fuel_flow_results, alpha=0.7, s=50, label='Implicit', color='blue')
    plt.scatter(desired_power/1000, fuel_flow_explicit, alpha=0.7, s=50, label='Explicit', color='red', marker='s')
    plt.xlabel('Desired Power (kW)')
    plt.ylabel('Fuel Flow (kg/h)')
    plt.title('Fuel Flow vs Desired Power - Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot differences
    plt.subplot(2, 2, 3)
    plt.scatter(desired_power/1000, frac_diff, alpha=0.7, s=50, color='green')
    plt.xlabel('Desired Power (kW)')
    plt.ylabel('Throttle Difference')
    plt.title('Throttle Fraction Differences')
    plt.grid(True, alpha=0.3)
    
    # Plot power errors
    plt.subplot(2, 2, 4)
    plt.scatter(desired_power/1000, relative_errors, alpha=0.7, s=50, color='orange')
    plt.xlabel('Desired Power (kW)')
    plt.ylabel('Power Error (%)')
    plt.title('Power Error vs Desired Power')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return frac_results, fuel_flow_results, calculated_power, desired_power


if __name__ == "__main__":
    
    # Number of nodes to evaluate
    # Test the power solver
    test_turbo_power_solver()
    
    # Test the power implicit solver
    test_turbo_power_implicit_solver()

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
    

    
    # Test the power implicit solver
    test_turbo_power_implicit_solver()

