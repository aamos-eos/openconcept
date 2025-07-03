import numpy as np
import openmdao.api as om
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.constants import g
import cProfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import json
import os, sys

# Import the global data store
from openconcept.propulsion.propeller_data import PropellerData



class EfficiencyCalc(om.ExplicitComponent):
    """
    Component that computes efficiency using the basic formula: eta = thrust * velocity / power
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of time nodes to evaluate')
    
    def setup(self):
        num_nodes = self.options['num_nodes']


        self.add_input('thrust_calc', val=10000.0, units='N', desc='Thrust', shape=(num_nodes, ))


        self.add_input('fltcond|Utrue', val=100.0, units='m/s', desc='Velocity', shape=(num_nodes))
        self.add_input('power', val=1000000.0, units='W', desc='Power', shape=(num_nodes, ))
        
        # Outputs
        self.add_output('eta', val=80, desc='Efficiency', shape=(num_nodes,))

        self.declare_partials('*', '*', method='exact')
    
    def compute(self, inputs, outputs):

        thrust = inputs['thrust_calc']
        velocity = inputs['fltcond|Utrue']
        power = inputs['power']
        
        # Basic efficiency formula: eta = thrust * velocity / power
        outputs['eta'] = thrust * velocity / power * 100  # Convert to percentage
    
    def compute_partials(self, inputs, partials):
        thrust = inputs['thrust_calc']
        velocity = inputs['fltcond|Utrue']
        power = inputs['power']
        
        num_nodes = self.options['num_nodes']
        
        partials['eta', 'thrust_calc'] = np.eye(num_nodes) * velocity / power * 100
        partials['eta', 'fltcond|Utrue'] = np.eye(num_nodes) * thrust / power * 100
        partials['eta', 'power'] = np.eye(num_nodes) * -thrust * velocity / (power ** 2) * 100



class PropellerRBFInterpolator(om.ExplicitComponent):
    """
    Component that uses scipy.optimize.minimize to find optimal rpm or power.
    Takes velocity, RPM, diameter, and power as inputs.
    """
    
    # Class-level interpolator (shared across all instances)
    _thrust_interpolator = None
    _interpolator_built = False
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('nd', default=10, desc='number of design points to evaluate')
        self.options.declare('use_dynamic_data', default=True, desc='use dynamic data if True, static if False')
        self.options.declare('power_set', default=False, desc='power is set (input)')
        self.options.declare('rpm_set', default=False, desc='rpm is set (input)')
        self.options.declare('thrust_set', default=True, desc='thrust is set (input)')
    
    @classmethod
    def _build_shared_interpolator(cls):
        """Build interpolator once for all instances"""
        if cls._interpolator_built:
            return
            
        print("Building shared propeller interpolator...")
        
        alt_data = PropellerData.dyn_dens_alt_data__m
        velocity_data = PropellerData.dyn_ktas_data__kts
        rpm_data = PropellerData.dyn_rpm_data__rpm
        power_data = PropellerData.dyn_power_data__W
        thrust_data = PropellerData.dyn_thrust_data__N
        X_train = np.column_stack([rpm_data, velocity_data, alt_data, power_data])
        cls._thrust_interpolator = RBFInterpolator(X_train, thrust_data, kernel='thin_plate_spline')
        
        cls._interpolator_built = True
        print("Shared propeller interpolator built successfully!")
    
    def setup(self):
        num_nodes = self.options['num_nodes']
        nd = self.options['nd']
        power_set = self.options['power_set']
        rpm_set = self.options['rpm_set']
        thrust_set = self.options['thrust_set']
        
        # Inputs
        self.add_input('fltcond|Utrue', val=100.0, units='m/s', desc='Airspeed matrix', shape=(num_nodes))
        self.add_input('diameter', val=4.4, units='m', desc='Diameter matrix', shape=(1))
        self.add_input('fltcond|h', val=10000.0, units='m', desc='Altitude matrix', shape=(num_nodes))
        
        if thrust_set:
            self.add_input('thrust', val=10000.0, units='N', desc='Required thrust', shape=(num_nodes))
        
            # Known variable (input)
            if power_set:
                self.add_input('power', val=1000000.0, units='W', desc='Power matrix', shape=(num_nodes))
            if rpm_set:
                self.add_input('rpm', val=1000.0, units='rpm', desc='RPM matrix', shape=(num_nodes))
            
            # Unknown variable (output - to be optimized)
            if power_set:
                self.add_output('rpm', val=1000.0, units='rpm', desc='Optimal RPM matrix', shape=(num_nodes))
            if rpm_set:
                self.add_output('power', val=1000000.0, units='W', desc='Optimal Power matrix', shape=(num_nodes))

        else:
            self.add_input('power', val=1000000.0, units='W', desc='Power matrix', shape=(num_nodes))
            self.add_input('rpm', val=1000.0, units='rpm', desc='RPM matrix', shape=(num_nodes))
        
        # Additional outputs
        self.add_output('J', val=0.5, desc='Advance ratio matrix', shape=(num_nodes))
        self.add_output('thrust_calc', val=10000.0, units='N', desc='Calculated thrust', shape=(num_nodes))
        
        # Ensure shared interpolator is built (only happens once)
        self._build_shared_interpolator()
        
        # Use finite difference for all partials since optimization is involved
        self.declare_partials('*', '*', method='fd')
    
    def _objective_function(self, x, thrust_required, velocity, altitude, power_set, power_value, rpm_value):
        """Objective function for optimization: minimize |thrust_required - thrust_calculated|"""
        if power_set:
            rpm = x
            power = power_value
        else:
            rpm = rpm_value
            power = x
        
        # Convert velocity from m/s to knots for interpolator
        velocity_kts = velocity / 0.514444
        
        # Interpolate thrust for all points at once
        X_query = np.column_stack([rpm, velocity_kts, altitude, power])
        thrust_calc = self._thrust_interpolator(X_query)
        
        return np.sum(np.abs(thrust_required - thrust_calc))
    
    def compute(self, inputs, outputs):
        velocity = inputs['fltcond|Utrue']  # This is in m/s
        diameter = inputs['diameter']
        altitude = inputs['fltcond|h']
        thrust_set = self.options['thrust_set']

        if thrust_set:
            thrust_required = inputs['thrust']

        power_set = self.options['power_set']
        rpm_set = self.options['rpm_set']
        
        num_nodes = self.options['num_nodes']

        if thrust_set:
            
            # Get known values
            if power_set:
                power_value = inputs['power']
                rpm_value = 1000.0 * np.ones(num_nodes)  # Initial guess for all points
            else:
                rpm_value = inputs['rpm']
                power_value = 1000000.0 * np.ones(num_nodes)  # Initial guess for all points
            
            # Set bounds for optimization
            if power_set:
                # Optimize rpm, bounds from 500 to 2000 rpm
                x0 = rpm_value
                bounds = [(900, 1200)] * num_nodes
            else:
                # Optimize power, bounds from 100kW to 3000kW
                x0 = power_value
                bounds = [(150e3, 2050e3)] * num_nodes
            
            # Optimize for all time points at once
            result = minimize(
                self._objective_function,
                x0,
                args=(thrust_required, velocity, altitude, power_set, power_value, rpm_value),
                bounds=bounds,
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            # Store results
            if power_set:
                outputs['rpm'] = result.x
                optimal_rpm = result.x
                optimal_power = power_value
            else:
                outputs['power'] = result.x
                optimal_rpm = rpm_value
                optimal_power = result.x
            
            # Calculate advance ratio for all points
            n = optimal_rpm / 60.0
            outputs['J'] = velocity / (n * diameter)
            
            # Convert velocity from m/s to knots for interpolator
            velocity_kts = velocity / 0.514444
            X_query = np.column_stack([optimal_rpm, velocity_kts, altitude, optimal_power])

            # Calculate final thrust for verification
            outputs['thrust_calc'] = self._thrust_interpolator(X_query)
        else:
            # Convert velocity from m/s to knots for interpolator
            velocity_kts = velocity / 0.514444
            X_query = np.column_stack([inputs['rpm'], velocity_kts, altitude, inputs['power']])
            outputs['thrust_calc'] = self._thrust_interpolator(X_query)
            outputs['J'] = velocity / (inputs['rpm'] / 60.0 * diameter)






class EmpiricalPropeller(om.Group):
    """
    Comprehensive group that solves the propeller map with all components.
    Handles different combinations of set/unknown variables and includes all necessary components.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of time nodes to evaluate')
        self.options.declare('nd', default=10, desc='number of design points to evaluate')
        self.options.declare('thrust_set', default=True, desc='thrust is set (input)')
        self.options.declare('power_set', default=False, desc='power is set (input)')
        self.options.declare('rpm_set', default=False, desc='rpm is set (input)')
        self.options.declare('use_dynamic_data', default=True, desc='use dynamic data if True, static if False')
    
        # Load data immediately when module is imported - this ensures it's available for all components
        PropellerData.load_data(prop_filename='openconcept/propulsion/empirical_data/DOWTY_prop_5blade_14p3ft.xlsx', sheet_name='data')


    def setup(self):
        num_nodes = self.options['num_nodes']
        thrust_set = self.options['thrust_set']
        power_set = self.options['power_set']
        rpm_set = self.options['rpm_set']
        use_dynamic_data = self.options['use_dynamic_data']
        
        
        # Add the RBF interpolator component (now implicit)
        self.add_subsystem('prop_interp', PropellerRBFInterpolator(num_nodes=num_nodes,  
                                                                   use_dynamic_data=use_dynamic_data,
                                                                   power_set=power_set,
                                                                   thrust_set=thrust_set,
                                                                   rpm_set=rpm_set), promotes=['*'])
        
        self.add_subsystem('efficiency_calc', EfficiencyCalc(num_nodes=num_nodes), promotes=['*'])
        

        

        
        # Set up the nonlinear solver
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 100
        self.nonlinear_solver.options['atol'] = 1e-6
        self.nonlinear_solver.options['rtol'] = 1e-6
        
        # Set up the linear solver
        self.linear_solver = om.DirectSolver()



def test_propeller_interpolation_accuracy():
    """
    Test propeller interpolation accuracy by comparing interpolated vs actual values
    Uses 100 random datapoints from the propeller map
    """
    print("Testing propeller interpolation accuracy...")
    
    # Get the data
    prop_data = PropellerData.get_data(prop_filename='openconcept/propulsion/empirical_data/DOWTY_prop_5blade_14p3ft.xlsx', sheet_name='data')
    
    # Use dynamic data for testing (non-zero airspeed)
    alt_data = prop_data.dyn_dens_alt_data__m
    velocity_data = prop_data.dyn_ktas_data__kts
    rpm_data = prop_data.dyn_rpm_data__rpm
    power_data = prop_data.dyn_power_data__W
    thrust_data = prop_data.dyn_thrust_data__N
    
    n_total_points = len(thrust_data)
    n_test_points = min(3000, n_total_points)  # Test 100 points or all if less than 100
    
    # Generate random indices
    np.random.seed(42)  # For reproducible results
    test_indices = np.random.choice(n_total_points, n_test_points, replace=False)
    
    print(f"Testing {n_test_points} randomly sampled points from {n_total_points} total propeller data points...")
    print(f"Altitude range: {alt_data.min():.1f} to {alt_data.max():.1f} m")
    print(f"Velocity range: {velocity_data.min():.1f} to {velocity_data.max():.1f} kts")
    print(f"RPM range: {rpm_data.min():.1f} to {rpm_data.max():.1f} rpm")
    print(f"Power range: {power_data.min()/1000:.1f} to {power_data.max()/1000:.1f} kW")
    print(f"Thrust range: {thrust_data.min():.1f} to {thrust_data.max():.1f} N")
    
    # Set up the OpenMDAO model to test the actual component
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Convert velocity from knots to m/s for OpenMDAO component
    velocity_mps = velocity_data[test_indices] * 0.514444
    
    # Add independent variables
    ivc.add_output('fltcond|Utrue', velocity_mps, units='m/s', desc='Airspeed')
    ivc.add_output('diameter', 4.4, units='m', desc='Propeller diameter')
    ivc.add_output('fltcond|h', alt_data[test_indices], units='m', desc='Altitude')
    ivc.add_output('power', power_data[test_indices], units='W', desc='Power')
    ivc.add_output('rpm', rpm_data[test_indices], units='rpm', desc='RPM')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])

    
    # Add the actual propeller component (power_set and rpm_set, calculate thrust)
    model.add_subsystem('prop_calc', 
                       EmpiricalPropeller(num_nodes=n_test_points, 
                                               thrust_set=False, 
                                               power_set=True,
                                               rpm_set=True,
                                               use_dynamic_data=True), 
                       promotes=['*'])

    prob = om.Problem(model, reports=False)
    prob.setup()
    om.n2(prob)
    
    # Run the model
    prob.run_model()
    
    # Get interpolated values from the OpenMDAO component
    y_interpolated = prob.get_val('thrust_calc')
    y_actual = thrust_data[test_indices]
    
    # Calculate statistics
    errors = np.abs(y_interpolated - y_actual)
    relative_errors = errors / y_actual * 100
    
    # Handle any infinite or NaN values
    relative_errors = np.where(np.isinf(relative_errors), 100.0, relative_errors)
    relative_errors = np.where(np.isnan(relative_errors), 100.0, relative_errors)
    
    print(f"\n=== Propeller Interpolation Accuracy Results ===")
    print(f"Mean absolute error: {np.mean(errors):.2f} N")
    print(f"Max absolute error: {np.max(errors):.2f} N")
    print(f"Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"Max relative error: {np.max(relative_errors):.2f}%")
    print(f"Standard deviation of relative error: {np.std(relative_errors):.2f}%")
    print(f"95th percentile relative error: {np.percentile(relative_errors, 95):.2f}%")
    
    # Create comprehensive visualization
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Actual vs Interpolated
    plt.subplot(3, 3, 1)
    plt.scatter(y_actual, y_interpolated, alpha=0.6, s=20)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Thrust (N)')
    plt.ylabel('Interpolated Thrust (N)')
    plt.title('Propeller: Actual vs Interpolated Thrust')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    plt.subplot(3, 3, 2)
    plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error (N)')
    plt.ylabel('Frequency')
    plt.title('Absolute Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Relative error distribution
    plt.subplot(3, 3, 3)
    plt.hist(relative_errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    plt.title('Relative Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Error vs actual value
    plt.subplot(3, 3, 4)
    plt.scatter(y_actual, relative_errors, alpha=0.6, s=20)
    plt.xlabel('Actual Thrust (N)')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs Actual Thrust')
    plt.grid(True, alpha=0.3)
    #plt.ylim(0, 25)
    
    # Plot 5: Error vs altitude
    plt.subplot(3, 3, 5)
    plt.scatter(alt_data[test_indices], relative_errors, alpha=0.6, s=20)
    plt.xlabel('Altitude (m)')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs Altitude')
    plt.grid(True, alpha=0.3)
    #plt.ylim(0, 25)
    
    # Plot 6: Error vs velocity
    plt.subplot(3, 3, 6)
    plt.scatter(velocity_data[test_indices], relative_errors, alpha=0.6, s=20)
    plt.xlabel('Velocity (kts)')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs Velocity')
    #plt.ylim(0, 25)
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Error vs RPM
    plt.subplot(3, 3, 7)
    plt.scatter(rpm_data[test_indices], relative_errors, alpha=0.6, s=20)
    plt.xlabel('RPM')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs RPM')
    plt.grid(True, alpha=0.3)
    #plt.ylim(0, 25)
    
    # Plot 8: Error vs power
    plt.subplot(3, 3, 8)
    plt.scatter(power_data[test_indices]/1000, relative_errors, alpha=0.6, s=20)
    plt.xlabel('Power (kW)')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs Power')
    plt.grid(True, alpha=0.3)
    #plt.ylim(0, 25)
    
    # Plot 9: Error vs advance ratio (J)
    # Convert knots to m/s for J calculation: 1 kt = 0.514444 m/s
    velocity_mps_for_j = velocity_data[test_indices] * 0.514444
    J_data = velocity_mps_for_j / (rpm_data[test_indices] / 60.0 * 4.4)  # Assuming 4.4m diameter
    plt.subplot(3, 3, 9)
    plt.scatter(J_data, relative_errors, alpha=0.6, s=20)
    plt.xlabel('Advance Ratio (J)')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs Advance Ratio')
    plt.grid(True, alpha=0.3)
    #plt.ylim(0, 25)
    plt.tight_layout()
    plt.show()
    
    # Create additional analysis plots
    plt.figure(figsize=(12, 8))
    
    # Plot error statistics by ranges
    plt.subplot(2, 2, 1)
    error_ranges = [0, 1, 2, 5, 10, 20, 50, 100]
    error_counts = []
    for i in range(len(error_ranges)-1):
        count = np.sum((relative_errors >= error_ranges[i]) & (relative_errors < error_ranges[i+1]))
        error_counts.append(count)
    
    plt.bar(range(len(error_counts)), error_counts, alpha=0.7)
    plt.xlabel('Error Range (%)')
    plt.ylabel('Number of Points')
    plt.title('Error Distribution by Ranges')
    plt.xticks(range(len(error_counts)), [f'{error_ranges[i]}-{error_ranges[i+1]}' for i in range(len(error_counts))])
    plt.grid(True, alpha=0.3)
    
    # Plot cumulative error distribution
    plt.subplot(2, 2, 2)
    sorted_errors = np.sort(relative_errors)
    cumulative_percent = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    plt.plot(sorted_errors, cumulative_percent, 'b-', linewidth=2)
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot error vs data density (distance to nearest neighbor)
    plt.subplot(2, 2, 3)
    from scipy.spatial.distance import cdist
    # Use the same order as the interpolator: [rpm, velocity_kts, altitude, power]
    X_test = np.column_stack([rpm_data[test_indices], velocity_data[test_indices], alt_data[test_indices], power_data[test_indices]])
    X_train = np.column_stack([rpm_data, velocity_data, alt_data, power_data])
    distances = cdist(X_test, X_train)
    min_distances = np.min(distances, axis=1)
    plt.scatter(min_distances, relative_errors, alpha=0.6, s=20)
    plt.xlabel('Distance to Nearest Training Point')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs Data Density')
    plt.grid(True, alpha=0.3)
    
    # Plot error vs thrust magnitude
    plt.subplot(2, 2, 4)
    plt.scatter(np.abs(y_actual), relative_errors, alpha=0.6, s=20)
    plt.xlabel('|Actual Thrust| (N)')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs Thrust Magnitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\n=== Detailed Error Analysis ===")
    print(f"Points with error < 1%: {np.sum(relative_errors < 1):d} ({np.sum(relative_errors < 1)/len(relative_errors)*100:.1f}%)")
    print(f"Points with error < 2%: {np.sum(relative_errors < 2):d} ({np.sum(relative_errors < 2)/len(relative_errors)*100:.1f}%)")
    print(f"Points with error < 5%: {np.sum(relative_errors < 5):d} ({np.sum(relative_errors < 5)/len(relative_errors)*100:.1f}%)")
    print(f"Points with error < 10%: {np.sum(relative_errors < 10):d} ({np.sum(relative_errors < 10)/len(relative_errors)*100:.1f}%)")
    print(f"Points with error > 20%: {np.sum(relative_errors > 20):d} ({np.sum(relative_errors > 20)/len(relative_errors)*100:.1f}%)")
    
    # Find worst cases
    worst_indices = np.argsort(relative_errors)[-5:]
    print(f"\n=== Worst 5 Cases ===")
    for i, idx in enumerate(worst_indices):
        test_idx = test_indices[idx]
        print(f"Case {i+1}: Alt={alt_data[test_idx]:.0f}m, V={velocity_data[test_idx]:.1f}kts, "
              f"RPM={rpm_data[test_idx]:.0f}, P={power_data[test_idx]/1000:.1f}kW, "
              f"T_actual={y_actual[idx]:.1f}N, T_interp={y_interpolated[idx]:.1f}N, "
              f"Error={relative_errors[idx]:.1f}%")
    
    return y_interpolated, y_actual, errors, relative_errors


def test_propeller_openmdao_component():
    """
    Test the OpenMDAO propeller component specifically
    """
    print("\n=== Testing OpenMDAO Propeller Component ===")
    
    # Get test data
    prop_data = PropellerData.get_data()
    n_test_points = 50  # Use fewer points for OpenMDAO testing
    
    # Generate random indices
    np.random.seed(42)
    test_indices = np.random.choice(len(prop_data.dyn_thrust_data__N), n_test_points, replace=False)
    
    # Set up the OpenMDAO model
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables
    # Convert velocity from knots to m/s for OpenMDAO component
    velocity_mps = prop_data.dyn_ktas_data__kts[test_indices] * 0.514444
    ivc.add_output('fltcond|Utrue', velocity_mps, units='m/s', desc='Airspeed')
    ivc.add_output('diameter', 4.4, units='m', desc='Propeller diameter')
    ivc.add_output('fltcond|h', prop_data.dyn_dens_alt_data__m[test_indices], units='m', desc='Altitude')
    ivc.add_output('power', prop_data.dyn_power_data__W[test_indices], units='W', desc='Power')
    ivc.add_output('rpm', prop_data.dyn_rpm_data__rpm[test_indices], units='rpm', desc='RPM')
    ivc.add_output('delta_isa', 0.0, units='K', desc='Temperature deviation from ISA')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    
    # Add propeller component (power_set and rpm_set, calculate thrust)
    model.add_subsystem('propeller_map', 
                       EmpiricalPropeller(num_nodes=n_test_points, 
                                        thrust_set=False, 
                                        power_set=True,
                                        rpm_set=True,
                                        use_dynamic_data=True), 
                       promotes=['*'])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Run the model
    prob.run_model()
    
    # Get results
    thrust_calc = prob.get_val('thrust_calc')
    thrust_actual = prop_data.dyn_thrust_data__N[test_indices]
    
    # Calculate errors
    errors = np.abs(thrust_calc - thrust_actual)
    relative_errors = errors / thrust_actual * 100
    
    print(f"OpenMDAO Component Results:")
    print(f"Mean absolute error: {np.mean(errors):.2f} N")
    print(f"Max absolute error: {np.max(errors):.2f} N")
    print(f"Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"Max relative error: {np.max(relative_errors):.2f}%")
    
    return thrust_calc, thrust_actual, errors, relative_errors


if __name__ == "__main__":
    # Test the complete propeller map group

    # Test interpolation accuracy
    test_propeller_interpolation_accuracy()
    
    # Test OpenMDAO component
    test_propeller_openmdao_component()
    
    # Check partials
    #prob.check_partials(compact_print=True)








