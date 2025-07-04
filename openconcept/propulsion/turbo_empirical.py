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
from openconcept.utilities import ElementMultiplyDivideComp, Integrator

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


class TurboMission(om.Group):
    """
    Group containing turbo fuel flow component and mission integrator.
    
    This group combines the turbo engine model with mission integration
    to calculate total fuel consumption over a time period.
    
    Parameters
    ----------
    num_nodes : int
        Number of analysis points (default: 100)
    duration : tuple
        Time interval for integration (start, end) in seconds (default: (0, 3600))
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=100, desc='Number of analysis points')
        self.options.declare('duration', default=3600, desc='Time interval (start, end) in seconds')
        self.options.declare('throttle_set', default=True, desc='Set throttle or power for gas turbine')
    
    def setup(self):
        nn = self.options['num_nodes']
        duration = self.options['duration']
        throttle_set = self.options['throttle_set']
        
        # Add the turbo component
        self.add_subsystem('turbo', TurboFuelFlowRBF(num_nodes=nn, throttle_set=throttle_set), promotes=["*"])
        #self.add_subsystem('turbo', TurboFuelFlowFromPowerMetaModel(num_nodes=nn, throttle_set=throttle_set), promotes=["*"])
        
        # Add mission integrator
        integrator = self.add_subsystem('mission_integrator', 
                                        Integrator(num_nodes=nn, 
                                                   time_setup="duration",
                                                   diff_units="h",
                                                   method="simpson"),
                                        promotes=["*"])
        
        # Set up integrator inputs/outputs
        integrator.add_integrand('fuel_consumption', units='kg', rate_name='fuel_flow_kgph')

        self.add_subsystem('obj_func', obj_func(num_nodes=nn), promotes=["*"])
        
        
        # Connect fuel flow to integrator
        #self.connect('fuel_flow_kgph', 'mission_integrator.fuel_flow_kgph')
    
class obj_func(om.ExplicitComponent):
    """
    Objective function for fuel consumption optimization
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('fuel_consumption', val=0.0, units='kg', desc='Fuel consumption', shape=(nn,))
        self.add_output('obj_func_val', val=0.0, desc='Objective function')
        
        self.declare_partials('obj_func_val', 'fuel_consumption', method='fd')
        
    def compute(self, inputs, outputs):
        outputs['obj_func_val'] = inputs['fuel_consumption'][-1]
        
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        partials['obj_func_val', 'fuel_consumption'] = np.zeros((nn))
        partials['obj_func_val', 'fuel_consumption'][-1] = 1.0


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
            fuel_flow_metamodel.add_output('power', 1000.0, training_data=TurboData.dyn_power_data__kW, units="kW", shape=(num_nodes,))
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
    
    

def optimize_fuel_consumption():
    """
    Set up an OpenMDAO optimizer to find optimal altitude, throttle, and Mach for minimum fuel consumption
    """
    print("Setting up fuel consumption optimization...")
    
    # Set up the OpenMDAO model
    model = om.Group()

    nn = 11
    duration =  2  # 1 hour mission
    
    # Design variables (to be optimized)
    ivc = om.IndepVarComp()
    ivc.add_output('fltcond|h', val=24000 * np.ones(nn), units='ft', desc='Altitude in ft')
    ivc.add_output('fltcond|M', val=0.2 * np.ones(nn), desc='Mach number')
    ivc.add_output('throttle', val=0.7 * np.ones(nn), desc='Throttle fraction')
    ivc.add_output('fltcond|disa', val=0.0 * np.ones(nn), desc='DISA in degrees Celsius')
    ivc.add_output('duration', val=duration, units='h', desc='Mission duration')
    ivc.add_output('fuel_consumption_initial', val=0, units='kg', desc='Fuel consumption')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    
    # Add the turbo mission group (includes integrator)
    model.add_subsystem('turbo_mission', TurboMission(
        num_nodes=nn, 
        duration=duration,
        throttle_set=True
    ), promotes=['*'])

    


    # Set up the problem
    prob = om.Problem(model, reports=False)
    
    # Set up the driver (optimizer)
    prob.driver = om.ScipyOptimizeDriver()
    #prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'Powell'
    prob.driver.options['tol'] = 1e-6 
    #prob.driver.options['optimizer'] = 'IPOPT'
    #prob.driver.opt_settings['MAXIT'] = 1000

    #prob.driver.options['tol'] = 1e-6
    #prob.driver.options['maxiter'] = 1000
    
    # Design variables: find altitude wiht lowest fuel consumption
    prob.model.add_design_var('fltcond|h', lower=2500, upper=26000, units='ft')
    
    # Objective (what to minimize) - now total fuel consumption over mission
    prob.model.add_objective('obj_func_val', scaler=1/700.0)
    
    # Constraints
    prob.model.add_constraint('power', lower=300, upper=800, units='kW')  # Power requirement


    # Set up the problem
    prob.setup()
    #om.n2(prob)
    
    print("Initial conditions:")
    print("  Altitude: ft")
    print(prob['fltcond|h'])
    print("  Mach:")
    print(prob['fltcond|M'])
    print("  Throttle:")
    print(prob['throttle'])
    print("  Initial fuel flow: kg/h")
    print(prob['fuel_flow_kgph'])
    print("  Initial power: kW")
    print(prob['power'])
    print("  Mission duration: seconds")
    print(duration)
    
    # Run the optimization
    print("\nRunning optimization...")
    prob.run_driver()
    
    # Results
    print("\nOptimization results:")
    print("  Optimal altitude: ft")
    print(prob['fltcond|h'])
    print("  Optimal Mach:")
    print(prob['fltcond|M'])
    print("  Optimal throttle:")
    print(prob['throttle'])
    print("  Final fuel flow: kg/h")
    print(prob['fuel_flow_kgph'])
    print("  Power output: kW")
    print(prob['power'])
    print("  Total fuel consumption: kg")
    print(prob.get_val('obj_func_val'))
    
    # Check if optimization was successful
    if prob.driver.fail:
        print("Warning: Optimization may not have converged!")
    else:
        print("Optimization completed successfully!")
    
    return {
        'altitude': prob['fltcond|h'][0],
        'mach': prob['fltcond|M'][0],
        'throttle': prob['throttle'][0],
        'fuel_flow': prob['fuel_flow_kgph'][0],
        'power': prob['power'][0],
        'total_fuel': prob.get_val('obj_func_val'),
    }


def test_turbomission():
    """
    Test function to verify TurboMission integrator is working correctly
    """
    print("="*60)
    print("TESTING TURBO MISSION INTEGRATOR")
    print("="*60)
    
    # Set up a simple mission
    nn = 11  # 10 time points
    duration = 1.0  # 1 hour mission
    
    # Create the model
    model = om.Group()

    altitude = np.linspace(10000, 15000, nn)
    mach = np.linspace(0.2, 0.4, nn)
    throttle = np.linspace(0.6, 0.9, nn)
    disa = np.linspace(0.0, 0.0, nn)
    
    # Add independent variables
    ivc = om.IndepVarComp()
    ivc.add_output('fltcond|h', val=altitude, units='ft', desc='Altitude')
    ivc.add_output('fltcond|M', val=mach, desc='Mach number')
    ivc.add_output('throttle', val=throttle, desc='Throttle fraction')
    ivc.add_output('fltcond|disa', val=disa, desc='DISA')
    ivc.add_output('duration', val=duration, units='h', desc='Mission duration')
    ivc.add_output('fuel_consumption_initial', val=0, units='kg', desc='Initial fuel consumption')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    
    # Add the turbo mission group
    model.add_subsystem('turbo_mission', TurboMission(
        num_nodes=nn,
        duration=duration,
        throttle_set=True
    ), promotes=['*'])
    
    # Set up the problem
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Run the model
    print("Running TurboMission test...")
    prob.run_model()
    
    # Get results
    fuel_flow = prob.get_val('fuel_flow_kgph')
    power = prob.get_val('power')
    fuel_consumption = prob.get_val('fuel_consumption')
    total_fuel = prob.get_val('obj_func_val')
    
    # Formatting Results
    """
    print(f"\nResults:")
    print(f"  Mission duration: {duration} hours")
    print(f"  Number of time points: {nn}")
    print(f"  Altitude: {prob['fltcond|h']:.1f} ft")
    print(f"  Mach: {prob['fltcond|M']:.3f}")
    print(f"  Throttle: {prob['throttle']:.3f}")
    print(f"  Average fuel flow: {np.mean(fuel_flow):.2f} kg/h")
    print(f"  Average power: {np.mean(power):.1f} kW")
    print(f"  Total fuel consumption: {fuel_consumption:.2f} kg")
    
    # Verify integration makes sense
    expected_fuel = np.mean(fuel_flow) * duration
    print(f"  Expected fuel consumption: {expected_fuel:.2f} kg")
    print(f"  Integration accuracy: {abs(fuel_consumption[0] - expected_fuel):.4f} kg")
    
    # Test with varying conditions
    print("\n" + "="*60)
    print("TESTING WITH VARYING CONDITIONS")
    print("="*60)


    
    # Plot the results
    time_points = np.linspace(0, duration, nn)
    
    plt.figure(figsize=(12, 8))
    
    # Plot altitude
    plt.subplot(2, 3, 1)
    plt.plot(time_points, altitude, 'b-', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Altitude (ft)')
    plt.title('Altitude Profile')
    plt.grid(True, alpha=0.3)
    
    # Plot Mach
    plt.subplot(2, 3, 2)
    plt.plot(time_points, mach, 'r-', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Mach Number')
    plt.title('Mach Profile')
    plt.grid(True, alpha=0.3)
    
    # Plot throttle
    plt.subplot(2, 3, 3)
    plt.plot(time_points, throttle, 'g-', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Throttle')
    plt.title('Throttle Profile')
    plt.grid(True, alpha=0.3)
    
    # Plot fuel flow
    plt.subplot(2, 3, 4)
    plt.plot(time_points, fuel_flow, 'orange', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Fuel Flow (kg/h)')
    plt.title('Fuel Flow Profile')
    plt.grid(True, alpha=0.3)
    
    # Plot power
    plt.subplot(2, 3, 5)
    plt.plot(time_points, power, 'purple', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title('Power Profile')
    plt.grid(True, alpha=0.3)
    
    # Plot cumulative fuel consumption
    plt.subplot(2, 3, 6)
    # Calculate cumulative fuel consumption manually for verification
    dt = duration / (nn - 1)  # Time step
    cumulative_fuel = np.cumsum(fuel_flow) * dt
    plt.plot(time_points, cumulative_fuel, 'brown', linewidth=2, label='Cumulative')
    plt.axhline(y=fuel_consumption[0], color='red', linestyle='--', label=f'Integrator: {fuel_consumption[0]:.2f} kg')
    plt.xlabel('Time (hours)')
    plt.ylabel('Cumulative Fuel (kg)')
    plt.title('Fuel Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("TURBO MISSION TEST COMPLETED")
    print("="*60)
    print("If the integration results make sense, the TurboMission integrator is working correctly!")
    """




if __name__ == "__main__":
    #test_rbf_component()
    # Test interpolation accuracy
    #test_interpolation_accuracy()
    
    # Test TurboMission integrator
    #test_turbomission()
    
    # Run fuel consumption optimization
    print("="*60)
    print("FUEL CONSUMPTION OPTIMIZATION")
    print("="*60)
    
    # Mission optimization
    print("\n1. Mission optimization (minimum total fuel consumption):")
    print("-" * 50)
    result1 = optimize_fuel_consumption()
    
    # Results summary
    print("\n" + "="*60)
    print("MISSION OPTIMIZATION RESULTS")
    print("="*60)
    print(f"{'Metric':<25} {'Value':<15} {'Units':<10}")
    print("-" * 60)
    print(f"{'Optimal Altitude':<25} {result1['altitude']:<15.1f} {'ft':<10}")
    print(f"{'Optimal Mach':<25} {result1['mach']:<15.3f} {'':<10}")
    print(f"{'Optimal Throttle':<25} {result1['throttle']:<15.3f} {'':<10}")
    print(f"{'Final Fuel Flow':<25} {result1['fuel_flow']:<15.2f} {'kg/h':<10}")
    print(f"{'Power Output':<25} {result1['power']:<15.1f} {'kW':<10}")
    print(f"{'Total Fuel Consumption':<25} {result1['total_fuel'][0]:<15.2f} {'kg':<10}")

