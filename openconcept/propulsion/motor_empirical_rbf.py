import numpy as np
import openmdao.api as om
from scipy.interpolate import RBFInterpolator

import numpy as np
import cProfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

import os, sys

# Import the global data store
from openconcept.propulsion.h3x_motor_data_web import MotorDataEffMap # Torque to efficiency lookup table
from openconcept.propulsion.h3x_motor_data_web import MotorDataPowerVoltCurve # Power to voltage lookup table
from openconcept.propulsion.h3x_motor_data_rfi import MotorDataPowerEffCurve # Power to efficiency lookup table



class MotorEfficiencyRBFInterpolator(om.ExplicitComponent):
    """
    Explicit component for 2D interpolation of RPM + Torque to Efficiency using RBF
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        num_nodes = self.options['num_nodes']
        
        self.add_input('rpm', units='rpm', shape=(num_nodes,), desc='Motor speed')
        self.add_input('torque', units='N*m', shape=(num_nodes,), desc='Motor torque')
        self.add_output('eff', units=None, shape=(num_nodes,), desc='Motor efficiency')
        
        # Create the RBF interpolator using the global data
        rpm_data = MotorDataEffMap.rpm_data
        torque_data = MotorDataEffMap.torque_data
        eff_data = MotorDataEffMap.eff_data
        
        # Create training points for RBF
        training_points = np.column_stack([rpm_data, torque_data])
        self.rbf_interpolator = RBFInterpolator(training_points, eff_data, kernel='thin_plate_spline')
        
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        rpm = inputs['rpm']
        torque = inputs['torque']
        
        # Create test points for interpolation
        test_points = np.column_stack([rpm, torque])
        
        # Interpolate efficiency
        eff = self.rbf_interpolator(test_points)
        
        outputs['eff'] = eff


class MotorVoltagePowerRBFInterpolator(om.ExplicitComponent):
    """
    Explicit component for 2D interpolation of RPM + Voltage to Power Limit using RBF
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        num_nodes = self.options['num_nodes']
        
        self.add_input('rpm', units='rpm', shape=(num_nodes,), desc='Motor speed')
        self.add_input('voltage', units='V', shape=(num_nodes,), desc='Motor voltage')
        self.add_output('mech_power_lim', units='kW', shape=(num_nodes,), desc='Mechanical power limit')
        
        # Create the RBF interpolator using the global data
        rpm_data = MotorDataPowerVoltCurve.rpm_data
        voltage_data = MotorDataPowerVoltCurve.voltage_data
        power_data = MotorDataPowerVoltCurve.power_data
        
        # Create training points for RBF
        training_points = np.column_stack([rpm_data, voltage_data])
        self.rbf_interpolator = RBFInterpolator(training_points, power_data, kernel='thin_plate_spline')
        
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        rpm = inputs['rpm']
        voltage = inputs['voltage']
        
        # Create test points for interpolation
        test_points = np.column_stack([rpm, voltage])
        
        # Interpolate power limit
        mech_power_lim = self.rbf_interpolator(test_points)
        
        outputs['mech_power_lim'] = mech_power_lim


class MotorPowerEfficiencyRBFInterpolator(om.ExplicitComponent):
    """
    Explicit component for 2D interpolation of Voltage + Power to Efficiency using RBF
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        num_nodes = self.options['num_nodes']
        
        self.add_input('voltage', units='V', shape=(num_nodes,), desc='Motor voltage')
        self.add_input('mech_power', units='W', shape=(num_nodes,), desc='Motor power')
        self.add_output('eff', units=None, shape=(num_nodes,), desc='Motor efficiency')
        
        # Create the RBF interpolator using the global data
        # Note: This assumes you have voltage, power, and efficiency data available
        # You may need to adjust the data source based on your available datasets
        voltage_data = MotorDataPowerEffCurve.voltage_data
        power_data = MotorDataPowerEffCurve.power_data__W
        eff_data = MotorDataPowerEffCurve.eff_data
        
        # Create training points for RBF
        training_points = np.column_stack([voltage_data, power_data])
        self.rbf_interpolator = RBFInterpolator(training_points, eff_data, kernel='thin_plate_spline')
        
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        voltage = inputs['voltage']
        power = inputs['mech_power']
        
        # Create test points for interpolation
        test_points = np.column_stack([voltage, power])
        
        # Interpolate efficiency
        eff = self.rbf_interpolator(test_points)
        
        outputs['eff'] = eff


def test_interpolation_accuracy():
    """
    Test interpolation accuracy by comparing interpolated vs actual values
    """
    print("Testing motor interpolation accuracy...")
    
    # Test 1: Efficiency Interpolation
    print("\n=== Testing Efficiency Interpolation ===")
    
    # Get the efficiency data
    motor_eff_data = MotorDataEffMap.get_data(motor_filename='openconcept/propulsion/empirical_data/H3X_HPDM_2300_eff.xlsx')
    
    # Use actual data points for testing
    n_test_points = min(20, len(motor_eff_data.rpm_data))  # Test up to 20 points
    
    # Generate random indices
    np.random.seed(42)  # For reproducible results
    test_indices = np.random.choice(len(motor_eff_data.rpm_data), n_test_points, replace=False)
    
    print(f"Testing efficiency interpolation with {n_test_points} actual data points...")
    
    # Set up the model for efficiency interpolation
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Use actual data values
    test_rpm = motor_eff_data.rpm_data[test_indices]
    test_torque = motor_eff_data.torque_data[test_indices]
    actual_eff = motor_eff_data.eff_data[test_indices]
    
    ivc.add_output('rpm', test_rpm, units='rpm', desc='Motor speed')
    ivc.add_output('torque', test_torque, units='N*m', desc='Motor torque')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    
    # Add efficiency interpolation component
    
    model.add_subsystem('motor_eff_interp', MotorEfficiencyRBFInterpolator(num_nodes=n_test_points), promotes=['*'])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    prob.run_model()
    
    # Get interpolated efficiency values
    interpolated_eff = prob.get_val('eff')
    
    # Calculate errors
    eff_errors = np.abs(interpolated_eff - actual_eff)
    eff_relative_errors = eff_errors / actual_eff * 100
    
    print(f"Efficiency Interpolation Results:")
    print(f"RPM range: {test_rpm.min():.0f} - {test_rpm.max():.0f} rpm")
    print(f"Torque range: {test_torque.min():.1f} - {test_torque.max():.1f} N*m")
    print(f"Actual efficiency range: {actual_eff.min():.3f} - {actual_eff.max():.3f}")
    print(f"Interpolated efficiency range: {interpolated_eff.min():.3f} - {interpolated_eff.max():.3f}")
    print(f"Mean absolute error: {np.mean(eff_errors):.6f}")
    print(f"Max absolute error: {np.max(eff_errors):.6f}")
    print(f"Mean relative error: {np.mean(eff_relative_errors):.2f}%")
    print(f"Max relative error: {np.max(eff_relative_errors):.2f}%")
    
    # Test 2: Voltage/Power Limit Interpolation
    print("\n=== Testing Voltage/Power Limit Interpolation ===")
    
    # Get the voltage data
    motor_volts_data = MotorDataPowerVoltCurve.get_data(motor_filename='openconcept/propulsion/empirical_data/H3X_HPDM_2300_volts.xlsx')
    
    # Use actual data points for testing
    n_test_points_volts = min(20, len(motor_volts_data.rpm_data))  # Test up to 20 points
    
    # Generate random indices for voltage test
    np.random.seed(123)  # Different seed for different test points
    test_indices_volts = np.random.choice(len(motor_volts_data.rpm_data), n_test_points_volts, replace=False)
    
    print(f"Testing voltage/power interpolation with {n_test_points_volts} actual data points...")
    
    # Set up the model for voltage/power interpolation
    model2 = om.Group()
    ivc2 = om.IndepVarComp()
    
    # Use actual data values
    test_rpm_volts = motor_volts_data.rpm_data[test_indices_volts]
    test_voltage = motor_volts_data.voltage_data[test_indices_volts]
    actual_power_lim = motor_volts_data.power_data[test_indices_volts]
    
    ivc2.add_output('rpm', test_rpm_volts, units='rpm', desc='Motor speed')
    ivc2.add_output('voltage', test_voltage, units='V', desc='Motor voltage')
    
    model2.add_subsystem('ivc', ivc2, promotes=['*'])
    
    # Add voltage/power interpolation component
    motor_voltage_interp = MotorVoltagePowerRBFInterpolator(num_nodes=n_test_points_volts)
    
    model2.add_subsystem('motor_voltage_interp', motor_voltage_interp, promotes=['*'])
    
    prob2 = om.Problem(model2, reports=False)
    prob2.setup()
    prob2.run_model()
    
    # Get interpolated power limit values
    interpolated_power_lim = prob2.get_val('mech_power_lim')
    
    # Calculate errors
    power_errors = np.abs(interpolated_power_lim - actual_power_lim)
    power_relative_errors = power_errors / actual_power_lim * 100
    
    print(f"Voltage/Power Interpolation Results:")
    print(f"RPM range: {test_rpm_volts.min():.0f} - {test_rpm_volts.max():.0f} rpm")
    print(f"Voltage range: {test_voltage.min():.0f} - {test_voltage.max():.0f} V")
    print(f"Actual power limit range: {actual_power_lim.min():.1f} - {actual_power_lim.max():.1f} kW")
    print(f"Interpolated power limit range: {interpolated_power_lim.min():.1f} - {interpolated_power_lim.max():.1f} kW")
    print(f"Mean absolute error: {np.mean(power_errors):.6f} kW")
    print(f"Max absolute error: {np.max(power_errors):.6f} kW")
    print(f"Mean relative error: {np.mean(power_relative_errors):.2f}%")
    print(f"Max relative error: {np.max(power_relative_errors):.2f}%")
    
    # Test 3: Power to Efficiency Interpolation
    print("\n=== Testing Power to Efficiency Interpolation ===")
    
    # Get the power to efficiency data
    motor_power_eff_data = MotorDataPowerEffCurve.get_data(motor_filename='openconcept/propulsion/empirical_data/H3X_HPDM-XXXX_1MW.xlsx')
    
    # Use actual data points for testing
    n_test_points_power_eff = min(20, len(motor_power_eff_data.voltage_data))  # Test up to 20 points
    
    # Generate random indices for power to efficiency test
    np.random.seed(456)  # Different seed for different test points
    test_indices_power_eff = np.random.choice(len(motor_power_eff_data.voltage_data), n_test_points_power_eff, replace=False)
    
    print(f"Testing power to efficiency interpolation with {n_test_points_power_eff} actual data points...")
    
    # Set up the model for power to efficiency interpolation
    model3 = om.Group()
    ivc3 = om.IndepVarComp()
    
    # Use actual data values
    test_voltage_power_eff = motor_power_eff_data.voltage_data[test_indices_power_eff]
    test_power_power_eff = motor_power_eff_data.power_data__W[test_indices_power_eff]
    actual_eff_power_eff = motor_power_eff_data.eff_data[test_indices_power_eff]
    
    ivc3.add_output('voltage', test_voltage_power_eff, units='V', desc='Motor voltage')
    ivc3.add_output('mech_power', test_power_power_eff, units='W', desc='Motor power')
    
    model3.add_subsystem('ivc', ivc3, promotes=['*'])
    
    # Add power to efficiency interpolation component
    motor_power_eff_interp = MotorPowerEfficiencyRBFInterpolator(num_nodes=n_test_points_power_eff)
    
    model3.add_subsystem('motor_power_eff_interp', motor_power_eff_interp, promotes=['*'])
    
    prob3 = om.Problem(model3, reports=False)
    prob3.setup()
    prob3.run_model()
    
    # Get interpolated efficiency values
    interpolated_eff_power_eff = prob3.get_val('eff')
    
    # Calculate errors
    eff_power_eff_errors = np.abs(interpolated_eff_power_eff - actual_eff_power_eff)
    eff_power_eff_relative_errors = eff_power_eff_errors / actual_eff_power_eff * 100
    
    print(f"Power to Efficiency Interpolation Results:")
    print(f"Voltage range: {test_voltage_power_eff.min():.0f} - {test_voltage_power_eff.max():.0f} V")
    print(f"Power range: {test_power_power_eff.min():.1f} - {test_power_power_eff.max():.1f} W")
    print(f"Actual efficiency range: {actual_eff_power_eff.min():.3f} - {actual_eff_power_eff.max():.3f}")
    print(f"Interpolated efficiency range: {interpolated_eff_power_eff.min():.3f} - {interpolated_eff_power_eff.max():.3f}")
    print(f"Mean absolute error: {np.mean(eff_power_eff_errors):.6f}")
    print(f"Max absolute error: {np.max(eff_power_eff_errors):.6f}")
    print(f"Mean relative error: {np.mean(eff_power_eff_relative_errors):.2f}%")
    print(f"Max relative error: {np.max(eff_power_eff_relative_errors):.2f}%")
    
    # Create comparison plots
    plt.figure(figsize=(15, 15))
    
    # Efficiency interpolation plots
    plt.subplot(3, 3, 1)
    plt.scatter(actual_eff, interpolated_eff, alpha=0.6, s=20)
    plt.plot([actual_eff.min(), actual_eff.max()], [actual_eff.min(), actual_eff.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Efficiency')
    plt.ylabel('Interpolated Efficiency')
    plt.title('Efficiency: Actual vs Interpolated')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 2)
    plt.hist(eff_errors, bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Efficiency Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 3)
    plt.scatter(actual_eff, eff_relative_errors, alpha=0.6, s=20)
    plt.xlabel('Actual Efficiency')
    plt.ylabel('Relative Error (%)')
    plt.title('Efficiency Error vs Actual Value')
    plt.grid(True, alpha=0.3)
    
    # Voltage/power interpolation plots
    plt.subplot(3, 3, 4)
    plt.scatter(actual_power_lim, interpolated_power_lim, alpha=0.6, s=20)
    plt.plot([actual_power_lim.min(), actual_power_lim.max()], [actual_power_lim.min(), actual_power_lim.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Power Limit (kW)')
    plt.ylabel('Interpolated Power Limit (kW)')
    plt.title('Power Limit: Actual vs Interpolated')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 5)
    plt.hist(power_errors, bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error (kW)')
    plt.ylabel('Frequency')
    plt.title('Power Limit Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 6)
    plt.scatter(actual_power_lim, power_relative_errors, alpha=0.6, s=20)
    plt.xlabel('Actual Power Limit (kW)')
    plt.ylabel('Relative Error (%)')
    plt.title('Power Limit Error vs Actual Value')
    plt.grid(True, alpha=0.3)
    
    # Power to Efficiency interpolation plots
    plt.subplot(3, 3, 7)
    plt.scatter(actual_eff_power_eff, interpolated_eff_power_eff, alpha=0.6, s=20)
    plt.plot([actual_eff_power_eff.min(), actual_eff_power_eff.max()], [actual_eff_power_eff.min(), actual_eff_power_eff.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Efficiency')
    plt.ylabel('Interpolated Efficiency')
    plt.title('Power to Efficiency: Actual vs Interpolated')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 8)
    plt.hist(eff_power_eff_errors, bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Power to Efficiency Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 9)
    plt.scatter(actual_eff_power_eff, eff_power_eff_relative_errors, alpha=0.6, s=20)
    plt.xlabel('Actual Efficiency')
    plt.ylabel('Relative Error (%)')
    plt.title('Power to Efficiency Error vs Actual Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return (interpolated_eff, actual_eff, eff_errors, eff_relative_errors,
            interpolated_power_lim, actual_power_lim, power_errors, power_relative_errors,
            interpolated_eff_power_eff, actual_eff_power_eff, eff_power_eff_errors, eff_power_eff_relative_errors)


class ComputeMotorPower(om.ExplicitComponent):
    """
    Computes the mechanical power output of the motor given torque, rpm, and voltage command.
    Limits power based on voltage constraints and updates torque if needed.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        # Load data immediately when module is imported - this ensures it's available for all components



    def setup(self):
        num_nodes = self.options['num_nodes']
        
        # Inputs
        self.add_input('torque_cmd', val=np.ones(num_nodes), units='N*m', 
                      desc='Commanded torque')
        self.add_input('rpm', val=np.ones(num_nodes), units='rpm', 
                      desc='Motor rotational speed')
        self.add_input('mech_power_lim', val=np.ones(num_nodes), units='kW', 
                      desc='Power limit from voltage interpolation')
        self.add_input('voltage', val=np.ones(num_nodes), units='V', 
                      desc='Motor voltage')
        
        # Outputs
        self.add_output('mech_power', val=np.ones(num_nodes), units='W', 
                       desc='Mechanical power output')
        self.add_output('torque', val=np.ones(num_nodes), units='N*m', 
                       desc='Actual torque (may be limited by power constraints)')
        
        # Declare partials
        ar = np.arange(num_nodes)
        self.declare_partials('mech_power', ['torque_cmd', 'rpm'], rows=ar, cols=ar)
        self.declare_partials('mech_power', ['mech_power_lim'], rows=ar, cols=ar)
        self.declare_partials('torque', ['torque_cmd', 'rpm'], rows=ar, cols=ar)
        self.declare_partials('torque', ['mech_power_lim'], rows=ar, cols=ar)
    
    def compute(self, inputs, outputs):
        torque_cmd = inputs['torque_cmd']
        rpm = inputs['rpm']
        voltage = inputs['voltage']
        mech_power_lim = inputs['mech_power_lim'] # kW
        
        power_lim = np.where(voltage > 700, torque_cmd * rpm * 2 * np.pi / 60, np.where(voltage < 400, 1e-3, mech_power_lim * 1000))  # Convert kW to W
        
        if np.any(power_lim <= 1e-3):
            print(f"WARNING: Motor voltage betlow cuf-off at indexes: {np.where(power_lim <= 1e-3)}")

        # Convert rpm to rad/s
        omega = rpm * 2 * np.pi / 60
        
        # Calculate requested mechanical power
        mech_power_req = torque_cmd * omega
        
        # Limit power to voltage constraint
        mech_power = np.minimum(mech_power_req, power_lim)
        
        # Calculate actual torque (may be limited by power constraints)
        # Avoid division by zero
        omega_safe = np.where(omega > 1e-6, omega, 1e-6)
        torque = mech_power / omega_safe
        
        outputs['mech_power'] = mech_power
        outputs['torque'] = torque
    
    def compute_partials(self, inputs, partials):
        torque_cmd = inputs['torque_cmd']
        rpm = inputs['rpm']
        mech_power_lim = inputs['mech_power_lim'] * 1000  # Convert kW to W
        
        # Convert rpm to rad/s
        omega = rpm * 2 * np.pi / 60
        omega_safe = np.where(omega > 1e-6, omega, 1e-6)
        
        # Calculate requested mechanical power
        mech_power_req = torque_cmd * omega
        
        # Determine which constraint is active
        power_limited = mech_power_req > mech_power_lim
        
        # Partial derivatives for mech_power
        partials['mech_power', 'torque_cmd'] = np.where(power_limited, 0, omega)
        partials['mech_power', 'rpm'] = np.where(power_limited, 0, torque_cmd * 2 * np.pi / 60)
        partials['mech_power', 'mech_power_lim'] = np.where(power_limited, 1000, 0)  # 1000 for kW to W conversion
        
        # Partial derivatives for torque
        partials['torque', 'torque_cmd'] = np.where(power_limited, 0, 1)
        partials['torque', 'rpm'] = np.where(power_limited, 
                                                   -mech_power_lim / (omega_safe**2) * 2 * np.pi / 60, 
                                                   0)
        partials['torque', 'mech_power_lim'] = np.where(power_limited, 1000 / omega_safe, 0)


class ComputeMotorElecDraw(om.ExplicitComponent):
    """
    Computes the electrical power draw of the motor based on mechanical power and efficiency.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
    
    def setup(self):
        num_nodes = self.options['num_nodes']
        
        # Inputs
        self.add_input('mech_power', val=np.ones(num_nodes), units='W', 
                      desc='Mechanical power output')
        self.add_input('eff', val=np.ones(num_nodes), units=None, 
                      desc='Motor efficiency')
        
        # Outputs
        self.add_output('elec_power', val=np.ones(num_nodes), units='W', 
                       desc='Electrical power draw')
        
        # Declare partials
        ar = np.arange(num_nodes)
        self.declare_partials('elec_power', ['mech_power', 'eff'], rows=ar, cols=ar)
    
    def compute(self, inputs, outputs):
        mech_power = inputs['mech_power']
        eff = inputs['eff']
        
        # Avoid division by zero and negative efficiency
        eff_safe = np.maximum(eff, 1e-6)
        
        # Electrical power = mechanical power / efficiency
        elec_power = mech_power / eff_safe
        
        outputs['elec_power'] = elec_power
    
    def compute_partials(self, inputs, partials):
        mech_power = inputs['mech_power']
        eff = inputs['eff']
        
        # Avoid division by zero and negative efficiency
        eff_safe = np.maximum(eff, 1e-6)
        
        # Partial derivatives
        partials['elec_power', 'mech_power'] = 1.0 / eff_safe
        partials['elec_power', 'eff'] = -mech_power / (eff_safe**2)


class EmpiricalMotor(om.Group):
    """
    Complete motor group that combines interpolation with power computation.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('power_set', default=False, desc='True when power is set by the user and specific torque/rpm combo is not known')
        self.options.declare('torque_rpm_set', default=False, desc='True when torque and RPM are set by the user and power is an output')
        
        # Load motor data
        MotorDataEffMap.load_data(motor_filename='openconcept/propulsion/empirical_data/H3X_HPDM_2300_eff.xlsx')
        MotorDataPowerVoltCurve.load_data(motor_filename='openconcept/propulsion/empirical_data/H3X_HPDM_2300_volts.xlsx')
        MotorDataPowerEffCurve.load_data(motor_filename='openconcept/propulsion/empirical_data/H3X_HPDM-XXXX_1MW.xlsx')
    
    def setup(self):
        num_nodes = self.options['num_nodes']    
        power_set = self.options['power_set']
        torque_rpm_set = self.options['torque_rpm_set']

        # Add all subsystems
        if torque_rpm_set:
            self.add_subsystem('motor_voltage_interp', MotorVoltagePowerRBFInterpolator(num_nodes=num_nodes), promotes=['*'])
            self.add_subsystem('compute_power', ComputeMotorPower(num_nodes=num_nodes), promotes=['*'])
            self.add_subsystem('motor_eff_interp', MotorEfficiencyRBFInterpolator(num_nodes=num_nodes), promotes=['*'])
        elif power_set:
            self.add_subsystem('motor_power_interp', MotorPowerEfficiencyRBFInterpolator(num_nodes=num_nodes), promotes=['*'])
        else:
            raise ValueError(f"Invalid mode. Choose one of: power_set={power_set}, torque_rpm_set={torque_rpm_set}")
        # end 

        # Add the electrical power computation
        self.add_subsystem('compute_elec', ComputeMotorElecDraw(num_nodes=num_nodes), promotes=['*'])
        



def test_motor_components():
    """
    Test the new motor power computation components
    """
    print("Testing motor power computation components...")
    
    num_nodes = 5
    
    # Set up the OpenMDAO model
    model = om.Group()
    ivc = om.IndepVarComp()

    power_set = True
    torque_rpm_set = False

    ivc.add_output('voltage', 700 * np.ones(num_nodes), units='V', desc='Motor voltage')
    
    # Add independent variables
    if torque_rpm_set:
        ivc.add_output('torque_cmd', 5000 * np.ones(num_nodes), units='N*m', desc='Commanded torque')
        ivc.add_output('rpm', 1500 * np.ones(num_nodes), units='rpm', desc='Motor speed')
    elif power_set:
        ivc.add_output('mech_power', 1000 * np.ones(num_nodes), units='kW', desc='Commanded power')

    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('motor', EmpiricalMotor(num_nodes=num_nodes), promotes=['*'])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    om.n2(prob)
    
    # Run the model
    prob.run_model()
    
    # Get results
    mech_power = prob.get_val('mech_power', units='W')
    elec_power = prob.get_val('elec_power', units='W')
    eff = prob.get_val('eff')
    
    print(f"Results for {num_nodes} nodes:")
    if torque_rpm_set:
        print(f"Commanded torque: {prob.get_val('torque_cmd', units='N*m')}")
        print(f"RPM: {prob.get_val('rpm', units='rpm')}")
        print(f"Voltage: {prob.get_val('voltage', units='V')}")
    elif power_set:
        print(f"Commanded power: {prob.get_val('mech_power', units='kW')}")

    print(f"Mechanical power: {mech_power} W")
    print(f"Efficiency: {eff}")
    print(f"Electrical power: {elec_power} W")
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    


if __name__ == "__main__":
    
    # Test interpolation accuracy (now includes both efficiency and voltage/power tests)
    test_interpolation_accuracy()

    # Test motor components
    test_motor_components()
