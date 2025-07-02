#!/usr/bin/env python3
"""
Test script for motor interpolation components using actual motor data.
"""

import numpy as np
import openmdao.api as om
from openconcept.propulsion.motor_empirical_metamodel import ComputeMotorPower, ComputeMotorElecDraw, EmpiricalMotor
from openconcept.propulsion.motor_data_graph import MotorDataGraphEff, MotorDataGraphVolts


def test_efficiency_interpolation():
    """
    Test efficiency interpolation using actual data from H3X_HPDM_2300_eff.xlsx
    """
    print("=== Testing Efficiency Interpolation ===")
    
    # Get the efficiency data
    motor_eff_data = MotorDataGraphEff.get_data()
    
    # Use actual data points for testing
    n_test_points = min(10, len(motor_eff_data.rpm_data))  # Test up to 10 points
    
    # Select random test points from the actual data
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
    motor_eff_interp = om.MetaModelUnStructuredComp(vec_size=n_test_points, default_surrogate=om.ResponseSurface())
    motor_eff_interp.add_input('rpm', 1.0, training_data=MotorDataGraphEff.rpm_data, units=None, shape=(n_test_points,))
    motor_eff_interp.add_input('torque', 1.0, training_data=MotorDataGraphEff.torque_data, units=None, shape=(n_test_points,))
    motor_eff_interp.add_output('eff', 1.0, training_data=MotorDataGraphEff.eff_data, units=None, shape=(n_test_points,))
    
    model.add_subsystem('motor_eff_interp', motor_eff_interp, promotes=['*'])
    
    prob = om.Problem(model)
    prob.setup()
    prob.run_model()
    
    # Get interpolated efficiency values
    interpolated_eff = prob.get_val('eff')
    
    # Calculate errors
    errors = np.abs(interpolated_eff - actual_eff)
    relative_errors = errors / actual_eff * 100
    
    print(f"\nEfficiency Interpolation Results:")
    print(f"Test points: {n_test_points}")
    print(f"RPM range: {test_rpm.min():.0f} - {test_rpm.max():.0f} rpm")
    print(f"Torque range: {test_torque.min():.1f} - {test_torque.max():.1f} N*m")
    print(f"Actual efficiency range: {actual_eff.min():.3f} - {actual_eff.max():.3f}")
    print(f"Interpolated efficiency range: {interpolated_eff.min():.3f} - {interpolated_eff.max():.3f}")
    print(f"Mean absolute error: {np.mean(errors):.6f}")
    print(f"Max absolute error: {np.max(errors):.6f}")
    print(f"Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"Max relative error: {np.max(relative_errors):.2f}%")
    
    # Check partials
    print("\nChecking efficiency interpolation partials...")
    prob.check_partials(compact_print=True)
    
    return interpolated_eff, actual_eff, errors, relative_errors


def test_voltage_power_interpolation():
    """
    Test voltage/power limit interpolation using actual data from H3X_HPDM_2300_volts.xlsx
    """
    print("\n=== Testing Voltage/Power Limit Interpolation ===")
    
    # Get the voltage data
    motor_volts_data = MotorDataGraphVolts.get_data()
    
    # Use actual data points for testing
    n_test_points = min(10, len(motor_volts_data.rpm_data))  # Test up to 10 points
    
    # Select random test points from the actual data
    np.random.seed(123)  # Different seed for different test points
    test_indices = np.random.choice(len(motor_volts_data.rpm_data), n_test_points, replace=False)
    
    print(f"Testing voltage/power interpolation with {n_test_points} actual data points...")
    
    # Set up the model for voltage/power interpolation
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Use actual data values
    test_rpm = motor_volts_data.rpm_data[test_indices]
    test_voltage = motor_volts_data.voltage_data[test_indices]
    actual_power_lim = motor_volts_data.power_data[test_indices]
    
    ivc.add_output('rpm', test_rpm, units='rpm', desc='Motor speed')
    ivc.add_output('voltage', test_voltage, units='V', desc='Motor voltage')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    
    # Add voltage/power interpolation component
    motor_voltage_interp = om.MetaModelUnStructuredComp(vec_size=n_test_points, default_surrogate=om.ResponseSurface())
    motor_voltage_interp.add_input('rpm', 1.0, training_data=MotorDataGraphVolts.rpm_data, units=None, shape=(n_test_points,))
    motor_voltage_interp.add_input('voltage', 1.0, training_data=MotorDataGraphVolts.voltage_data, units=None, shape=(n_test_points,))
    motor_voltage_interp.add_output('mech_power_lim', 1.0, training_data=MotorDataGraphVolts.power_data, units=None, shape=(n_test_points,))
    
    model.add_subsystem('motor_voltage_interp', motor_voltage_interp, promotes=['*'])
    
    prob = om.Problem(model)
    prob.setup()
    prob.run_model()
    
    # Get interpolated power limit values
    interpolated_power_lim = prob.get_val('mech_power_lim')
    
    # Calculate errors
    errors = np.abs(interpolated_power_lim - actual_power_lim)
    relative_errors = errors / actual_power_lim * 100
    
    print(f"\nVoltage/Power Interpolation Results:")
    print(f"Test points: {n_test_points}")
    print(f"RPM range: {test_rpm.min():.0f} - {test_rpm.max():.0f} rpm")
    print(f"Voltage range: {test_voltage.min():.0f} - {test_voltage.max():.0f} V")
    print(f"Actual power limit range: {actual_power_lim.min():.1f} - {actual_power_lim.max():.1f} kW")
    print(f"Interpolated power limit range: {interpolated_power_lim.min():.1f} - {interpolated_power_lim.max():.1f} kW")
    print(f"Mean absolute error: {np.mean(errors):.6f} kW")
    print(f"Max absolute error: {np.max(errors):.6f} kW")
    print(f"Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"Max relative error: {np.max(relative_errors):.2f}%")
    
    # Check partials
    print("\nChecking voltage/power interpolation partials...")
    prob.check_partials(compact_print=True)
    
    return interpolated_power_lim, actual_power_lim, errors, relative_errors


def test_complete_motor_system():
    """
    Test the complete motor system with actual data
    """
    print("\n=== Testing Complete Motor System ===")
    
    # Get data from both datasets
    motor_eff_data = MotorDataGraphEff.get_data()
    motor_volts_data = MotorDataGraphVolts.get_data()
    
    # Use a smaller number of test points for the complete system
    n_test_points = 5
    
    # Select test points from efficiency data
    np.random.seed(456)
    test_indices = np.random.choice(len(motor_eff_data.rpm_data), n_test_points, replace=False)
    
    print(f"Testing complete motor system with {n_test_points} actual data points...")
    
    # Set up the complete motor model
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Use actual data values
    test_rpm = motor_eff_data.rpm_data[test_indices]
    test_torque_cmd = motor_eff_data.torque_data[test_indices]
    test_voltage = 700 * np.ones(n_test_points)  # Use constant voltage for simplicity
    
    ivc.add_output('torque_cmd', test_torque_cmd, units='N*m', desc='Commanded torque')
    ivc.add_output('rpm', test_rpm, units='rpm', desc='Motor speed')
    ivc.add_output('voltage', test_voltage, units='V', desc='Motor voltage')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('motor', EmpiricalMotor(num_nodes=n_test_points), promotes=['*'])
    
    prob = om.Problem(model)
    prob.setup()
    prob.run_model()
    
    # Get all outputs
    mech_power = prob.get_val('mech_power', units='W')
    torque_actual = prob.get_val('torque', units='N*m')
    elec_power = prob.get_val('elec_power', units='W')
    eff = prob.get_val('eff')
    mech_power_lim = prob.get_val('mech_power_lim', units='kW')
    
    print(f"\nComplete Motor System Results:")
    print(f"Test points: {n_test_points}")
    print(f"RPM range: {test_rpm.min():.0f} - {test_rpm.max():.0f} rpm")
    print(f"Commanded torque range: {test_torque_cmd.min():.1f} - {test_torque_cmd.max():.1f} N*m")
    print(f"Voltage: {test_voltage[0]:.0f} V (constant)")
    print(f"Power limit range: {mech_power_lim.min():.1f} - {mech_power_lim.max():.1f} kW")
    print(f"Actual torque range: {torque_actual.min():.1f} - {torque_actual.max():.1f} N*m")
    print(f"Mechanical power range: {mech_power.min():.0f} - {mech_power.max():.0f} W")
    print(f"Efficiency range: {eff.min():.3f} - {eff.max():.3f}")
    print(f"Electrical power range: {elec_power.min():.0f} - {elec_power.max():.0f} W")
    
    # Check partials
    print("\nChecking complete motor system partials...")
    prob.check_partials(compact_print=True)
    
    return mech_power, torque_actual, elec_power, eff


if __name__ == "__main__":
    # Test efficiency interpolation
    eff_interp, eff_actual, eff_errors, eff_rel_errors = test_efficiency_interpolation()
    
    # Test voltage/power interpolation
    power_interp, power_actual, power_errors, power_rel_errors = test_voltage_power_interpolation()
    
    # Test complete motor system
    mech_power, torque_actual, elec_power, eff = test_complete_motor_system()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*50) 