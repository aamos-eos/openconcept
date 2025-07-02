import numpy as np
import openmdao.api as om

import numpy as np
import cProfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RBFInterpolator, interpn

import json
#import jax 
#jax.config.update("jax_enable_x64", True)

import os, sys

# Import the global data store
from openconcept.propulsion.battery_data import BatteryData


class MotorVoltage(om.ExplicitComponent):
    """
    Simple explicit component to determine the voltage at the motor.
    
    Takes i_cell and p_train_elec, n_motor, n_str and n_parallel_per_str as inputs.
    Calculates p_motor = p_train_elec / n_motors
    and v_motor = p_motor / (i_cell * n_str * n_parallel_per_str)
    
    Parameters
    ----------
    num_nodes : int
        Number of analysis points (default: 1)
    
    Inputs
    ------
    i_cell : array_like
        Cell current [A]
    p_train_elec : array_like
        Total propulsion electric power demand [W]
    n_motor : int
        Number of electric motors
    n_str : int
        Number of battery strings
    n_parallel_per_str : int
        Number of cells in parallel per string
    
    Outputs
    -------
    v_motor : array_like
        Voltage at the motor [V]
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
    
    def setup(self):
        nn = self.options['num_nodes']
        
        self.add_input('i_cell', units='A', shape=(nn,),
                      desc='Cell current')
        self.add_input('p_train_elec', units='W', shape=(nn,),
                      desc='Total propulsion electric power demand')
        self.add_input('n_motor', units=None, shape=(1,),
                      desc='Number of electric motors')
        self.add_input('n_str', units=None, shape=(1,),
                      desc='Number of battery strings')
        self.add_input('n_parallel_per_str', units=None, shape=(1,),
                      desc='Number of cells in parallel per string')
        
        self.add_output('v_motor', units='V', shape=(nn,),
                       desc='Voltage at the motor')
        
        # Declare partials
        self.declare_partials('*', '*', method='exact')
    
    def compute(self, inputs, outputs):
        i_cell = inputs['i_cell']
        p_train_elec = inputs['p_train_elec']
        n_motor = inputs['n_motor']
        n_str = inputs['n_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        
        # Calculate power per motor
        p_motor = p_train_elec / n_motor
        
        # Calculate voltage at motor
        v_motor = p_motor / (i_cell * n_str * n_parallel_per_str)
        
        outputs['v_motor'] = v_motor
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        
        i_cell = inputs['i_cell']
        p_train_elec = inputs['p_train_elec']
        n_motor = inputs['n_motor']
        n_str = inputs['n_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        
        # Calculate intermediate values
        p_motor = p_train_elec / n_motor
        denominator = i_cell * n_str * n_parallel_per_str
        
        # ∂(v_motor)/∂(i_cell) = -p_motor / (i_cell^2 * n_str * n_parallel_per_str)
        partials['v_motor', 'i_cell'] = -p_motor / (i_cell ** 2 * n_str * n_parallel_per_str) * np.eye(nn)
        
        # ∂(v_motor)/∂(p_train_elec) = 1 / (n_motor * i_cell * n_str * n_parallel_per_str)
        partials['v_motor', 'p_train_elec'] = 1.0 / (n_motor * i_cell * n_str * n_parallel_per_str) * np.eye(nn)
        
        # ∂(v_motor)/∂(n_motor) = -p_train_elec / (n_motor^2 * i_cell * n_str * n_parallel_per_str)
        partials['v_motor', 'n_motor'] = -p_train_elec / (n_motor ** 2 * i_cell * n_str * n_parallel_per_str)
        
        # ∂(v_motor)/∂(n_str) = -p_motor / (i_cell * n_str^2 * n_parallel_per_str)
        partials['v_motor', 'n_str'] = -p_motor / (i_cell * n_str ** 2 * n_parallel_per_str)
        
        # ∂(v_motor)/∂(n_parallel_per_str) = -p_motor / (i_cell * n_str * n_parallel_per_str^2)
        partials['v_motor', 'n_parallel_per_str'] = -p_motor / (i_cell * n_str * n_parallel_per_str ** 2)


    

# Battery Pack

class OCVInterp1D(om.ExplicitComponent):
    """
    Explicit component for 1D interpolation of SOC to OCV using interpn
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        num_nodes = self.options['num_nodes']
        
        self.add_input('soc', units=None, shape=(num_nodes,), desc='State of charge')
        self.add_output('ocv_cell', units='V', shape=(num_nodes,), desc='Open circuit voltage of the battery cell')
        
        # Store the training data for interpolation
        self.soc_data = BatteryData.soc_data
        self.ocv_data = BatteryData.ocv_data
        
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        soc = inputs['soc']
        
        # Vectorized interpolation using interpn
        ocv_cell = interpn((self.soc_data,), self.ocv_data, soc, method='linear', bounds_error=False, fill_value=None)
        
        outputs['ocv_cell'] = ocv_cell


class LineVoltageInterp2D(om.ExplicitComponent):
    """
    Explicit component for 2D interpolation of SOC + C-rate to line voltage using RBF
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('v_cutoff', default=2.5, desc='Voltage cutoff for cell')

    def setup(self):
        num_nodes = self.options['num_nodes']
        
        self.add_input('soc', units=None, shape=(num_nodes,), desc='State of charge')
        self.add_input('c_rate', units='1/h', shape=(num_nodes,), desc='C-rate of the cell')
        self.add_output('vline_cell', units='V', shape=(num_nodes,), desc='Voltage of the battery cell')
        
        # Create the RBF interpolator using the global data
        soc_mesh_flat = BatteryData.soc_mesh_flat
        c_rate_mesh_flat = BatteryData.c_rate_mesh_flat
        line_voltage_flat = BatteryData.line_voltage_flat
        
        # Create training points for RBF
        training_points = np.column_stack([soc_mesh_flat, c_rate_mesh_flat])
        self.rbf_interpolator = RBFInterpolator(training_points, line_voltage_flat, kernel='thin_plate_spline')
        
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        soc = inputs['soc']
        c_rate = inputs['c_rate']
        
        # Create test points for interpolation
        test_points = np.column_stack([soc, c_rate])
        
        # Interpolate line voltage
        vline_cell = np.maximum(self.rbf_interpolator(test_points), self.options['v_cutoff'])
        
        outputs['vline_cell'] = vline_cell


class IR0Interp2D(om.ExplicitComponent):
    """
    Explicit component for 2D interpolation of SOC + C-rate to IR0 using RBF
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        num_nodes = self.options['num_nodes']
        
        self.add_input('soc', units=None, shape=(num_nodes,), desc='State of charge')
        self.add_input('c_rate', units='1/h', shape=(num_nodes,), desc='C-rate of the cell')
        self.add_output('ir0_cell', units='ohm', shape=(num_nodes,), desc='Internal resistance of the battery cell')
        
        # Create the RBF interpolator using the global data
        soc_mesh_flat = BatteryData.soc_mesh_flat
        c_rate_mesh_flat = BatteryData.c_rate_mesh_flat
        ir0_flat = BatteryData.ir0_flat
        
        # Create training points for RBF
        training_points = np.column_stack([soc_mesh_flat, c_rate_mesh_flat])
        self.rbf_interpolator = RBFInterpolator(training_points, ir0_flat, kernel='thin_plate_spline')
        
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        soc = inputs['soc']
        c_rate = inputs['c_rate']
        
        # Create test points for interpolation
        test_points = np.column_stack([soc, c_rate])
        
        # Interpolate IR0
        ir0_cell = self.rbf_interpolator(test_points)
        
        outputs['ir0_cell'] = ir0_cell


class ChargeVoltage(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']

        self.add_input('i_cell', units='A', shape=(num_nodes,))
        self.add_input('p_cell_charge', units='W', shape=(num_nodes,))
        self.add_output('vline_cell', units='V', shape=(num_nodes,))
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        i_cell = inputs['i_cell']
        p_cell_charge = inputs['p_cell_charge']

        outputs['vline_cell'] = p_cell_charge / i_cell
    
    def compute_partials(self, inputs, partials):
        # Unpack Options
        num_nodes = self.options['num_nodes']

        i_cell = inputs['i_cell']
        p_cell_charge = inputs['p_cell_charge']
        
        partials['vline_cell', 'i_cell'] = np.eye(num_nodes) * (-p_cell_charge / i_cell**2)
        partials['vline_cell', 'p_cell_charge'] = np.eye(num_nodes) * (1 / i_cell)
        
        

class CellChargePowerResidualComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']

        self.add_input('ocv_cell', units='V', shape=(num_nodes,))
        self.add_input('ir0_cell', units='ohm', shape=(num_nodes,))
        self.add_input('p_cell_charge', units='W', shape=(num_nodes,))

        self.add_output('i_cell', units='A', shape=(num_nodes,))

        self.declare_partials('*', '*', method='exact')  # use finite difference for now

    def apply_nonlinear(self, inputs, outputs, residuals):
        i_cell = inputs['i_cell']
        ocv_cell = inputs['ocv_cell']
        ir0_cell = inputs['ir0_cell']
        p_cell_charge = inputs['p_cell_charge']

        i_cell = outputs['i_cell']

        v_cell = ocv_cell + i_cell * ir0_cell
        p_est = i_cell * v_cell

        residuals['i_cell'] = p_est - p_cell_charge

    def linearize(self, inputs, outputs, partials):

        # Unpack Options
        num_nodes = self.options['num_nodes']

        i_cell = inputs['i_cell']
        ocv_cell = inputs['ocv_cell']
        ir0_cell = inputs['ir0_cell']
        # p_cell_tgt = inputs['p_cell_tgt']

        # residual = i_cell * (ocv_cell + i_cell * ir0_cell) - p_cell_tgt
        # d(residual)/d(i_cell) = ocv_cell + 2 * i_cell * ir0_cell
        partials['i_cell', 'i_cell'] = np.eye(num_nodes) * (ocv_cell + 2 * i_cell * ir0_cell)
        # d(residual)/d(ocv_cell) = i_cell
        partials['i_cell', 'ocv_cell'] = np.eye(num_nodes) * i_cell
        # d(residual)/d(ir0_cell) = i_cell ** 2
        partials['i_cell', 'ir0_cell'] = np.eye(num_nodes) * i_cell ** 2
        # d(residual)/d(p_cell_charge) = -1
        partials['i_cell', 'p_cell_charge'] = -1.0 * np.eye(num_nodes)


class UpdateCellState(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('mode', types=str)
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')


    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']

        self.add_input('i_cell', units='A', desc='cell current', shape=(num_nodes,))
        self.add_input('dtime', units='s', desc='time step', shape=(num_nodes,))
        self.add_input('cell_capacity', units='A*h', desc='cell capacity', shape=(1,))
        self.add_input('soc_init', units=None, desc='initial state of charge', shape=(1,))
        self.add_input('ir0_cell', units='ohm', desc='internal resistance of the cell', shape=(num_nodes,))
        self.add_input('n_str', units=None, desc='number of strings', shape=(1,))
        self.add_input('n_series_per_str', units=None, desc='number of cells in series per string', shape=(1,))
        self.add_input('n_parallel_per_str', units=None, desc='number of cells in parallel per string', shape=(1,))
        self.add_input('ocv_cell', units='V', desc='open circuit voltage of the cell', shape=(num_nodes,))
        self.add_input('vline_cell', units='V', desc='line voltage of the cell', shape=(num_nodes,))
        self.add_input('t_cell_init', units='K', desc='initial temperature of the cell', shape=(1,))
        self.add_input('m_cell', units='kg', desc='mass of the cell', shape=(1,))
        self.add_input('cp_cell', units='J/kg/K', desc='specific heat capacity of the cell', shape=(1,))
        self.add_input('q_cool_bat', units='kW', desc='cooling power of the battery', shape=(num_nodes,))

        self.add_output('soc', units=None, desc='state of charge of the battery', shape=(num_nodes,))
        self.add_output('t_cell', units='K', desc='temperature of the cell', shape=(num_nodes,))
        self.add_output('eta_cell', units=None, desc='efficiency of the cell', shape=(num_nodes,))
        self.add_output('net_q_heat_ess', units='W', desc='net heat of the battery', shape=(num_nodes,))
        self.add_output('heat_ess', units='W', desc='heat of the battery', shape=(num_nodes,))
        self.add_output('c_rate', units='1/h', desc='c-rate of the cell', shape=(num_nodes,))

    def compute(self, inputs, outputs):


        # Unpack inputs
        i_cell = inputs['i_cell']
        soc_init = inputs['soc_init']
        dtime = inputs['dtime']
        cap_cell = inputs['cell_capacity']
        ir0_cell = inputs['ir0_cell']
        n_str = inputs['n_str']
        n_series_per_str = inputs['n_series_per_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        q_cool_bat = inputs['q_cool_bat']
        ocv_cell = inputs['ocv_cell']
        vline_cell = inputs['vline_cell']
        t_cell_init = inputs['t_cell_init']
        m_cell = inputs['m_cell']
        cp_cell = inputs['cp_cell']

        cap_delta = np.cumsum(i_cell * dtime / cap_cell)
        n_cells = n_str * n_series_per_str * n_parallel_per_str

        if self.options['mode'] == 'charge':
            soc = soc_init + cap_delta
            net_q_heat_cell = ir0_cell * i_cell**2 - q_cool_bat*1000./ n_cells
            heat_cell = ir0_cell * i_cell**2 

        elif self.options['mode'] == 'discharge':
            soc = soc_init - cap_delta
            net_q_heat_cell = (ocv_cell - vline_cell) * i_cell - q_cool_bat*1000./ n_cells
            heat_cell = (ocv_cell - vline_cell) * i_cell
        else:
            raise ValueError(f"Invalid mode: {self.options['mode']}")
        
        t_cell = t_cell_init + np.cumsum(net_q_heat_cell * dtime / (m_cell * cp_cell))
        eta_cell = 1- heat_cell / (vline_cell * i_cell)

        # Compute number of cells
        n_cells_ess = n_str * n_series_per_str * n_parallel_per_str

        # Battery-level state variables (multiply cell-level by n_cells_total)
        net_q_heat_ess = net_q_heat_cell * n_cells_ess
        heat_ess = heat_cell * n_cells_ess

        # Compute c-rate
        c_rate = i_cell / cap_cell

        outputs['soc'] = soc
        outputs['t_cell'] = t_cell
        outputs['eta_cell'] = eta_cell
        outputs['net_q_heat_ess'] = net_q_heat_ess
        outputs['heat_ess'] = heat_ess
        outputs['c_rate'] = c_rate
    

class DischargeEmpiricalBattery(om.Group):

    def initialize(self):

        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('v_cutoff', default=2.5, desc='Voltage cutoff for cell')

        BatteryData.load_data(bat_filename='openconcept/propulsion/empirical_data/inHouse_battery_1motorConfig_208s27p_4grp_to_each_nacelle.xlsx', 
                            cell_sheetname='BOL_cell_fct_CRate', 
                            config_sheetname='battery_config')

    def setup(self):

        # Unpack Options
        num_nodes = self.options['num_nodes']
        self.add_subsystem('ptrain_hv_in', HVCurrentImplicit(num_nodes = num_nodes), promotes=['*'])
        self.add_subsystem('motor_voltage', MotorVoltage(num_nodes=num_nodes), promotes=['*'])
        self.add_subsystem('aux_out', AuxCurrentImplicit(num_nodes = num_nodes), promotes=['*'])

        self.add_subsystem('update_cell_state', UpdateCellState(mode='discharge', num_nodes = num_nodes),promotes=['*'])

        # 1D interpolation: SOC to OCV only
        self.add_subsystem('ocv_interp', OCVInterp1D(num_nodes=num_nodes), promotes=['*'])

        # 2D interpolation: SOC + C-rate meshgrid to line voltage and IR0
        # Line voltage interpolator (2D: SOC + C-rate)
        self.add_subsystem('line_voltage_crate_interp', LineVoltageInterp2D(num_nodes=num_nodes, v_cutoff=self.options['v_cutoff']), promotes=['*'])

        # IR0 interpolator (2D: SOC + C-rate)
        self.add_subsystem('ir0_map', IR0Interp2D(num_nodes=num_nodes), promotes=['*'])

        # Solvers
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options['maxiter'] = 100
        self.nonlinear_solver.options['atol'] = 1e-6
        self.nonlinear_solver.options['rtol'] = 1e-6
        self.nonlinear_solver.options['iprint'] = 2
        #self.nonlinear_solver.options['iprint_level'] = 0
        self.linear_solver = om.DirectSolver()



class ChargeEmpiricalBattery(om.Group):

    def initialize(self):
        # No need for data options anymore - data is global
        self.options.declare('charge_mode', default='known_power', desc='charge mode')
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
    
    def setup(self):

        # Unpack Options
        num_nodes = self.options['num_nodes']
        # No need to unpack data options anymore
        #ocv_interp = OCVInterp()
        
        # 1D interpolation: SOC to OCV only
        self.add_subsystem('ocv_interp', OCVInterp1D(num_nodes=num_nodes), promotes=['*'])

        # 2D interpolation: SOC + C-rate meshgrid to IR0
        self.add_subsystem('ir0_map', IR0Interp2D(num_nodes=num_nodes), promotes=['*'])

        if self.options['charge_mode'] == 'known_power':
            self.add_subsystem('cell_solver', CellChargePowerResidualComp(num_nodes = num_nodes), promotes=['*'])
            self.add_subsystem('charge_voltage', ChargeVoltage(num_nodes = num_nodes), promotes=['*'])
        elif self.options['charge_mode'] == 'known_current':
            self.add_subsystem('cell_solver', CellChargeCurrent(num_nodes = num_nodes), promotes=['*'])
        else:
            raise ValueError(f"Invalid charge mode: {self.options['charge_mode']}. Options are 'known_power' or 'known_current'.")



        self.add_subsystem('update_cell_state', UpdateCellState(mode='charge', num_nodes = num_nodes),promotes=['*'])

        # Solvers
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.linear_solver = om.DirectSolver()





import openmdao.api as om

class HVCurrentImplicit(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']

        self.add_input('p_train_elec', units='W', desc='Total propulsion electric power demand', shape=(num_nodes,))
        self.add_input('rloop_motor_dc_in', units='ohm', desc='DC loop resistance to each motor', shape=(num_nodes,))
        self.add_input('vline_cell', units='V', desc='cell line voltage', shape=(num_nodes,))
        self.add_input('n_str', desc='number of battery strings', shape=(1,))
        self.add_input('n_parallel_per_str', desc='number of cells in parallel per string', shape=(1,))

        self.add_output('i_cell', units='A', desc='cell current', shape=(num_nodes,))

        self.declare_partials(of='*', wrt='*', method='exact')

    def apply_nonlinear(self, inputs, outputs, residuals):

        p_motor_total = inputs['p_train_elec']
        rloop_motor_dc_in = inputs['rloop_motor_dc_in']
        vline_cell = inputs['vline_cell']
        n_str = inputs['n_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        i_cell = outputs['i_cell']

        p_cell = p_motor_total / n_str / n_parallel_per_str


        residuals['i_cell'] = p_cell - i_cell * (vline_cell - i_cell * rloop_motor_dc_in)

    def linearize(self, inputs, outputs, partials):

        num_nodes = self.options['num_nodes']

        p_motor_total = inputs['p_train_elec']
        rloop_motor_dc_in = inputs['rloop_motor_dc_in']
        vline_cell = inputs['vline_cell']
        n_str = inputs['n_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        i_cell = outputs['i_cell']

        p_cell = p_motor_total / n_str / n_parallel_per_str

        partials['i_cell', 'p_train_elec'] = np.eye(num_nodes) * 1.0 / n_str / n_parallel_per_str
        partials['i_cell', 'rloop_motor_dc_in'] = np.eye(num_nodes) * i_cell **2
        partials['i_cell', 'vline_cell'] = np.eye(num_nodes) * -i_cell
        partials['i_cell', 'n_str'] =  -p_cell / n_str ** 2 / n_parallel_per_str
        partials['i_cell', 'n_parallel_per_str'] = -p_cell / n_str / n_parallel_per_str ** 2
        partials['i_cell', 'i_cell'] = np.eye(num_nodes) * - (vline_cell -2* i_cell * rloop_motor_dc_in)



class AuxCurrentImplicit(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']

        self.add_input('p_aux_elec', units='W', desc='Auxiliary electric load on LV side', shape=(num_nodes,))
        self.add_input('rloop_aux_conv_hv_dc_in', units='ohm', desc='DC loop resistance to LV converter', shape=(num_nodes,))
        self.add_input('vline_cell', units='V', desc='cell line voltage', shape=(num_nodes,))
        self.add_input('eta_converter', units=None, desc='efficiency of the converter', shape=(num_nodes,))

        self.add_output('i_conv_hv', units='A', desc='HV side current into LV converter', shape=(num_nodes,))

        self.declare_partials(of='*', wrt='*', method='exact')

    def apply_nonlinear(self, inputs, outputs, residuals):
        
        p_aux = inputs['p_aux_elec']
        eta = inputs['eta_converter']
        r_conv = inputs['rloop_aux_conv_hv_dc_in']
        i_conv = outputs['i_conv_hv']
        vline_cell = inputs['vline_cell']

        p_conv_in = p_aux / eta

        residuals['i_conv_hv'] = p_conv_in - i_conv * (vline_cell - i_conv * r_conv)

    def linearize(self, inputs, outputs, partials):

        num_nodes = self.options['num_nodes']
        p_aux = inputs['p_aux_elec']
        eta = inputs['eta_converter']
        r_conv = inputs['rloop_aux_conv_hv_dc_in']
        i_conv = outputs['i_conv_hv']
        vline_cell = inputs['vline_cell']

        partials['i_conv_hv', 'p_aux_elec'] = np.eye(num_nodes) * 1.0 / eta
        partials['i_conv_hv', 'eta_converter'] = np.eye(num_nodes) * -p_aux / eta ** 2
        partials['i_conv_hv', 'rloop_aux_conv_hv_dc_in'] = np.eye(num_nodes) * i_conv ** 2
        partials['i_conv_hv', 'vline_cell'] = np.eye(num_nodes) * -i_conv
        partials['i_conv_hv', 'i_conv_hv'] = np.eye(num_nodes) * - (vline_cell - 2 * i_conv * r_conv)


class CellChargeCurrent(om.ExplicitComponent):
    """
    Computes cell voltage and current for a battery cell under constant current/constant voltage charging.
    Variable names are consistent with the rest of the codebase.
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']

        self.add_input('ocv_cell', units='V', desc='Open circuit voltage of the cell', shape=(num_nodes,))
        self.add_input('ir0_cell', units='ohm', desc='Internal resistance of the cell', shape=(num_nodes,))
        self.add_input('i_cell_charge_lim', units='A', desc='Constant current setpoint (pack level)', shape=(1,))
        self.add_input('v_cell_charge_lim', units='V', desc='Constant voltage setpoint (pack level)', shape=(1,))
        self.add_input('n_series_per_str', desc='Number of cells in series', shape=(1,))
        self.add_input('n_parallel_per_str', desc='Number of cells in parallel', shape=(1,))
        self.add_output('vline_cell', units='V', desc='Cell terminal voltage', shape=(num_nodes,))
        self.add_output('i_cell', units='A', desc='Cell current', shape=(num_nodes,))
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        ocv_cell = inputs['ocv_cell']
        ir0_cell = inputs['ir0_cell']
        i_cell_charge_lim = inputs['i_cell_charge_lim']
        v_cell_charge_lim = inputs['v_cell_charge_lim']
        n_series_per_str = inputs['n_series_per_str']
        n_parallel_per_str = inputs['n_parallel_per_str']

        vline_cell = np.minimum(ocv_cell + ir0_cell * i_cell_charge_lim / n_parallel_per_str, v_cell_charge_lim / n_series_per_str)
        i_cell = np.minimum(i_cell_charge_lim, (v_cell_charge_lim - ocv_cell) / ir0_cell)

        outputs['vline_cell'] = vline_cell
        outputs['i_cell'] = i_cell


def test_battery_interpolation_accuracy():
    """
    Test battery interpolation accuracy by comparing interpolated vs actual values
    Tests line voltage and IR0 interpolation (2D RBF) using all 75 data points
    """
    print("Testing battery interpolation accuracy...")
    
    # Get the data
    bat_data = BatteryData.get_data(bat_filename='openconcept/propulsion/empirical_data/inHouse_battery_1motorConfig_208s27p_4grp_to_each_nacelle.xlsx', 
                                    cell_sheetname='BOL_cell_fct_CRate', 
                                    config_sheetname='battery_config')
    
    # Get all available data points for 2D interpolations
    line_voltage_flat = bat_data.line_voltage_flat
    ir0_flat = bat_data.ir0_flat
    soc_mesh_flat = bat_data.soc_mesh_flat
    c_rate_mesh_flat = bat_data.c_rate_mesh_flat
    
    n_total_points = len(line_voltage_flat)  # Should be 75 (15 SOC × 5 C-rate)
    
    print(f"Testing all {n_total_points} data points for 2D interpolations...")
    print(f"SOC range: {soc_mesh_flat.min():.2f} to {soc_mesh_flat.max():.2f}")
    print(f"C-rate range: {c_rate_mesh_flat.min():.2f} to {c_rate_mesh_flat.max():.2f} 1/h")
    print(f"Line voltage range: {line_voltage_flat.min():.2f} to {line_voltage_flat.max():.2f} V")
    print(f"IR0 range: {ir0_flat.min():.4f} to {ir0_flat.max():.4f} ohm")
    
    # Set up the OpenMDAO model to test the actual components
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables for all 75 points
    ivc.add_output('soc', soc_mesh_flat, units=None, desc='State of charge')
    ivc.add_output('c_rate', c_rate_mesh_flat, units='1/h', desc='C-rate of the cell')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    
    # Add the actual interpolation components
    model.add_subsystem('ocv_interp', OCVInterp1D(num_nodes=n_total_points), promotes=['*'])
    model.add_subsystem('line_voltage_interp', LineVoltageInterp2D(num_nodes=n_total_points), promotes=['*'])
    model.add_subsystem('ir0_interp', IR0Interp2D(num_nodes=n_total_points), promotes=['*'])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Run the model
    prob.run_model()
    
    # Get interpolated values from the OpenMDAO components
    vline_interpolated = prob.get_val('vline_cell')
    ir0_interpolated = prob.get_val('ir0_cell')
    
    # Get actual values
    vline_actual = line_voltage_flat
    ir0_actual = ir0_flat
    
    # Calculate statistics for each output using vectorized operations
    vline_errors = np.abs(vline_interpolated - vline_actual)
    vline_relative_errors = vline_errors / vline_actual * 100
    
    ir0_errors = np.abs(ir0_interpolated - ir0_actual)
    ir0_relative_errors = ir0_errors / ir0_actual * 100
    
    # Handle any infinite or NaN values
    vline_relative_errors = np.where(np.isinf(vline_relative_errors), 100.0, vline_relative_errors)
    vline_relative_errors = np.where(np.isnan(vline_relative_errors), 100.0, vline_relative_errors)
    
    ir0_relative_errors = np.where(np.isinf(ir0_relative_errors), 100.0, ir0_relative_errors)
    ir0_relative_errors = np.where(np.isnan(ir0_relative_errors), 100.0, ir0_relative_errors)
    
    print(f"\n=== Battery Interpolation Accuracy Results ===")
    print(f"Line Voltage - Mean absolute error: {np.mean(vline_errors):.4f} V")
    print(f"Line Voltage - Max absolute error: {np.max(vline_errors):.4f} V")
    print(f"Line Voltage - Mean relative error: {np.mean(vline_relative_errors):.2f}%")
    print(f"Line Voltage - Max relative error: {np.max(vline_relative_errors):.2f}%")
    
    print(f"IR0 - Mean absolute error: {np.mean(ir0_errors):.6f} ohm")
    print(f"IR0 - Max absolute error: {np.max(ir0_errors):.6f} ohm")
    print(f"IR0 - Mean relative error: {np.mean(ir0_relative_errors):.2f}%")
    print(f"IR0 - Max relative error: {np.max(ir0_relative_errors):.2f}%")
    
    # Create comprehensive visualization
    plt.figure(figsize=(15, 10))
    
    # Line Voltage plots
    plt.subplot(2, 4, 1)
    plt.scatter(vline_actual, vline_interpolated, alpha=0.6, s=20)
    plt.plot([vline_actual.min(), vline_actual.max()], [vline_actual.min(), vline_actual.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Line Voltage (V)')
    plt.ylabel('Interpolated Line Voltage (V)')
    plt.title('Line Voltage: Actual vs Interpolated')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 2)
    plt.hist(vline_errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error (V)')
    plt.ylabel('Frequency')
    plt.title('Line Voltage Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 3)
    plt.hist(vline_relative_errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    plt.title('Line Voltage Relative Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 4)
    plt.scatter(soc_mesh_flat, vline_relative_errors, alpha=0.6, s=20)
    plt.xlabel('SOC')
    plt.ylabel('Relative Error (%)')
    plt.title('Line Voltage Error vs SOC')
    plt.grid(True, alpha=0.3)
    
    # IR0 plots
    plt.subplot(2, 4, 5)
    plt.scatter(ir0_actual, ir0_interpolated, alpha=0.6, s=20)
    plt.plot([ir0_actual.min(), ir0_actual.max()], [ir0_actual.min(), ir0_actual.max()], 'r--', linewidth=2)
    plt.xlabel('Actual IR0 (ohm)')
    plt.ylabel('Interpolated IR0 (ohm)')
    plt.title('IR0: Actual vs Interpolated')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 6)
    plt.hist(ir0_errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error (ohm)')
    plt.ylabel('Frequency')
    plt.title('IR0 Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 7)
    plt.hist(ir0_relative_errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    plt.title('IR0 Relative Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 8)
    plt.scatter(soc_mesh_flat, ir0_relative_errors, alpha=0.6, s=20)
    plt.xlabel('SOC')
    plt.ylabel('Relative Error (%)')
    plt.title('IR0 Error vs SOC')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create additional analysis plots
    plt.figure(figsize=(15, 10))
    
    # Error vs SOC and C-rate for both outputs
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(soc_mesh_flat, c_rate_mesh_flat, c=vline_relative_errors, cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Line Voltage Relative Error (%)')
    plt.xlabel('SOC')
    plt.ylabel('C-rate (1/h)')
    plt.title('Line Voltage Error vs SOC and C-rate')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(soc_mesh_flat, c_rate_mesh_flat, c=ir0_relative_errors, cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, label='IR0 Relative Error (%)')
    plt.xlabel('SOC')
    plt.ylabel('C-rate (1/h)')
    plt.title('IR0 Error vs SOC and C-rate')
    plt.grid(True, alpha=0.3)
    
    # Cumulative error distributions
    plt.subplot(2, 3, 3)
    sorted_vline_errors = np.sort(vline_relative_errors)
    sorted_ir0_errors = np.sort(ir0_relative_errors)
    cumulative_percent = np.arange(1, len(sorted_vline_errors) + 1) / len(sorted_vline_errors) * 100
    
    plt.plot(sorted_vline_errors, cumulative_percent, 'g-', linewidth=2, label='Line Voltage')
    plt.plot(sorted_ir0_errors, cumulative_percent, 'r-', linewidth=2, label='IR0')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Error Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error statistics by ranges using vectorized operations
    plt.subplot(2, 3, 4)
    error_ranges = [0, 0.1, 0.5, 1, 2, 5, 10, 50]
    vline_counts = []
    ir0_counts = []
    
    # Vectorized counting
    for i in range(len(error_ranges)-1):
        vline_count = np.sum((vline_relative_errors >= error_ranges[i]) & (vline_relative_errors < error_ranges[i+1]))
        ir0_count = np.sum((ir0_relative_errors >= error_ranges[i]) & (ir0_relative_errors < error_ranges[i+1]))
        vline_counts.append(vline_count)
        ir0_counts.append(ir0_count)
    
    x = np.arange(len(vline_counts))
    width = 0.35
    
    plt.bar(x - width/2, vline_counts, width, alpha=0.7, label='Line Voltage')
    plt.bar(x + width/2, ir0_counts, width, alpha=0.7, label='IR0')
    
    plt.xlabel('Error Range (%)')
    plt.ylabel('Number of Points')
    plt.title('Error Distribution by Ranges')
    plt.xticks(x, [f'{error_ranges[i]}-{error_ranges[i+1]}' for i in range(len(error_ranges)-1)])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary statistics table
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    # Create summary text
    summary_text = f"""Battery Interpolation Summary

Line Voltage (2D RBF):
  Mean Error: {np.mean(vline_relative_errors):.2f}%
  Max Error: {np.max(vline_relative_errors):.2f}%
  < 0.1%: {np.sum(vline_relative_errors < 0.1):d} ({np.sum(vline_relative_errors < 0.1)/len(vline_relative_errors)*100:.1f}%)
  < 1%: {np.sum(vline_relative_errors < 1):d} ({np.sum(vline_relative_errors < 1)/len(vline_relative_errors)*100:.1f}%)

IR0 (2D RBF):
  Mean Error: {np.mean(ir0_relative_errors):.2f}%
  Max Error: {np.max(ir0_relative_errors):.2f}%
  < 0.1%: {np.sum(ir0_relative_errors < 0.1):d} ({np.sum(ir0_relative_errors < 0.1)/len(ir0_relative_errors)*100:.1f}%)
  < 1%: {np.sum(ir0_relative_errors < 1):d} ({np.sum(ir0_relative_errors < 1)/len(ir0_relative_errors)*100:.1f}%)"""
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics using vectorized operations
    print(f"\n=== Detailed Error Analysis ===")
    print(f"Line Voltage - Points with error < 0.1%: {np.sum(vline_relative_errors < 0.1):d} ({np.sum(vline_relative_errors < 0.1)/len(vline_relative_errors)*100:.1f}%)")
    print(f"Line Voltage - Points with error < 1%: {np.sum(vline_relative_errors < 1):d} ({np.sum(vline_relative_errors < 1)/len(vline_relative_errors)*100:.1f}%)")
    print(f"Line Voltage - Points with error > 5%: {np.sum(vline_relative_errors > 5):d} ({np.sum(vline_relative_errors > 5)/len(vline_relative_errors)*100:.1f}%)")
    
    print(f"IR0 - Points with error < 0.1%: {np.sum(ir0_relative_errors < 0.1):d} ({np.sum(ir0_relative_errors < 0.1)/len(ir0_relative_errors)*100:.1f}%)")
    print(f"IR0 - Points with error < 1%: {np.sum(ir0_relative_errors < 1):d} ({np.sum(ir0_relative_errors < 1)/len(ir0_relative_errors)*100:.1f}%)")
    print(f"IR0 - Points with error > 5%: {np.sum(ir0_relative_errors > 5):d} ({np.sum(ir0_relative_errors > 5)/len(ir0_relative_errors)*100:.1f}%)")
    
    # Find worst cases using vectorized operations
    worst_vline_idx = np.argmax(vline_relative_errors)
    worst_ir0_idx = np.argmax(ir0_relative_errors)
    
    print(f"\n=== Worst Cases ===")
    print(f"Line Voltage - Worst case: SOC={soc_mesh_flat[worst_vline_idx]:.3f}, C-rate={c_rate_mesh_flat[worst_vline_idx]:.2f}, "
          f"Actual={vline_actual[worst_vline_idx]:.3f}V, Interp={vline_interpolated[worst_vline_idx]:.3f}V, "
          f"Error={vline_relative_errors[worst_vline_idx]:.2f}%")
    
    print(f"IR0 - Worst case: SOC={soc_mesh_flat[worst_ir0_idx]:.3f}, C-rate={c_rate_mesh_flat[worst_ir0_idx]:.2f}, "
          f"Actual={ir0_actual[worst_ir0_idx]:.6f}ohm, Interp={ir0_interpolated[worst_ir0_idx]:.6f}ohm, "
          f"Error={ir0_relative_errors[worst_ir0_idx]:.2f}%")
    
    return {
        'vline': (vline_interpolated, vline_actual, vline_errors, vline_relative_errors),
        'ir0': (ir0_interpolated, ir0_actual, ir0_errors, ir0_relative_errors)
    }




if __name__ == "__main__":



    
    # Get the data for any other calculations you need
    bat_data = BatteryData.get_data(bat_filename='openconcept/propulsion/empirical_data/inHouse_battery_1motorConfig_208s27p_4grp_to_each_nacelle.xlsx', 
                                    cell_sheetname='BOL_cell_fct_CRate', 
                                    config_sheetname='battery_config')
    

    
    
    # Create a new instance of the BatteryPack component
    # Numebr of nodes to evaluate
    num_nodes = 10
    #bat_pack = BatteryPack(num_nodes = num_nodes)


    # Set up the OpenMDAO model
    model = om.Group()
    ivc = om.IndepVarComp()

    ivc.add_output('n_strngs', 4, desc='number of battery strings')
    ivc.add_output('p_train_elec', 100 * np.ones(num_nodes), units='W', desc='Total propulsion electric power demand')
    ivc.add_output('n_motor', 4, desc='Number of electric motors')
    ivc.add_output('rloop_motor_dc_in', 0.001 * np.ones(num_nodes), units='ohm', desc='DC loop resistance to each motor')
    ivc.add_output('n_str', bat_data.n_str, desc='number of battery strings')
    ivc.add_output('dtime', 1 * np.ones(num_nodes), units='s', desc='time step')
    ivc.add_output('eta_converter', 0.95 * np.ones(num_nodes), units=None, desc='efficiency of the converter')

    ivc.add_output('p_aux_elec', 100 * np.ones(num_nodes), units='W', desc='Auxiliary electric load on LV side')
    ivc.add_output('rloop_aux_conv_hv_dc_in', 0.001 * np.ones(num_nodes), units='ohm', desc='DC loop resistance to LV converter')

    # Load up Battery Data
    ivc.add_output('cell_capacity', bat_data.cell_Ah_capacity, units='A*h', desc='Cell capacity')
    ivc.add_output('t_cell_init', 30, units='K', desc='Initial temperature of the cell')
    ivc.add_output('m_cell', bat_data.m_cell, units='kg', desc='Mass of the cell')
    ivc.add_output('cp_cell', bat_data.cp_cell, units='J/kg/K', desc='Specific heat capacity of the cell')
    ivc.add_output('q_cool_bat', 25 * np.ones(num_nodes), units='kW', desc='Cooling power of the battery')
    ivc.add_output('soc_init', bat_data.soc_init, units=None, desc='Initial state of charge of the battery')

    # Charge Cell Specific Inputs
    ivc.add_output('i_cell_charge_lim', 700, units='A', desc='Constant current setpoint (pack level)')
    ivc.add_output('v_cell_charge_lim', 860, units='V', desc='Constant voltage setpoint (pack level)')
    ivc.add_output('n_series_per_str', bat_data.n_series_per_str, desc='Number of cells in series')
    ivc.add_output('n_parallel_per_str', bat_data.n_parallel_per_str, desc='Number of cells in parallel')
    ivc.add_output('p_cell_charge', 100 * np.ones(num_nodes), units='kW', desc='Cell charge power')

    


    model.add_subsystem('ivc', ivc, promotes=['*'])
    #model.add_subsystem('charge_cell_group', ChargeEmpiricalBattery(charge_mode='known_power', num_nodes=num_nodes), promotes=['*'])

    model.add_subsystem('discharge_cell_group', DischargeEmpiricalBattery(num_nodes=num_nodes, v_cutoff=2.5), promotes=['*'])

    #model.options['default_surrogate'] = om.NearestNeighbor()
    prob = om.Problem(model, reports=False)
    
    # Final check - all training data

    # Define all inputs 

    
    prob.setup()

    om.n2(prob)
    prob.run_model()


    
    # Print results
    print("\n" + "="*50)
    print("BATTERY SYSTEM RESULTS")
    print("="*50)
    
    # Get SOC values
    soc_values = prob.get_val('soc', units=None)
    print(f"SOC values: {soc_values}")
    
    # Get line voltage values
    vline_cell_values = prob.get_val('vline_cell', units='V')
    print(f"Line voltage values (V): {vline_cell_values}")
    
    # Get OCV values for comparison
    ocv_cell_values = prob.get_val('ocv_cell', units='V')
    print(f"OCV values (V): {ocv_cell_values}")
    
    # Get C-rate values
    c_rate_values = prob.get_val('c_rate', units='1/h')
    print(f"C-rate values (1/h): {c_rate_values}")
    
    # Get cell current values
    i_cell_values = prob.get_val('i_cell', units='A')
    print(f"Cell current values (A): {i_cell_values}")
    
    print("="*50)
    
    # Create plots showing deterioration over time
    print("\nCreating deterioration plots...")
    
    # Create time array for x-axis
    time_steps = np.arange(num_nodes)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: SOC deterioration
    ax1.plot(time_steps, soc_values, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('State of Charge')
    ax1.set_title('SOC Deterioration Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Open Circuit Voltage deterioration
    ax2.plot(time_steps, ocv_cell_values, 'g-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Open Circuit Voltage (V)')
    ax2.set_title('OCV Deterioration Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Line Voltage deterioration
    ax3.plot(time_steps, vline_cell_values, 'r-^', linewidth=2, markersize=6)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Line Voltage (V)')
    ax3.set_title('Line Voltage Deterioration Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: C-rate and current
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(time_steps, c_rate_values, 'purple', linewidth=2, label='C-rate')
    line2 = ax4_twin.plot(time_steps, i_cell_values, 'orange', linewidth=2, label='Cell Current')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('C-rate (1/h)', color='purple')
    ax4_twin.set_ylabel('Cell Current (A)', color='orange')
    ax4.set_title('C-rate and Current Over Time')
    ax4.grid(True, alpha=0.3)
    
    # Add legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Create a combined plot showing voltage comparison
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(time_steps, ocv_cell_values, 'g-s', linewidth=2, markersize=6, label='Open Circuit Voltage')
    ax.plot(time_steps, vline_cell_values, 'r-^', linewidth=2, markersize=6, label='Line Voltage')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Voltage Comparison: OCV vs Line Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Initial SOC: {soc_values[0]:.4f}")
    print(f"Final SOC: {soc_values[-1]:.4f}")
    print(f"SOC Change: {soc_values[0] - soc_values[-1]:.4f}")
    print(f"Initial OCV: {ocv_cell_values[0]:.2f} V")
    print(f"Final OCV: {ocv_cell_values[-1]:.2f} V")
    print(f"OCV Change: {ocv_cell_values[0] - ocv_cell_values[-1]:.2f} V")
    print(f"Initial Line Voltage: {vline_cell_values[0]:.2f} V")
    print(f"Final Line Voltage: {vline_cell_values[-1]:.2f} V")
    print(f"Line Voltage Change: {vline_cell_values[0] - vline_cell_values[-1]:.2f} V")
    print(f"Average C-rate: {np.mean(c_rate_values):.2f} 1/h")
    print(f"Average Current: {np.mean(i_cell_values):.2f} A")
    print("="*50)

    


    # Test battery interpolation accuracy
    #test_battery_interpolation_accuracy()


