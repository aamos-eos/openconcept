import numpy as np
from openmdao.api import ExplicitComponent


class PlanetaryGearbox(ExplicitComponent):
    """
    A planetary gearbox component that handles multiple motor inputs and one turbine input.
    
    This component implements a planetary gear system where:
    - Multiple motors are combined with individual gear ratios
    - One turbine has its own gear ratio
    - The motor and turbine inputs are blended using planetary gear physics
    - Output is a single shaft with combined characteristics
    
    Parameters
    ----------
    num_nodes : int
        Number of analysis points (default: 1)
    n_motors : int
        Number of motor input shafts (default: 2)
    motor_gb_efficiency : float
        Motor gear efficiency (default: 0.95)
    turbine_gb_efficiency : float
        Turbine gear efficiency (default: 0.90)
    N_sun : int
        Number of sun gear teeth (default: 30)
    N_ring : int
        Number of ring gear teeth (default: 90)
    
    Inputs
    ------
    motor{i}_rpm : array_like
        Motor {i} RPM [rpm]
    motor{i}_torque : array_like
        Motor {i} torque [N*m]
    motor_gear_ratio : array_like
        Gear ratio for motor reduction (dimensionless)
    turbine_rpm : array_like
        Turbine RPM [rpm]
    turbine_torque : array_like
        Turbine torque [N*m]
    turbine_gear_ratio : array_like
        Gear ratio for turbine reduction (dimensionless)
    
    Outputs
    -------
    shaft_out_rpm : array_like
        Output shaft RPM [rpm]
    shaft_out_torque : array_like
        Output shaft torque [N*m]
    power_loss : array_like
        Power lost in the gearbox [W]
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('n_motors', default=2, desc='Number of motor input shafts')
        self.options.declare('motor_gb_efficiency', default=0.95, desc='Motor gear efficiency')
        self.options.declare('turbine_gb_efficiency', default=0.90, desc='Turbine gear efficiency')
        self.options.declare('N_sun', default=30, desc='Number of sun gear teeth')
        self.options.declare('N_ring', default=90, desc='Number of ring gear teeth')
    
    def setup(self):
        nn = self.options['num_nodes']
        n_motors = self.options['n_motors']
        
        # Motor input variables
        for i in range(n_motors):
            self.add_input(f'motor{i+1}_rpm', val=np.ones(nn) * 5000.0, units='rpm',
                          desc=f'Motor {i+1} RPM')
            self.add_input(f'motor{i+1}_torque', val=np.ones(nn) * 100.0, units='N*m',
                          desc=f'Motor {i+1} torque')
        
        # Motor gear ratio (same for all motors)
        self.add_input('motor_gear_ratio', val=3.0, units=None,
                      desc='Gear ratio for motor reduction')
        
        # Turbine input variables
        self.add_input('turbine_rpm', val=np.ones(nn) * 45000.0, units='rpm',
                      desc='Turbine RPM')
        self.add_input('turbine_torque', val=np.ones(nn) * 50.0, units='N*m',
                      desc='Turbine torque')
        self.add_input('turbine_gear_ratio', val=20.0, units=None,
                      desc='Gear ratio for turbine reduction')
        
        # Output shaft variables
        self.add_output('shaft_out_rpm', val=np.ones(nn) * 2000.0, units='rpm',
                       desc='Output shaft RPM')
        self.add_output('shaft_out_torque', val=np.ones(nn) * 200.0, units='N*m',
                       desc='Output shaft torque')
        self.add_output('power_loss', val=np.zeros(nn), units='W',
                       desc='Power lost in the gearbox')
        
        # Declare partials
        for i in range(n_motors):
            # RPM partials
            self.declare_partials('shaft_out_rpm', f'motor{i+1}_rpm',
                                rows=np.arange(nn), cols=np.arange(nn))
            self.declare_partials('shaft_out_rpm', f'motor{i+1}_torque',
                                rows=np.arange(nn), cols=np.arange(nn))
            
            # Torque partials
            self.declare_partials('shaft_out_torque', f'motor{i+1}_rpm',
                                rows=np.arange(nn), cols=np.arange(nn))
            self.declare_partials('shaft_out_torque', f'motor{i+1}_torque',
                                rows=np.arange(nn), cols=np.arange(nn))
        
        # Gear ratio partials
        self.declare_partials('shaft_out_rpm', ['motor_gear_ratio', 'turbine_gear_ratio'],
                            rows=np.arange(nn), cols=np.zeros(nn, dtype=int))
        self.declare_partials('shaft_out_torque', ['motor_gear_ratio', 'turbine_gear_ratio'],
                            rows=np.arange(nn), cols=np.zeros(nn, dtype=int))
        
        # Turbine partials
        self.declare_partials('shaft_out_rpm', ['turbine_rpm', 'turbine_torque'],
                            rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('shaft_out_torque', ['turbine_rpm', 'turbine_torque'],
                            rows=np.arange(nn), cols=np.arange(nn))
        
        # Power loss partials
        for i in range(n_motors):
            self.declare_partials('power_loss', [f'motor{i+1}_rpm', f'motor{i+1}_torque'],
                                rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('power_loss', ['turbine_rpm', 'turbine_torque'],
                            rows=np.arange(nn), cols=np.arange(nn))
    
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        n_motors = self.options['n_motors']
        motor_gb_eff = self.options['motor_gb_efficiency']
        turbine_gb_eff = self.options['turbine_gb_efficiency']
        N_sun = self.options['N_sun']
        N_ring = self.options['N_ring']
        
        # Calculate planetary gear blending factor
        alpha = 1.0 / (1.0 + N_ring / N_sun)
        
        # Get gear ratios
        motor_gear_ratio = inputs['motor_gear_ratio']
        turbine_gear_ratio = inputs['turbine_gear_ratio']
        
        # Combine motor inputs (average RPM, sum torque)
        motor_rpm_sum = 0.0
        motor_torque_sum = 0.0
        
        for i in range(n_motors):
            motor_rpm = inputs[f'motor{i+1}_rpm']
            motor_torque = inputs[f'motor{i+1}_torque']
            motor_rpm_sum += motor_rpm
            motor_torque_sum += motor_torque
        
        # Average motor RPM and sum motor torque
        combined_motor_rpm = motor_rpm_sum / n_motors
        combined_motor_torque = motor_torque_sum
        
        # Reduce motor RPM and increase motor torque
        combined_motor_rpm_reduced = combined_motor_rpm / motor_gear_ratio
        combined_motor_torque_increased = combined_motor_torque * motor_gear_ratio * motor_gb_eff
        
        # Reduce turbine RPM and increase turbine torque
        turbine_rpm = inputs['turbine_rpm']
        turbine_torque = inputs['turbine_torque']
        turbine_reduced_rpm = turbine_rpm / turbine_gear_ratio
        turbine_reduced_torque = turbine_torque * turbine_gear_ratio * turbine_gb_eff
        
        # Planetary gear blending
        output_rpm = alpha * combined_motor_rpm_reduced + (1.0 - alpha) * turbine_reduced_rpm
        
        # Calculate powers
        motor_power = combined_motor_torque_increased * combined_motor_rpm_reduced * 2.0 * np.pi / 60.0
        turbine_power = turbine_reduced_torque * turbine_reduced_rpm * 2.0 * np.pi / 60.0
        total_power = motor_power + turbine_power
        
        # Calculate output torque from total power
        # Avoid division by zero
        output_torque = np.where(output_rpm > 0,
                                total_power * 60.0 / (2.0 * np.pi * output_rpm),
                                np.zeros_like(output_rpm))
        
        # Set outputs
        outputs['shaft_out_rpm'] = output_rpm
        outputs['shaft_out_torque'] = output_torque
        
        # Calculate power loss
        total_input_power = 0.0
        for i in range(n_motors):
            motor_rpm = inputs[f'motor{i+1}_rpm']
            motor_torque = inputs[f'motor{i+1}_torque']
            total_input_power += motor_rpm * motor_torque * 2.0 * np.pi / 60.0
        
        turbine_input_power = turbine_rpm * turbine_torque * 2.0 * np.pi / 60.0
        total_input_power += turbine_input_power
        
        outputs['power_loss'] = total_input_power - total_power
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        n_motors = self.options['n_motors']
        motor_gb_eff = self.options['motor_gb_efficiency']
        turbine_gb_eff = self.options['turbine_gb_efficiency']
        N_sun = self.options['N_sun']
        N_ring = self.options['N_ring']
        
        # Calculate planetary gear blending factor
        alpha = 1.0 / (1.0 + N_ring / N_sun)
        
        # Get gear ratios
        motor_gear_ratio = inputs['motor_gear_ratio']
        turbine_gear_ratio = inputs['turbine_gear_ratio']
        
        # Calculate intermediate values
        motor_rpm_sum = 0.0
        motor_torque_sum = 0.0
        
        for i in range(n_motors):
            motor_rpm = inputs[f'motor{i+1}_rpm']
            motor_torque = inputs[f'motor{i+1}_torque']
            motor_rpm_sum += motor_rpm
            motor_torque_sum += motor_torque
        
        combined_motor_rpm = motor_rpm_sum / n_motors
        combined_motor_torque = motor_torque_sum
        
        combined_motor_rpm_reduced = combined_motor_rpm / motor_gear_ratio
        combined_motor_torque_increased = combined_motor_torque * motor_gear_ratio * motor_gb_eff
        
        turbine_rpm = inputs['turbine_rpm']
        turbine_torque = inputs['turbine_torque']
        turbine_reduced_rpm = turbine_rpm / turbine_gear_ratio
        turbine_reduced_torque = turbine_torque * turbine_gear_ratio * turbine_gb_eff
        
        output_rpm = alpha * combined_motor_rpm_reduced + (1.0 - alpha) * turbine_reduced_rpm
        
        motor_power = combined_motor_torque_increased * combined_motor_rpm_reduced * 2.0 * np.pi / 60.0
        turbine_power = turbine_reduced_torque * turbine_reduced_rpm * 2.0 * np.pi / 60.0
        total_power = motor_power + turbine_power
        
        # Calculate partials for each motor
        for i in range(n_motors):
            # ∂(output_rpm)/∂(motor_rpm) = alpha / (n_motors * motor_gear_ratio)
            partials['shaft_out_rpm', f'motor{i+1}_rpm'] = alpha / (n_motors * motor_gear_ratio) * np.ones(nn)
            
            # ∂(output_rpm)/∂(motor_torque) = 0 (torque doesn't affect RPM in this model)
            partials['shaft_out_rpm', f'motor{i+1}_torque'] = np.zeros(nn)
            
            # ∂(output_torque)/∂(motor_rpm) and ∂(output_torque)/∂(motor_torque) are complex
            # For now, we'll use simplified partials
            partials['shaft_out_torque', f'motor{i+1}_rpm'] = np.zeros(nn)
            partials['shaft_out_torque', f'motor{i+1}_torque'] = np.zeros(nn)
        
        # Gear ratio partials
        # ∂(output_rpm)/∂(motor_gear_ratio) = -alpha * combined_motor_rpm / (motor_gear_ratio^2)
        partials['shaft_out_rpm', 'motor_gear_ratio'] = -alpha * combined_motor_rpm / (motor_gear_ratio ** 2)
        
        # ∂(output_rpm)/∂(turbine_gear_ratio) = -(1-alpha) * turbine_rpm / (turbine_gear_ratio^2)
        partials['shaft_out_rpm', 'turbine_gear_ratio'] = -(1.0 - alpha) * turbine_rpm / (turbine_gear_ratio ** 2)
        
        # Simplified torque partials for gear ratios
        partials['shaft_out_torque', 'motor_gear_ratio'] = np.zeros(nn)
        partials['shaft_out_torque', 'turbine_gear_ratio'] = np.zeros(nn)
        
        # Turbine partials
        # ∂(output_rpm)/∂(turbine_rpm) = (1-alpha) / turbine_gear_ratio
        partials['shaft_out_rpm', 'turbine_rpm'] = (1.0 - alpha) / turbine_gear_ratio * np.ones(nn)
        
        # ∂(output_rpm)/∂(turbine_torque) = 0
        partials['shaft_out_rpm', 'turbine_torque'] = np.zeros(nn)
        
        # Simplified torque partials for turbine
        partials['shaft_out_torque', 'turbine_rpm'] = np.zeros(nn)
        partials['shaft_out_torque', 'turbine_torque'] = np.zeros(nn)
        
        # Power loss partials
        for i in range(n_motors):
            motor_rpm = inputs[f'motor{i+1}_rpm']
            motor_torque = inputs[f'motor{i+1}_torque']
            
            # ∂(power_loss)/∂(motor_rpm) = 2π * motor_torque / 60
            partials['power_loss', f'motor{i+1}_rpm'] = 2.0 * np.pi * motor_torque / 60.0
            
            # ∂(power_loss)/∂(motor_torque) = 2π * motor_rpm / 60
            partials['power_loss', f'motor{i+1}_torque'] = 2.0 * np.pi * motor_rpm / 60.0
        
        # Turbine power loss partials
        partials['power_loss', 'turbine_rpm'] = 2.0 * np.pi * turbine_torque / 60.0
        partials['power_loss', 'turbine_torque'] = 2.0 * np.pi * turbine_rpm / 60.0


class SimpleGearbox(ExplicitComponent):
    """
    A simplified gearbox component for single input/single output cases.
    
    This is a simpler version of the Gearbox component for cases where you only
    need one input shaft and one output shaft.
    
    Parameters
    ----------
    num_nodes : int
        Number of analysis points (default: 1)
    efficiency : float
        Gearbox efficiency (default: 0.98)
    
    Inputs
    ------
    shaft_in_rpm : array_like
        Input shaft RPM [rpm]
    shaft_in_torque : array_like
        Input shaft torque [N*m]
    gear_ratio : array_like
        Gear ratio (dimensionless)
    
    Outputs
    -------
    shaft_out_rpm : array_like
        Output shaft RPM [rpm]
    shaft_out_torque : array_like
        Output shaft torque [N*m]
    power_loss : array_like
        Power lost in the gearbox [W]
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('efficiency', default=0.98, desc='Gearbox efficiency')
    
    def setup(self):
        nn = self.options['num_nodes']
        
        self.add_input('shaft_in_rpm', val=np.ones(nn) * 1000.0, units='rpm',
                      desc='Input shaft RPM')
        self.add_input('shaft_in_torque', val=np.ones(nn) * 100.0, units='N*m',
                      desc='Input shaft torque')
        self.add_input('gear_ratio', val=2.0, units=None,
                      desc='Gear ratio (output/input)')
        
        self.add_output('shaft_out_rpm', val=np.ones(nn) * 2000.0, units='rpm',
                       desc='Output shaft RPM')
        self.add_output('shaft_out_torque', val=np.ones(nn) * 50.0, units='N*m',
                       desc='Output shaft torque')
        self.add_output('power_loss', val=np.zeros(nn), units='W',
                       desc='Power lost in the gearbox')
        
        # Declare partials
        self.declare_partials('shaft_out_rpm', ['shaft_in_rpm', 'gear_ratio'],
                            rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('shaft_out_torque', ['shaft_in_torque', 'gear_ratio'],
                            rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('power_loss', ['shaft_in_rpm', 'shaft_in_torque'],
                            rows=np.arange(nn), cols=np.arange(nn))
    
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        efficiency = self.options['efficiency']
        
        rpm_in = inputs['shaft_in_rpm']
        torque_in = inputs['shaft_in_torque']
        gear_ratio = inputs['gear_ratio']
        
        # Output RPM (directly proportional)
        outputs['shaft_out_rpm'] = rpm_in * gear_ratio
        
        # Output torque (inversely proportional, accounting for efficiency)
        outputs['shaft_out_torque'] = torque_in * efficiency / gear_ratio
        
        # Power loss
        power_in = 2.0 * np.pi * rpm_in * torque_in / 60.0
        outputs['power_loss'] = power_in * (1.0 - efficiency)
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        efficiency = self.options['efficiency']
        
        rpm_in = inputs['shaft_in_rpm']
        torque_in = inputs['shaft_in_torque']
        gear_ratio = inputs['gear_ratio']
        
        # RPM partials
        partials['shaft_out_rpm', 'shaft_in_rpm'] = gear_ratio * np.ones(nn)
        partials['shaft_out_rpm', 'gear_ratio'] = rpm_in
        
        # Torque partials
        partials['shaft_out_torque', 'shaft_in_torque'] = (efficiency / gear_ratio) * np.ones(nn)
        partials['shaft_out_torque', 'gear_ratio'] = -torque_in * efficiency / (gear_ratio ** 2)
        
        # Power loss partials
        partials['power_loss', 'shaft_in_rpm'] = (1.0 - efficiency) * 2.0 * np.pi * torque_in / 60.0
        partials['power_loss', 'shaft_in_torque'] = (1.0 - efficiency) * 2.0 * np.pi * rpm_in / 60.0
