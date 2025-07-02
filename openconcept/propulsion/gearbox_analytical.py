import numpy as np
from openmdao.api import ExplicitComponent, Group


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
        self.add_output('carrier_rpm', val=np.ones(nn) * 2000.0, units='rpm',
                       desc='Carrier (output shaft) RPM')
        self.add_output('carrier_torque', val=np.ones(nn) * 200.0, units='N*m',
                       desc='Carrier (output shaft) torque')
        
        # Declare partials
        self.declare_partials('*', '*', method='exact')
    
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
        outputs['carrier_rpm'] = output_rpm
        outputs['carrier_torque'] = output_torque
    
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
            # ∂(carrier_rpm)/∂(motor_rpm) = alpha / (n_motors * motor_gear_ratio)
            partials['carrier_rpm', f'motor{i+1}_rpm'] = alpha / (n_motors * motor_gear_ratio) * np.eye(nn)
            
            # ∂(carrier_rpm)/∂(motor_torque) = 0 (torque doesn't affect RPM in this model)
            partials['carrier_rpm', f'motor{i+1}_torque'] = np.eye(nn)
            
            # ∂(carrier_torque)/∂(motor_rpm) and ∂(carrier_torque)/∂(motor_torque) are complex
            # For now, we'll use simplified partials
            partials['carrier_torque', f'motor{i+1}_rpm'] = np.eye(nn) * 0
            partials['carrier_torque', f'motor{i+1}_torque'] = np.eye(nn) * 0
        
        # Gear ratio partials
        # ∂(carrier_rpm)/∂(motor_gear_ratio) = -alpha * combined_motor_rpm / (motor_gear_ratio^2)
        partials['carrier_rpm', 'motor_gear_ratio'] = -alpha * combined_motor_rpm / (motor_gear_ratio ** 2)
        
        # ∂(carrier_rpm)/∂(turbine_gear_ratio) = -(1-alpha) * turbine_rpm / (turbine_gear_ratio^2)
        partials['carrier_rpm', 'turbine_gear_ratio'] = -(1.0 - alpha) * turbine_rpm / (turbine_gear_ratio ** 2)
        
        # Simplified torque partials for gear ratios
        partials['carrier_torque', 'motor_gear_ratio'] = np.zeros(nn) 
        partials['carrier_torque', 'turbine_gear_ratio'] = np.zeros(nn) 
        
        # Turbine partials
        # ∂(carrier_rpm)/∂(turbine_rpm) = (1-alpha) / turbine_gear_ratio
        partials['carrier_rpm', 'turbine_rpm'] = (1.0 - alpha) / turbine_gear_ratio * np.eye(nn)
        
        # ∂(carrier_rpm)/∂(turbine_torque) = 0
        partials['carrier_rpm', 'turbine_torque'] = np.eye(nn) * 0
        
        # Simplified torque partials for turbine
        partials['carrier_torque', 'turbine_rpm'] = np.eye(nn) * 0
        partials['carrier_torque', 'turbine_torque'] = np.eye(nn) * 0


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
        
        # Declare partials
        self.declare_partials('*', '*', method='exact')
    
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        efficiency = self.options['efficiency']
        
        rpm_in = inputs['shaft_in_rpm']
        torque_in = inputs['shaft_in_torque']
        gear_ratio = inputs['gear_ratio']
        
        # Output RPM (inversely proportional - gear ratio reduces RPM)
        outputs['shaft_out_rpm'] = rpm_in / gear_ratio
        
        # Output torque (directly proportional - gear ratio increases torque, accounting for efficiency)
        outputs['shaft_out_torque'] = torque_in * gear_ratio * efficiency
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        efficiency = self.options['efficiency']
        
        rpm_in = inputs['shaft_in_rpm']
        torque_in = inputs['shaft_in_torque']
        gear_ratio = inputs['gear_ratio']
        
        # RPM partials
        partials['shaft_out_rpm', 'shaft_in_rpm'] = (1.0 / gear_ratio) * np.eye(nn)
        partials['shaft_out_rpm', 'gear_ratio'] = -rpm_in / (gear_ratio ** 2)
        
        # Torque partials
        partials['shaft_out_torque', 'shaft_in_torque'] = (gear_ratio * efficiency) * np.eye(nn)
        partials['shaft_out_torque', 'gear_ratio'] = torque_in * efficiency


class CombinedGearboxGroup(Group):
    """
    A group that combines both simple and planetary gearbox components for testing.
    
    This group demonstrates how multiple gearbox types can be used together
    in a larger propulsion system model.
    
    Parameters
    ----------
    num_nodes : int
        Number of analysis points (default: 1)
    n_motors : int
        Number of motor input shafts for planetary gearbox (default: 2)
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('n_motors', default=2, desc='Number of motor input shafts for planetary gearbox')
        self.options.declare('output_set', default=False, desc='True when output (carrier) RPM, output power, turbine RPM, and gear ratios are set')

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        n_motors = self.options['n_motors']
        output_set = self.options['output_set']
        
        # Add the simple gearbox
        self.add_subsystem('simple_gearbox', SimpleGearbox(num_nodes=num_nodes), promotes=['*'])
        
        if output_set:
            # Add the flipped planetary gearbox
            self.add_subsystem('planetary_gearbox', FlippedPlanetaryGearbox(num_nodes=num_nodes, n_motors=n_motors), promotes=['*'])
        else:
            # Add the planetary gearbox
            self.add_subsystem('planetary_gearbox', PlanetaryGearbox(num_nodes=num_nodes, n_motors=n_motors), promotes=['*'])
        # end 


class FlippedPlanetaryGearbox(ExplicitComponent):
    """
    Flipped planetary gearbox: Given output (carrier) RPM, output power, turbine RPM, and gear ratios,
    solve for required input motor RPM and torques, and turbine torque.
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

        # Inputs
        self.add_input('carrier_rpm', shape=(nn,), units='rpm', desc='Output shaft (carrier) RPM')
        self.add_input('output_power', shape=(nn,), units='W', desc='Output shaft power (W)')
        self.add_input('turbine_rpm', shape=(nn,), units='rpm', desc='Turbine RPM')
        self.add_input('motor_gear_ratio', units=None, desc='Motor gear ratio')
        self.add_input('turbine_gear_ratio', units=None, desc='Turbine gear ratio')

        # Outputs (per motor)
        for i in range(n_motors):
            self.add_output(f'motor{i+1}_rpm', shape=(nn,), units='rpm', desc=f'Motor {i+1} RPM')
            self.add_output(f'motor{i+1}_torque', shape=(nn,), units='N*m', desc=f'Motor {i+1} torque')
        self.add_output('turbine_torque', shape=(nn,), units='N*m', desc='Turbine torque')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        n_motors = self.options['n_motors']
        motor_gb_eff = self.options['motor_gb_efficiency']
        turbine_gb_eff = self.options['turbine_gb_efficiency']
        N_sun = self.options['N_sun']
        N_ring = self.options['N_ring']

        carrier_rpm = inputs['carrier_rpm']
        output_power = inputs['output_power']
        turbine_rpm = inputs['turbine_rpm']
        motor_gear_ratio = inputs['motor_gear_ratio']
        turbine_gear_ratio = inputs['turbine_gear_ratio']

        # Planetary gear blending factor
        alpha = 1.0 / (1.0 + N_ring / N_sun)

        # Turbine reduced RPM
        turbine_reduced_rpm = turbine_rpm / turbine_gear_ratio

        # Solve for required reduced motor RPM
        # carrier_rpm = alpha * (motor_rpm / motor_gear_ratio) + (1-alpha) * turbine_reduced_rpm
        # => motor_rpm = ((carrier_rpm - (1-alpha)*turbine_reduced_rpm)/alpha) * motor_gear_ratio
        motor_rpm_reduced = (carrier_rpm - (1.0 - alpha) * turbine_reduced_rpm) / alpha
        motor_rpm = motor_rpm_reduced * motor_gear_ratio

        # Output torque from output power and carrier RPM
        omega_c = carrier_rpm * 2.0 * np.pi / 60.0
        carrier_torque = np.where(omega_c > 1e-8, output_power / omega_c, 0.0)

        # Power split: output power = motor_power + turbine_power
        # Let x = fraction of power from motors, (1-x) from turbine
        # We'll use the same alpha as the blending factor for power split
        # (This is a simplification; for more accurate modeling, solve for actual power split)
        motor_power = alpha * output_power
        turbine_power = (1.0 - alpha) * output_power

        # Each motor shares power equally
        motor_power_each = motor_power / n_motors
        # Each motor's reduced RPM
        omega_m_reduced = motor_rpm_reduced * 2.0 * np.pi / 60.0
        # Each motor's torque at output (after gear)
        motor_torque_out = np.where(omega_m_reduced > 1e-8, motor_power_each / omega_m_reduced, 0.0)
        # Input torque before gear and efficiency
        motor_torque_in = motor_torque_out / (motor_gear_ratio * motor_gb_eff)

        # Turbine torque at output (after gear)
        omega_t_reduced = turbine_reduced_rpm * 2.0 * np.pi / 60.0
        turbine_torque_out = np.where(omega_t_reduced > 1e-8, turbine_power / omega_t_reduced, 0.0)
        # Input torque before gear and efficiency
        turbine_torque_in = turbine_torque_out / (turbine_gear_ratio * turbine_gb_eff)

        # Set outputs
        for i in range(n_motors):
            outputs[f'motor{i+1}_rpm'] = motor_rpm
            outputs[f'motor{i+1}_torque'] = motor_torque_in
        outputs['turbine_torque'] = turbine_torque_in


def test_gearbox_components():
    """
    Test both gearbox components in a combined group
    """
    import openmdao.api as om
    
    print("Testing combined gearbox group...")
    
    num_nodes = 5
    
    # Set up the OpenMDAO model
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables for simple gearbox
    ivc.add_output('shaft_in_rpm', 3000 * np.ones(num_nodes), units='rpm', desc='Input shaft RPM')
    ivc.add_output('shaft_in_torque', 200 * np.ones(num_nodes), units='N*m', desc='Input shaft torque')
    ivc.add_output('gear_ratio', 2.5, units=None, desc='Gear ratio')
    
    # Add independent variables for planetary gearbox
    ivc.add_output('motor1_rpm', 5000 * np.ones(num_nodes), units='rpm', desc='Motor 1 RPM')
    ivc.add_output('motor1_torque', 150 * np.ones(num_nodes), units='N*m', desc='Motor 1 torque')
    ivc.add_output('motor2_rpm', 4800 * np.ones(num_nodes), units='rpm', desc='Motor 2 RPM')
    ivc.add_output('motor2_torque', 120 * np.ones(num_nodes), units='N*m', desc='Motor 2 torque')
    ivc.add_output('motor_gear_ratio', 3.0, units=None, desc='Motor gear ratio')
    ivc.add_output('turbine_rpm', 45000 * np.ones(num_nodes), units='rpm', desc='Turbine RPM')
    ivc.add_output('turbine_torque', 80 * np.ones(num_nodes), units='N*m', desc='Turbine torque')
    ivc.add_output('turbine_gear_ratio', 20.0, units=None, desc='Turbine gear ratio')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('gearbox_group', CombinedGearboxGroup(num_nodes=num_nodes, n_motors=2), promotes=['*'])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    #om.n2(prob)
    prob.run_model()
    
    # Get results for simple gearbox
    simple_shaft_out_rpm = prob.get_val('shaft_out_rpm', units='rpm')
    simple_shaft_out_torque = prob.get_val('shaft_out_torque', units='N*m')
    
    print(f"Combined Gearbox Group Results for {num_nodes} nodes:")
    print(f"\nSimple Gearbox:")
    print(f"  Input RPM: {prob.get_val('shaft_in_rpm', units='rpm')}")
    print(f"  Input Torque: {prob.get_val('shaft_in_torque', units='N*m')}")
    print(f"  Gear Ratio: {prob.get_val('gear_ratio')}")
    print(f"  Output RPM: {simple_shaft_out_rpm}")
    print(f"  Output Torque: {simple_shaft_out_torque}")
    
    print(f"\nPlanetary Gearbox:")
    print(f"  Motor 1 RPM: {prob.get_val('motor1_rpm', units='rpm')}")
    print(f"  Motor 1 Torque: {prob.get_val('motor1_torque', units='N*m')}")
    print(f"  Motor 2 RPM: {prob.get_val('motor2_rpm', units='rpm')}")
    print(f"  Motor 2 Torque: {prob.get_val('motor2_torque', units='N*m')}")
    print(f"  Motor Gear Ratio: {prob.get_val('motor_gear_ratio')}")
    print(f"  Turbine RPM: {prob.get_val('turbine_rpm', units='rpm')}")
    print(f"  Turbine Torque: {prob.get_val('turbine_torque', units='N*m')}")
    print(f"  Turbine Gear Ratio: {prob.get_val('turbine_gear_ratio')}")
    print(f"  Carrier RPM: {prob.get_val('carrier_rpm', units='rpm')}")
    print(f"  Carrier Torque: {prob.get_val('carrier_torque', units='N*m')}")
    
    # Check partials for the entire group
    print("\nChecking combined gearbox group partials...")
    prob.check_partials(compact_print=True)
    
    return (simple_shaft_out_rpm, simple_shaft_out_torque)


if __name__ == "__main__":
    test_gearbox_components()
