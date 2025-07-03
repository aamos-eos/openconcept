from openconcept.propulsion import SimpleMotor, PowerSplitNacelle, SimpleGenerator, SimpleTurboshaft, SimplePropeller
from openconcept.energy_storage import SimpleBattery, SOCBattery
from openconcept.utilities import DVLabel, AddSubtractComp, ElementMultiplyDivideComp
from openconcept.propulsion import  EmpiricalPropeller,  DischargeEmpiricalBattery, ChargeEmpiricalBattery
from openconcept.propulsion.motor_empirical_rbf import EmpiricalMotor
from openconcept.propulsion.turbo_empirical import TurboFuelFlowRBF
from openconcept.propulsion.nacelle_splitter import PowerSplitNacelle
from openmdao.api import Group, BalanceComp, IndepVarComp, n2, ExplicitComponent
import numpy as np
from openconcept.propulsion.battery_data import BatteryData
from openconcept.propulsion.gearbox_analytical import Gearbox, PlanetaryGearbox, FlippedPlanetaryGearbox

# Flipped planetary gearbox computes input motor RPM and torque from output (carrier) RPM, output power, turbine RPM, and gear ratios
# Planetary gearbox computes output carrier RPM and torque from input motor and turbine RPM & torque and respective gear ratios

# Load data immediately when module is imported - this ensures it's available for all components
BatteryData.load_data(bat_filename='openconcept/propulsion/empirical_data/inHouse_battery_1motorConfig_208s27p_4grp_to_each_nacelle.xlsx', 
                      cell_sheetname='BOL_cell_fct_CRate', 
                      config_sheetname='battery_config')




class ParallelHybridNacelle(Group):

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")
        self.options.declare("num_em_per_nac", default=2, desc="Number of motors per nacelle")
        self.options.declare("rule", default="fraction", desc="Rule for power split between turbo and electric motor.\
                             'fraction' means that the power split is a fraction of the total power.\
                             'fixed' means that the power split is derived from a fixed amount of power.")
        self.options.declare("bias", default="gt", desc="Determines which component the power split rule is with respect to")
        self.options.declare("prop_thrust_set", default=False, desc="Solve for prop thrust from thrust input")
        self.options.declare("nac_power_set", default=False, desc="Solve for power from power input")
        self.options.declare("prop_rpm_set", default=False, desc="Solve for torque from RPM input")
        self.options.declare("nac_throttle_set", default=False, desc="Solve for throttle from throttle input")
        self.options.declare("motor_gb_efficiency", default=1, desc="Efficiency of motor reduction gearbox")
        self.options.declare("turbine_gb_efficiency", default=1, desc="Efficiency of turbine reduction gearbox")
        self.options.declare("motor_prop_gb_efficiency", default=0.95, desc="Efficiency of motor to propeller planetary gearbox")
        self.options.declare("turbine_prop_gb_efficiency", default=0.95, desc="Efficiency of turbine to propeller planetary gearbox")
        self.options.declare("gt_throttle_set", default=False, desc="gas turbine takes throttle as input. Otherwise, power is input")
        self.options.declare("motor_torque_rpm_set", default=False, desc="motor takes torque and rpm as input. Otherwise, power is input")

    def setup(self):
        nn = self.options["num_nodes"]
        nm = self.options["num_em_per_nac"]
        rule = self.options["rule"]
        bias = self.options["bias"]
        prop_thrust_set = self.options["prop_thrust_set"]
        nac_power_set = self.options["nac_power_set"]
        prop_rpm_set = self.options["prop_rpm_set"]
        nac_throttle_set = self.options["nac_throttle_set"]
        motor_gb_eta = self.options["motor_gb_efficiency"]
        turbine_gb_eta = self.options["turbine_gb_efficiency"]
        motor_prop_gb_eta = self.options["motor_prop_gb_efficiency"]
        turbine_prop_gb_eta = self.options["turbine_prop_gb_efficiency"]
        gt_throttle_set = self.options["gt_throttle_set"]
        motor_torque_rpm_set = self.options["motor_torque_rpm_set"]


        if nac_throttle_set:
            nacelle_throttle_to_power = ElementMultiplyDivideComp()
            nacelle_throttle_to_power.add_equation(output_name="mech_power_out", 
                                    input_names=["throttle", "max_rated_power"], 
                                    vec_size=[nn,1],
                                    input_units=[None, "W"],
                                    divide=[False, False])
            self.add_subsystem("nacelle_throttle_to_power", nacelle_throttle_to_power, promotes_inputs=["*"], promotes_outputs=["*"])        # end
        # define design variables that are independent of flight condition or control states
        



        if not prop_thrust_set and not nac_power_set and not nac_throttle_set:
            # If individual motor and turbine component powers are specified
            add_shaft_powers = AddSubtractComp(units="W")
            add_shaft_powers.add_equation(output_name="total_mech_power_out", 
                                    input_names=["unit_mech_power_in_em", 
                                                "unit_mech_power_in_gt"], 
                                    vec_size=nn,
                                    scaling_factors=[nm * motor_gb_eta * motor_prop_gb_eta, 1 * turbine_gb_eta * turbine_prop_gb_eta])
            self.add_subsystem("add_shaft_powers", add_shaft_powers, promotes_inputs=["*"], promotes_outputs=["*"])
        else:
            self.add_subsystem("hybrid_split", PowerSplitNacelle(rule=rule, num_nodes=nn, bias=bias, num_em_per_nac=nm), promotes_inputs=["*"], promotes_outputs=["*"])
        # end

        for i in range(nm):
            self.add_subsystem(f"motor{i+1}", EmpiricalMotor(num_nodes=nn, torque_rpm_set=motor_torque_rpm_set), promotes_inputs=[])
        # end

            

        # If individual motor and turbine component powers are determined
        scale_mech_power_cmd = ElementMultiplyDivideComp()
        scale_mech_power_cmd.add_equation(output_name="prop_power_in", 
                                input_names=["mech_power_out","carrier_efficiency"], 
                                vec_size=[nn,1],
                                input_units=["W", None],
                                divide=[False, False])
        self.add_subsystem("scale_mech_power_cmd", scale_mech_power_cmd, promotes_inputs=["*"], promotes_outputs=["*"])



        # Gas Turbine Gearbox
        """
        self.add_subsystem('turbo_gearbox', SimpleGearbox(num_nodes=nn), promotes=[])



        # Planetary Gearbox for turbine and motor combination
        if prop_thrust_set:
            self.add_subsystem('planetary_gearbox', FlippedPlanetaryGearbox(num_nodes=nn, n_motors=nm), promotes=['motor_gear_ratio','turbine_gear_ratio'])
        elif nac_power_set:
            # Power is input. Solve for efficiency
            self.add_subsystem('planetary_gearbox', PlanetaryGearbox(num_nodes=nn, n_motors=nm), promotes=['motor_gear_ratio','turbine_gear_ratio'])
        # end
        """

        # If nacelle throttle is not set, then 

        self.add_subsystem("turb", TurboFuelFlowRBF(num_nodes=nn, throttle_set=gt_throttle_set), promotes_outputs=[], promotes_inputs=["fltcond|*"])

        self.add_subsystem("prop", EmpiricalPropeller(num_nodes=nn,
                                                      thrust_set = prop_thrust_set,
                                                      power_set = nac_power_set,
                                                      rpm_set = prop_rpm_set, 
                                                      use_dynamic_data = True), promotes_inputs=["fltcond|*"])

        self.connect("prop_power_in", "prop.power")

        # Organize power split between turbo and electric motor for each nacelle
        #self.connect("prop.shaft_power_in", "hybrid_split.power_in")
        # end



class QuadParallelHybridElectricPropulsionSystem(Group):
    """
    This is an example model of a parallel-hybrid propulsion system. Four motors
    draw electrical load from a battery pack. The control inputs are the motor throttle setting.

    Fuel flows and prop thrust should be fairly accurate. Heat constraints haven't yet been incorporated.

    The "pilot" controls thrust by varying the motor throttles from 0 to 100+% of rated power. She may also vary the percentage of battery versus fuel being used
    by varying the power_split_fraction

    This module alone cannot produce accurate fuel flows, battery loads, etc. You must do the following, either with an implicit solver or with the optimizer:
    - Set eng1.throttle such that gen1.elec_power_out = hybrid_split.power_out_A

    The battery does not track its own state of charge (SOC); it is connected to elec_load simply so that the discharge rate can be compared to the discharge rate capability of the battery.
    SOC and fuel flows should be time-integrated at a higher level (in the mission analysis codes)

    Arrows show flow of information. In openConcept, mechanical power operates on a 'push' basis, while electrical load operates on a 'pull' basis. We reconcile these flows across an implicit gap by driving a residual to 0 using a solver.

    .. code::

        eng1.throttle                                                           hybrid_split.power_split_fraction           motor1.throttle
            ||                                                                                   ||                             ||
        eng1 --shaft_power_out--> gen1 --elec_power_out--> {IMPLICIT GAP} <--power_out_B         ||           <--elec_load-- motor1 --shaft_power_out --> prop1 -->thrust
            ||                                                                             hybrid_split <--elec_load  ++
            ||                                            batt1.elec_load <--power_out_A                       <--elec_load-- motor2 --shaft_power_out --> prop2 -->thrust
            V                                                                   V                                              ||
        fuel_flow (integrate over time)                                   elec_load (integrate over time to obtain SOC)       motor2.throttle

    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")
        self.options.declare("num_props", default=4, desc="Number of props")
        self.options.declare("num_em_per_nac", default=2, desc="Number of motors per nacelle")
        self.options.declare("rule", default="fraction", desc="Rule for power split between turbo and electric motor.\
                             'fraction' means that the power split is a fraction of the total power.\
                             'fixed' means that the power split is derived from a fixed amount of power.")
        self.options.declare("bias", default="gt", desc="Determines which component the power split rule is with respect to")
        self.options.declare("mode", default="mode_in", desc="Mode for motor input. 'torque_speed_in' means that the motor input is torque and speed. 'power_in' means that the motor input is power.")
        self.options.declare("nac_throttle_set", default=False, desc="Solve for power from throttle input")
        self.options.declare("nac_power_set", default=False, desc="Solve for power from power input")
        self.options.declare("prop_rpm_set", default=False, desc="Solve for power from rpm input")
        self.options.declare("prop_thrust_set", default=False, desc="Solve for power from thrust input")
        self.options.declare("gt_throttle_set", default=False, desc="gas turbine takes throttle as input. Otherwise, power is input")
        self.options.declare("motor_torque_rpm_set", default=False, desc="motor takes torque and rpm as input. Otherwise, power is input")

    def setup(self):
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        nm = self.options["num_em_per_nac"]
        rule = self.options["rule"]
        bias = self.options["bias"]
        nac_throttle_set = self.options["nac_throttle_set"]
        nac_power_set = self.options["nac_power_set"]
        prop_rpm_set = self.options["prop_rpm_set"]
        prop_thrust_set = self.options["prop_thrust_set"]
        gt_throttle_set = self.options["gt_throttle_set"]
        motor_torque_rpm_set = self.options["motor_torque_rpm_set"]



        dvlist = [
            ["ac|propulsion|engine|rating", "eng_rating", 745.0, "kW"],
            ["ac|propulsion|propeller|diameter", "prop_diameter", 3.9, "m"],
            ["ac|propulsion|motor|rating", "motor_rating", 1000.0, "kW"],
            ["ac|propulsion|generator|rating", "gen_rating", 250.0, "kW"],
            ["ac|weights|W_battery", "batt_weight", 2000, "kg"],
            ["ac|propulsion|battery|specific_energy", "specific_energy", 300, "W*h/kg"],
            ["ac|propulsion|battery|n_str", "n_str", 4, None],
            ["ac|propulsion|battery|n_motor", "n_motor", 4, None],
            ["ac|propulsion|battery|rloop_motor_dc_in", "rloop_motor_dc_in", 0.001 * np.ones(nn), "ohm"],
            ["ac|propulsion|battery|eta_converter", "eta_converter", 0.95 * np.ones(nn), None],
            ["ac|propulsion|battery|cell_capacity", "cell_capacity", 29.97, "A*h"],
            ["ac|propulsion|battery|t_cell_init", "t_cell_init", 30, "K"],
            ["ac|propulsion|battery|m_cell", "m_cell", 0.346, "kg"],
            ["ac|propulsion|battery|cp_cell", "cp_cell", 900, "J/kg/K"],
            ["ac|propulsion|battery|q_cool_bat", "q_cool_bat", 25 * np.ones(nn), "kW"],
            ["ac|propulsion|battery|soc_init", "soc_init", 1, None],
            ["ac|propulsion|battery|i_cell_charge_lim", "i_cell_charge_lim", 100, "A"],
            ["ac|propulsion|battery|v_cell_charge_lim", "v_cell_charge_lim", 4.2, "V"],
            ["ac|propulsion|battery|n_series_per_str", "n_series_per_str", 1, None],
            ["ac|propulsion|battery|n_parallel_per_str", "n_parallel_per_str", 1, None],
            ["ac|propulsion|motor|gear_ratio", "motor_gear_ratio", 3, None],
            ["ac|propulsion|turbine|gear_ratio", "turbine_gear_ratio", 20, None],
            ["ac|propulsion|motor|torque_cmd", "torque_cmd", 5000 * np.ones(nn), "N*m"],
            ["ac|propulsion|motor|rpm", "motor_rpm", 1000 * np.ones(nn), "rpm"],
            ["ac|propulsion|carrier_gear_efficiency", "carrier_gear_efficiency", 0.95, None],
        ]
        
        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])
        # introduce model components

        for i in range(npp):
            self.add_subsystem(f"nacelle{i+1}", ParallelHybridNacelle(num_nodes=nn, 
                                                                      num_em_per_nac=nm, 
                                                                      rule=rule, 
                                                                      bias=bias, 
                                                                      nac_throttle_set=nac_throttle_set, 
                                                                      nac_power_set=nac_power_set, 
                                                                      prop_rpm_set=prop_rpm_set,
                                                                      prop_thrust_set=prop_thrust_set,
                                                                      gt_throttle_set=gt_throttle_set,
                                                                      motor_torque_rpm_set=motor_torque_rpm_set), 
                                                                      promotes_inputs=["fltcond|*",], 
                                                                      promotes_outputs=[])
        # end





        # propulsion models expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        """
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation("motor2throttle", input_names=["throttle", "propulsor_active"], vec_size=nn)
        self.add_subsystem("failedmotor", failedengine, promotes_inputs=["throttle", "propulsor_active"])

        self.connect("failedmotor.motor2throttle", "motor2.throttle")
        """



        # Combine Electrical Loads for Motor
        input_motor_names_str = [f"motor{i+1}_{j+1}_elec_load" for i in range(npp) for j in range(nm)]
        addpower = AddSubtractComp(
            output_name="motors_elec_load",
            input_names=input_motor_names_str,
            units="kW",
            vec_size=nn,
        )
        # Combine Thrust for Propellers
        input_prop_names_str = [f"prop{i+1}_thrust" for i in range(npp)]
        addpower.add_equation(
            output_name="thrust", input_names=input_prop_names_str, units="N", vec_size=nn
        )

        self.add_subsystem("add_power", subsys=addpower, promotes_outputs=["*"])

        self.add_subsystem(
            "batt1", DischargeEmpiricalBattery(num_nodes=nn, v_cutoff=2.5), promotes_inputs=['*']
        )
        input_motor_weight_names_str = []
        input_prop_weight_names_str = []
        input_turbine_weight_names_str = []

        input_motor_elec_power_names_str = []
        for i_p in range(npp):
            # Connect Electrical Loads for Motor
            for i_m in range(nm):
                self.connect(f"nacelle{i_p+1}.motor{i_m+1}.elec_power", f"add_power.motor{i_p+1}_{i_m+1}_elec_load")
                input_motor_weight_names_str.append(f"nacelle{i_p+1}.motor{i_m+1}_weight")
                input_motor_elec_power_names_str.append(f"nacelle{i_p+1}.motor{i_m+1}.elec_power")
                self.connect("batt1.v_motor", f"nacelle{i_p+1}.motor{i_m+1}.voltage")
                self.connect(f"nacelle{i_p+1}.unit_mech_power_in_em", f"nacelle{i_p+1}.motor{i_m+1}.mech_power")

                #self.connect(f"nacelle{i_p+1}.motor{i_m+1}.torque", f"nacelle{i_p+1}.planetary_gearbox.motor{i_m+1}_torque")
                #self.connect(f"nacelle1{i_p+1}.motor{i_m+1}.rpm", f"nacelle{i_p+1}.planetary_gearbox.motor{i_m+1}_rpm")

                

            # end
            #self.connect(f"nacelle{i_p+1}.prop.thrust", f"add_power.prop{i_p+1}_thrust")
            input_prop_weight_names_str.append(f"nacelle{i_p+1}.prop{i_p+1}_weight")
            input_turbine_weight_names_str.append(f"nacelle{i_p+1}.turb.weight")
            self.connect("prop_diameter", f"nacelle{i_p+1}.prop.diameter")
            self.connect("motor_rating", f"nacelle{i_p+1}.power_rating_em")
            self.connect("eng_rating", f"nacelle{i_p+1}.power_rating_gt")
            self.connect("carrier_gear_efficiency",f"nacelle{i_p+1}.carrier_efficiency")
            self.connect(f"nacelle{i_p+1}.prop.thrust_calc", f"add_power.prop{i_p+1}_thrust")
            #self.connect(f"nacelle1.{i}", f"nacelle{i_p+1}.unit_mech_power_in_gt")
            #self.connect(f"nacelle{i_p+1}|power_split_fraction", f"nacelle{i_p+1}.{power_split_fraction_name}")
            #self.connect("num_gt_per_nac", f"nacelle{i_p+1}.num_gt_per_nac")
            #self.connect("num_em_per_nac", f"nacelle{i_p+1}.num_em_per_nac")
        # end



        #self.connect("hybrid_split.power_out_B", "eng_nac_throttle_set.gt_power_required")
        #self.connect("nacelle.turb.elec_power_out", "eng_nac_throttle_set.gt_power_available")
        #self.connect("eng_nac_throttle_set.eng_throttle", "eng1.throttle")

        # Connect Weights
        addweights = AddSubtractComp(
            output_name="motors_weight", input_names=input_motor_weight_names_str, units="kg"
        )
        addweights.add_equation(
            output_name="propellers_weight", input_names=input_prop_weight_names_str, units="kg"
        )
        addweights.add_equation(
            output_name="turbine_weight", input_names=input_turbine_weight_names_str, units="kg"
        )

        # Sum Electrical power
        add_elec_power = AddSubtractComp()
        add_elec_power.add_equation(
            output_name="p_train_elec", input_names=input_motor_elec_power_names_str
        )


        #self.add_subsystem("add_weights", subsys=addweights, promotes_inputs=["*"], promotes_outputs=["*"])

        #relabel = [["hybrid_split_A_in", "battery_load", np.ones(nn) * 260.0, "kW"]]
        #self.add_subsystem("relabel", DVLabel(relabel), promotes_outputs=["battery_load"])
        #self.connect("hybrid_split.power_out_A", "relabel.hybrid_split_A_in")

        #self.connect("motor1.component_weight", "motor1_weight")
        #self.connect("motor2.component_weight", "motor2_weight")
        #self.connect("prop1.component_weight", "prop1_weight")
        #self.connect("prop2.component_weight", "prop2_weight")

        # connect design variables to model component inputs
        #self.connect("eng_rating", "eng1.shaft_power_rating")
        #self.connect("prop_diameter", ["nacelle1.prop.diameter", "nacelle2.prop.diameter", "nacelle3.prop.diameter", "nacelle4.prop.diameter"])
        #self.connect("motor_rating", ["motor1.elec_power_rating", "motor2.elec_power_rating"])
        #self.connect("motor_rating", ["prop1.power_rating", "prop2.power_rating"])
        #self.connect("gen_rating", "gen1.elec_power_rating")
        #self.connect("batt_weight", "batt1.battery_weight")

        #self.set_input_defaults("nacelle1.prop.diameter", val=2.5, units='m')
        #self.set_input_defaults("nacelle2.prop.diameter", val=2.5, units='m')
        #self.set_input_defaults("nacelle3.prop.diameter", val=2.5, units='m')
        #self.set_input_defaults("nacelle4.prop.diameter", val=2.5, units='m')


#!/usr/bin/env python3
"""
Test script for QuadParallelHybridElectricPropulsionSystem

This script demonstrates how to set up and run the quad parallel hybrid 
propulsion system with default parameters and various test conditions.
"""

import numpy as np
import openmdao.api as om
from openconcept.propulsion.systems.parallel_hybrid import QuadParallelHybridElectricPropulsionSystem

if __name__ == "__main__":
    print("=" * 60)
    print("Testing QuadParallelHybridElectricPropulsionSystem")
    print("=" * 60)
    
    # Create the problem
    prob = om.Problem()
    
    # Number of analysis points
    num_nodes = 5
    
    # Add IndepVarComp that promotes all its outputs
    ivc = IndepVarComp()
    
    # Flight conditions
    ivc.add_output('fltcond|rho', val=1.225 * np.ones(num_nodes), units='kg/m**3')
    ivc.add_output('fltcond|Utrue', val=100.0 * np.ones(num_nodes), units='m/s')
    ivc.add_output('fltcond|h', val=1000.0 * np.ones(num_nodes), units='m')
    ivc.add_output('fltcond|M', val=0.3 * np.ones(num_nodes), units=None)
    ivc.add_output('fltcond|T', val=288.15 * np.ones(num_nodes), units='K')
    ivc.add_output('fltcond|p', val=101325.0 * np.ones(num_nodes), units='Pa')
    ivc.add_output('fltcond|q', val=0.5 * 1.225 * 100.0**2 * np.ones(num_nodes), units='Pa')
    ivc.add_output('num_gt_per_nac', val=1, units=None)
    ivc.add_output('num_em_per_nac', val=2, units=None)
    
    # Control inputs
    ivc.add_output('throttle', val=0.8 * np.ones(num_nodes), units=None)
    ivc.add_output('propulsor_active', val=1.0 * np.ones(num_nodes), units=None)
    
    # Mission parameters
    ivc.add_output('duration', val=3600.0 , units='s')
    
    # Design variables
    ivc.add_output('ac|propulsion|engine|rating', val=745.0, units='kW')
    ivc.add_output('ac|propulsion|propeller|diameter', val=2.5, units='m')
    ivc.add_output('ac|propulsion|motor|rating', val=1000.0, units='kW')
    ivc.add_output('ac|propulsion|generator|rating', val=250.0, units='kW')
    ivc.add_output('ac|weights|W_battery', val=2000.0, units='kg')
    ivc.add_output('ac|propulsion|battery|specific_energy', val=300.0, units='W*h/kg')

    # Propeller Inputs
    ivc.add_output('diameter', val=4.4, units='m', desc='Propeller diameter')
    ivc.add_output('prop_rpm', val=1000.0 * np.ones(num_nodes), units='rpm', desc='RPM')

    # Turbine Inputs
    ivc.add_output('fltcond|disa', 0 * np.ones(num_nodes), desc='DISA in degrees Celsius')
    ivc.add_output('frac', 0.6 * np.ones(num_nodes), desc='Throttle fraction')

    # Motor Inputs
    ivc.add_output('ac|propulsion|motor|gear_ratio', 3.0, units=None, desc='Motor gear ratio')
    ivc.add_output('ac|propulsion|turbine|gear_ratio', 20.0, units=None, desc='Turbine gear ratio')

    ivc.add_output('ac|propulsion|motor|torque_cmd', 5000 * np.ones(num_nodes), units='N*m', desc='Commanded torque')
    ivc.add_output('ac|propulsion|motor|rpm', 1000 * np.ones(num_nodes), units='rpm', desc='Motor RPM')
    #ivc.add_output('voltage', 700 * np.ones(num_nodes), units='V', desc='Motor voltage')
    


    # Battery Inputs
    bat_data = BatteryData.get_data(bat_filename='openconcept/propulsion/empirical_data/inHouse_battery_1motorConfig_208s27p_4grp_to_each_nacelle.xlsx', 
                                    cell_sheetname='BOL_cell_fct_CRate', 
                                    config_sheetname='battery_config')
    ivc.add_output('ac|propulsion|battery|n_str', 4, desc='number of battery strings')
    ivc.add_output('ac|propulsion|battery|n_motor', 4, desc='Number of electric motors')
    ivc.add_output('ac|propulsion|battery|rloop_motor_dc_in', 0.001 * np.ones(num_nodes), units='ohm', desc='DC loop resistance to each motor')
    ivc.add_output('dtime', 1 * np.ones(num_nodes), units='s', desc='time step')
    ivc.add_output('ac|propulsion|battery|eta_converter', 0.95 * np.ones(num_nodes), units=None, desc='efficiency of the converter')

    ivc.add_output('p_aux_elec', 100 * np.ones(num_nodes), units='W', desc='Auxiliary electric load on LV side')
    ivc.add_output('rloop_aux_conv_hv_dc_in', 0.001 * np.ones(num_nodes), units='ohm', desc='DC loop resistance to LV converter')

    # Load up Battery Data
    ivc.add_output('ac|propulsion|battery|cell_capacity', bat_data.cell_Ah_capacity, units='A*h', desc='Cell capacity')
    ivc.add_output('ac|propulsion|battery|t_cell_init', 30, units='K', desc='Initial temperature of the cell')
    ivc.add_output('ac|propulsion|battery|m_cell', bat_data.m_cell, units='kg', desc='Mass of the cell')
    ivc.add_output('ac|propulsion|battery|cp_cell', bat_data.cp_cell, units='J/kg/K', desc='Specific heat capacity of the cell')
    ivc.add_output('ac|propulsion|battery|q_cool_bat', 25 * np.ones(num_nodes), units='kW', desc='Cooling power of the battery')
    ivc.add_output('ac|propulsion|battery|soc_init', bat_data.soc_init, units=None, desc='Initial state of charge of the battery')
    ivc.add_output('ac|propulsion|battery|i_cell_charge_lim', 700, units='A', desc='Constant current setpoint (pack level)')
    ivc.add_output('ac|propulsion|battery|v_cell_charge_lim', 860, units='V', desc='Constant voltage setpoint (pack level)')
    ivc.add_output('ac|propulsion|battery|n_series_per_str', bat_data.n_series_per_str, desc='Number of cells in series')
    ivc.add_output('ac|propulsion|battery|n_parallel_per_str', bat_data.n_parallel_per_str, desc='Number of cells in parallel')
    
    # Power split fraction
    ivc.add_output('nacelle1|power_split_fraction', val=0.5 * np.ones(num_nodes), units=None)
    ivc.add_output('nacelle2|power_split_fraction', val=0.5 * np.ones(num_nodes), units=None)
    ivc.add_output('nacelle3|power_split_fraction', val=0.5 * np.ones(num_nodes), units=None)
    ivc.add_output('nacelle4|power_split_fraction', val=0.5 * np.ones(num_nodes), units=None)

    # Propeller Inputs
    ivc.add_output('nacelle1|prop_rpm', val=1000.0 * np.ones(num_nodes), units='rpm', desc='Propeller RPM')
    ivc.add_output('nacelle2|prop_rpm', val=1000.0 * np.ones(num_nodes), units='rpm', desc='Propeller RPM')
    ivc.add_output('nacelle3|prop_rpm', val=1000.0 * np.ones(num_nodes), units='rpm', desc='Propeller RPM')
    ivc.add_output('nacelle4|prop_rpm', val=1000.0 * np.ones(num_nodes), units='rpm', desc='Propeller RPM')

    # Gearbox Efficiency
    ivc.add_output('ac|propulsion|carrier_gear_efficiency', 0.95, units=None, desc='Carrier gear in planetary gearbox efficiency')



    # Nacelle power
    p_max_nac_kW = 2700.0
    ivc.add_output('nacelle1|max_rated_power', val=p_max_nac_kW , units='kW')
    ivc.add_output('nacelle2|max_rated_power', val=p_max_nac_kW , units='kW')
    ivc.add_output('nacelle3|max_rated_power', val=p_max_nac_kW , units='kW')
    ivc.add_output('nacelle4|max_rated_power', val=p_max_nac_kW , units='kW')

    prob.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

    npp = 4 # num nodes in analysis (time indexes)
    nm = 2 # num motors per nacelle
    rule = "fraction"
    bias = "gt"
    nac_power_set = True # When true, power is an input. Thrust is an output
    prop_rpm_set = False
    prop_thrust_set = False # When true, thrust is an input. Power is an output
    nac_throttle_set = True # When true, nacelle power = max power * throttle. else, nacelle power out is defined. 
    gt_throttle_set = True
    motor_torque_rpm_set = False

    nac_throttle = np.ones((num_nodes,)) * 0.5
    gt_throttle = np.ones((num_nodes,)) * 0.5
    nac_mech_power__kW = p_max_nac_kW * nac_throttle

    if nac_power_set:
        if nac_throttle_set:
            ivc.add_output('nacelle1|throttle', nac_throttle, units=None)
            ivc.add_output('nacelle2|throttle', nac_throttle, units=None)
            ivc.add_output('nacelle3|throttle', nac_throttle, units=None)
            ivc.add_output('nacelle4|throttle', nac_throttle, units=None)
        else:
            ivc.add_output('nacelle1|total_mech_power_out', nac_mech_power__kW, units='kW')
            ivc.add_output('nacelle2|total_mech_power_out', nac_mech_power__kW, units='kW')
            ivc.add_output('nacelle3|total_mech_power_out', nac_mech_power__kW, units='kW')
            ivc.add_output('nacelle4|total_mech_power_out', nac_mech_power__kW, units='kW')
    # end

    ivc.add_output('nacelle1|gt_throttle', gt_throttle, units=None)
    ivc.add_output('nacelle2|gt_throttle', gt_throttle, units=None)
    ivc.add_output('nacelle3|gt_throttle', gt_throttle, units=None)
    ivc.add_output('nacelle4|gt_throttle', gt_throttle, units=None)



    

    # Add the propulsion system
    prob.model.add_subsystem(
        'propulsion',
        QuadParallelHybridElectricPropulsionSystem(num_nodes=num_nodes, 
                                                   num_props=npp, 
                                                   num_em_per_nac=nm, 
                                                   rule=rule, 
                                                   bias=bias, 
                                                   nac_power_set = nac_power_set,
                                                   prop_thrust_set = prop_thrust_set,
                                                   nac_throttle_set = nac_throttle_set,
                                                   prop_rpm_set = prop_rpm_set,
                                                   gt_throttle_set = gt_throttle_set,
                                                   motor_torque_rpm_set = motor_torque_rpm_set,
                                                   ),
        promotes_inputs=['*'],
        promotes_outputs=['*']
    )
    
    # Set up the problem
    for i in range(npp):
        for j in range(nm):
            #prob.model.connect("motor_rpm", [f"nacelle{i+1}.motor{j+1}.rpm", f"nacelle{i+1}.planetary_gearbox.motor{j+1}_rpm"])
            #prob.model.connect("torque_cmd", f"nacelle{i+1}.motor{j+1}.torque_cmd")
            pass
        # end
        prob.model.connect(f"nacelle{i+1}|power_split_fraction", f"nacelle{i+1}.power_split_fraction_gt")

        if nac_throttle_set:
            prob.model.connect(f"nacelle{i+1}|throttle", f"nacelle{i+1}.throttle")
        else:
            prob.model.connect(f"nacelle{i+1}|total_mech_power_out", f"nacelle{i+1}.total_mech_power_out")
        # end
        prob.model.connect(f"nacelle{i+1}|max_rated_power", f"nacelle{i+1}.max_rated_power")
        prob.model.connect(f"nacelle{i+1}|prop_rpm", f"nacelle{i+1}.prop.rpm")
        prob.model.connect(f"nacelle{i+1}|gt_throttle", f"nacelle{i+1}.turb.throttle")
    # end

    prob.setup()

    n2(prob)
    
    print("Problem setup complete!")
    
    # Run the model
    print("\nRunning the model...")
    prob.run_model()
    print("Model executed successfully!")
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # System outputs
    print(f"\nTotal Thrust: {prob.get_val('thrust', units='N')}")
    print(f"Fuel Flow: {prob.get_val('fuel_flow', units='kg/s')}")
    print(f"Motors Electrical Load: {prob.get_val('motors_elec_load', units='kW')}")
    
    # Component weights
    print(f"\nComponent Weights:")
    print(f"  Motors Weight: {prob.get_val('motors_weight', units='kg')}")
    print(f"  Propellers Weight: {prob.get_val('propellers_weight', units='kg')}")
    print(f"  Turbine Weight: {prob.get_val('turbine_weight', units='kg')}")
    
    # Individual nacelle outputs
    print(f"\nIndividual Nacelle Outputs:")
    for i in range(4):
        nacelle_num = i + 1
        print(f"\nNacelle {nacelle_num}:")
        print(f"  Motor 1 Elec Load: {prob.get_val(f'nacelle{nacelle_num}.motor1.elec_load', units='kW')}")
        print(f"  Motor 2 Elec Load: {prob.get_val(f'nacelle{nacelle_num}.motor2.elec_load', units='kW')}")
        print(f"  Propeller Thrust: {prob.get_val(f'nacelle{nacelle_num}.prop.thrust', units='N')}")
        print(f"  Turboshaft Power: {prob.get_val(f'nacelle{nacelle_num}.turb.shaft_power_out', units='kW')}")
    
    # Battery state
    print(f"\nBattery State:")
    print(f"  SOC: {prob.get_val('batt1.SOC')}")
    print(f"  Heat Output: {prob.get_val('batt1.heat_out', units='W')}")
    
    # Power split details
    print(f"\nPower Split Details (Nacelle 1):")
    print(f"  Power In: {prob.get_val('nacelle1.hybrid_split.power_in', units='kW')}")
    print(f"  Power Out A (Battery): {prob.get_val('nacelle1.hybrid_split.power_out_A', units='kW')}")
    print(f"  Power Out B (Generator): {prob.get_val('nacelle1.hybrid_split.power_out_B', units='kW')}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

