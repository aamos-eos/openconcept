from openconcept.propulsion import SimpleMotor, PowerSplitNacelle, SimpleGenerator, SimpleTurboshaft, SimplePropeller
from openconcept.energy_storage import SimpleBattery, SOCBattery
from openconcept.utilities import DVLabel, AddSubtractComp, ElementMultiplyDivideComp
from openconcept.propulsion import EmpiricalMotor, EmpiricalPropeller, EmpiricalDynamicTurbo, EmpiricalStaticTurbo, DischargeEmpiricalBattery, ChargeEmpiricalBattery



from openmdao.api import Group, BalanceComp, IndepVarComp, n2
import numpy as np
from openconcept.propulsion.battery_data import BatteryData

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

    def setup(self):
        nn = self.options["num_nodes"]
        nm = self.options["num_em_per_nac"]
        rule = self.options["rule"]
        bias = self.options["bias"]

        # define design variables that are independent of flight condition or control states
        


        for i in range(nm):
            self.add_subsystem(f"motor{i+1}", EmpiricalMotor(num_nodes=nn), promotes_inputs=["throttle"])
        # end
        self.add_subsystem("turb", EmpiricalDynamicTurbo(num_nodes=nn), promotes_outputs=["fuel_flow"])
        self.add_subsystem("prop", EmpiricalPropeller(num_nodes=nn,
                                                      known_thrust = True,
                                                      known_power = False,
                                                      known_rpm = True, 
                                                      use_dynamic_data = True), promotes_inputs=["fltcond|*"])


        # Organize power split between turbo and electric motor for each nacelle
        self.add_subsystem("hybrid_split", PowerSplitNacelle(rule=rule, num_nodes=nn, bias=bias, num_em_per_nac=nm), promotes_inputs=["*"], promotes_outputs=["*"])
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

    def setup(self):
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        nm = self.options["num_em_per_nac"]
        rule = self.options["rule"]
        bias = self.options["bias"]



        dvlist = [
            ["ac|propulsion|engine|rating", "eng_rating", 745.0, "kW"],
            ["ac|propulsion|propeller|diameter", "prop_diameter", 3.9, "m"],
            ["ac|propulsion|motor|rating", "motor_rating", 1000.0, "kW"],
            ["ac|propulsion|generator|rating", "gen_rating", 250.0, "kW"],
            ["ac|weights|W_battery", "batt_weight", 2000, "kg"],
            ["ac|propulsion|battery|specific_energy", "specific_energy", 300, "W*h/kg"],
        ]
        
        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])
        # introduce model components

        for i in range(npp):
            self.add_subsystem(f"nacelle{i+1}", ParallelHybridNacelle(num_nodes=nn, num_em_per_nac=nm, rule=rule, bias=bias), promotes_inputs=["fltcond|*"], promotes_outputs=[])
        # end





        # propulsion models expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        """
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation("motor2throttle", input_names=["throttle", "propulsor_active"], vec_size=nn)
        self.add_subsystem("failedmotor", failedengine, promotes_inputs=["throttle", "propulsor_active"])

        self.connect("failedmotor.motor2throttle", "motor2.throttle")
        """



        # Combine Electrical Loads for Motor
        input_motor_names_str = [f"motor{i+1}_elec_load" for i in range(npp)]
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

        input_motor_weight_names_str = []
        input_prop_weight_names_str = []
        input_turbine_weight_names_str = []
        for i_p in range(npp):
            # Connect Electrical Loads for Motor
            for i_m in range(nm):
                #self.connect(f"nacelle{i_p+1}.motor{i_m+1}_elec_load", f"add_power.motor{i_m+1}_elec_load")
                input_motor_weight_names_str.append(f"nacelle{i_p+1}.motor{i_m+1}_weight")
            # end
            self.connect(f"nacelle{i_p+1}.prop.thrust", f"add_power.prop{i_p+1}_thrust")
            input_prop_weight_names_str.append(f"nacelle{i_p+1}.prop{i_p+1}_weight")
            input_turbine_weight_names_str.append(f"nacelle{i_p+1}.turb.weight")
            self.connect("prop_diameter", f"nacelle{i_p+1}.prop.diameter")
            self.connect("motor_rating", f"nacelle{i_p+1}.power_rating_em")
            self.connect("eng_rating", f"nacelle{i_p+1}.power_rating_gt")
            #self.connect(f"nacelle{i_p+1}|power_split_fraction", f"nacelle{i_p+1}.{power_split_fraction_name}")
            #self.connect("num_gt_per_nac", f"nacelle{i_p+1}.num_gt_per_nac")
            #self.connect("num_em_per_nac", f"nacelle{i_p+1}.num_em_per_nac")
        # end


        self.add_subsystem(
            "batt1", SOCBattery(num_nodes=nn, efficiency=0.97), promotes_inputs=["duration", "specific_energy"]
        )
        self.connect("motors_elec_load", "batt1.elec_load")
        self.add_subsystem(
            "eng_throttle_set",
            BalanceComp(
                name="eng_throttle",
                val=np.ones((nn,)) * 0.5,
                units=None,
                eq_units="kW",
                rhs_name="gt_power_required",
                lhs_name="gt_power_available",
            ),
        )
        #self.connect("hybrid_split.power_out_B", "eng_throttle_set.gt_power_required")
        #self.connect("nacelle.turb.elec_power_out", "eng_throttle_set.gt_power_available")
        #self.connect("eng_throttle_set.eng_throttle", "eng1.throttle")

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
    ivc.add_output('rpm', val=1000.0 * np.ones(num_nodes), units='rpm', desc='RPM')

    # Turbine Inputs
    ivc.add_output('fltcond|h', 10000*0.3048 * np.ones(num_nodes), units='m', desc='Altitude in meters')
    ivc.add_output('disa', 0 * np.ones(num_nodes), desc='DISA in degrees Celsius')
    ivc.add_output('frac', 0.6 * np.ones(num_nodes), desc='Throttle fraction')

    # Motor Inputs
    ivc.add_output('voltage', 700 * np.ones(num_nodes), units=None, desc='Motor voltage')
    ivc.add_output('power_W', 125 * 1000 * np.ones(num_nodes), units='W', desc='Motor power')

    # Battery Inputs
    bat_data = BatteryData.get_data()

    ivc.add_output('n_strngs', 4, desc='number of battery strings')
    ivc.add_output('p_ptrain_elec', 100 * np.ones(num_nodes), units='W', desc='Total propulsion electric power demand')
    ivc.add_output('n_motors', 4, desc='Number of electric motors')
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
    
    # Power split fraction
    ivc.add_output('nacelle1|power_split_fraction', val=0.5 * np.ones(num_nodes), units=None)
    ivc.add_output('nacelle2|power_split_fraction', val=0.5 * np.ones(num_nodes), units=None)
    ivc.add_output('nacelle3|power_split_fraction', val=0.5 * np.ones(num_nodes), units=None)
    ivc.add_output('nacelle4|power_split_fraction', val=0.5 * np.ones(num_nodes), units=None)

    prob.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

    npp = 4
    nm = 2
    rule = "fraction"
    bias = "gt"


    

    # Add the propulsion system
    prob.model.add_subsystem(
        'propulsion',
        QuadParallelHybridElectricPropulsionSystem(num_nodes=num_nodes, num_props=npp, num_em_per_nac=nm, rule=rule, bias=bias),
        promotes_inputs=['*'],
        promotes_outputs=['*']
    )
    
    # Set up the problem
    for i in range(npp):
        for j in range(nm):
            pass
        # end
        prob.model.connect(f"nacelle{i+1}|power_split_fraction", f"nacelle{i+1}.power_split_fraction_gt")
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

