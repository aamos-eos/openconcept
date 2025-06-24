from openconcept.propulsion import SimpleMotor, PowerSplit, SimpleGenerator, SimpleTurboshaft, SimplePropeller
from openconcept.energy_storage import SimpleBattery, SOCBattery
from openconcept.utilities import DVLabel, AddSubtractComp, ElementMultiplyDivideComp

from openmdao.api import Group, BalanceComp
import numpy as np



class ParallelHybridNacelle(Group):

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")
        self.options.declare("num_motors_per_nacelle", default=2, desc="Number of motors per nacelle")
        self.options.declare("hybrid_split_rule", default="fraction", desc="Rule for power split between turbo and electric motor.\
                             'fraction' means that the power split is a fraction of the total power.\
                             'fixed' means that the power split is derived from a fixed amount of power.")

    def setup(self):
        nn = self.options["num_nodes"]
        nm = self.options["num_motors_per_nacelle"]
        rule = self.options["hybrid_split_rule"]

        # define design variables that are independent of flight condition or control states
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

        for i in range(nm):
            self.add_subsystem(f"motor{i+1}", SimpleMotor(efficiency=0.97, num_nodes=nn), promotes_inputs=["throttle"])
        # end
        self.add_subsystem("turb", SimpleTurboshaft(num_nodes=nn), promotes_inputs=["fltcond|*", "shaft_power_in"])
        self.add_subsystem("prop", SimplePropeller(num_nodes=nn), promotes_inputs=["fltcond|*"])


        # Organize power split between turbo and electric motor for each nacelle
        self.add_subsystem("hybrid_split", PowerSplit(rule=rule, num_nodes=nn))
        self.connect("prop.shaft_power_in", "hybrid_split.power_in")
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
        self.options.declare("num_motors_per_nacelle", default=2, desc="Number of motors per nacelle")

    def setup(self):
        nn = self.options["num_nodes"]
        np = self.options["num_props"]
        nm = self.options["num_motors_per_nacelle"]



        # propulsion models expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation("motor2throttle", input_names=["throttle", "propulsor_active"], vec_size=nn)
        self.add_subsystem("failedmotor", failedengine, promotes_inputs=["throttle", "propulsor_active"])

        self.connect("failedmotor.motor2throttle", "motor2.throttle")



        # Combine Electrical Loads for Motor
        input_motor_names_str = [f"motor{i+1}_elec_load" for i in range(np)]
        addpower = AddSubtractComp(
            output_name="motors_elec_load",
            input_names=input_motor_names_str,
            units="kW",
            vec_size=nn,
        )
        # Combine Thrust for Propellers
        input_prop_names_str = [f"prop{i+1}_thrust" for i in range(np)]
        addpower.add_equation(
            output_name="thrust", input_names=input_prop_names_str, units="N", vec_size=nn
        )

        self.add_subsystem("add_power", subsys=addpower, promotes_outputs=["*"])

        for i_p in range(np):
            # Connect Electrical Loads for Motor
            for i_m in range(nm):
                self.connect(f"nacelle{i_p+1}.motor{i_m+1}_elec_load", f"add_power.motor{i_m+1}_elec_load")
            # end
            self.connect(f"prop{i_p+1}.thrust", f"add_power.prop{i_p+1}_thrust")
        # end

        self.add_subsystem("hybrid_split", PowerSplit(rule="fraction", num_nodes=nn))
        self.connect("motors_elec_load", "hybrid_split.power_in")



        self.add_subsystem(
            "batt1", SOCBattery(num_nodes=nn, efficiency=0.97), promotes_inputs=["duration", "specific_energy"]
        )
        self.connect("hybrid_split.power_out_A", "batt1.elec_load")
        self.add_subsystem(
            "eng_throttle_set",
            BalanceComp(
                name="eng_throttle",
                val=np.ones((nn,)) * 0.5,
                units=None,
                eq_units="kW",
                rhs_name="gen_power_required",
                lhs_name="gen_power_available",
            ),
        )
        self.connect("hybrid_split.power_out_B", "eng_throttle_set.gen_power_required")
        self.connect("gen1.elec_power_out", "eng_throttle_set.gen_power_available")
        self.connect("eng_throttle_set.eng_throttle", "eng1.throttle")

        # Connect Weights
        input_motor_weight_names_str = [f"nacelle{i+1}.motor{i+1}_weight" for i in range(np)]
        input_prop_weight_names_str = [f"nacelle{i+1}.prop{i+1}_weight" for i in range(np)]
        addweights = AddSubtractComp(
            output_name="motors_weight", input_names=input_motor_weight_names_str, units="kg"
        )
        addweights.add_equation(
            output_name="propellers_weight", input_names=input_prop_weight_names_str, units="kg"
        )
        self.add_subsystem("add_weights", subsys=addweights, promotes_inputs=["*"], promotes_outputs=["*"])

        relabel = [["hybrid_split_A_in", "battery_load", np.ones(nn) * 260.0, "kW"]]
        self.add_subsystem("relabel", DVLabel(relabel), promotes_outputs=["battery_load"])
        self.connect("hybrid_split.power_out_A", "relabel.hybrid_split_A_in")

        self.connect("motor1.component_weight", "motor1_weight")
        self.connect("motor2.component_weight", "motor2_weight")
        self.connect("prop1.component_weight", "prop1_weight")
        self.connect("prop2.component_weight", "prop2_weight")

        # connect design variables to model component inputs
        self.connect("eng_rating", "eng1.shaft_power_rating")
        self.connect("prop_diameter", ["prop1.diameter", "prop2.diameter"])
        self.connect("motor_rating", ["motor1.elec_power_rating", "motor2.elec_power_rating"])
        self.connect("motor_rating", ["prop1.power_rating", "prop2.power_rating"])
        self.connect("gen_rating", "gen1.elec_power_rating")
        self.connect("batt_weight", "batt1.battery_weight")