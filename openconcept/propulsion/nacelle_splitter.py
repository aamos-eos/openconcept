import numpy as np
from openmdao.api import ExplicitComponent


class PowerSplitNacelle(ExplicitComponent):
    """
    A power split mechanism between the turbo and electric motor within the nacelle

    Inputs
    ------
    mech_power_out : float
        Total power fed to the nacelle. (vector, W)
    power_rating_gt : float
        Maximum rated power of the gas turbine. (scalar, W)
    power_rating_em : float
        Maximum rated power of the electric motor. (scalar, W)
    num_gt_per_nac : int
        Number of gas turbines in the nacelle. (scalar, int)
    num_em_per_nac : int
        Number of electric motors in the nacelle. (scalar, int)
    power_split_fraction_gt:
        If ``'rule'`` is set to ``'fraction'``, sets percentage of input power directed
        to gas turbine (minus losses). (vector, dimensionless)
    power_split_fraction_em:
        If ``'rule'`` is set to ``'fraction'``, sets percentage of input power directed
        to electric motor (minus losses). (vector, dimensionless)
    power_split_amount_gt:
        If ``'rule'`` is set to ``'fixed'``, sets amount of input power to gas turbine (minus
        losses). (vector, W)
    power_split_amount_em:
        If ``'rule'`` is set to ``'fixed'``, sets amount of input power to electric motor (minus
        losses). (vector, W)

    Outputs
    -------
    power_out_A : float
        Power sent to first output (vector, W)
    power_out_B : float
        Power sent to second output (vector, W)
    heat_out : float
        Waste heat produced (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_weight : float
        Weight of the component (scalar, kg)
    component_sizing_margin : float
        Equal to 1 when fed full rated power (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    power_split_rule : str
        Power split control rule to use; either ``'fixed'`` where a set
        amount of power is sent to Output A or ``'fraction'`` where a
        fraction of the total power is sent to Output A.
    power_split_bias : str
        Determines which component the power split rule is with respect to.
        Either ``'gt'`` for gas turbine or ``'em'`` for electric motor.
    efficiency : float
        Component efficiency (default 1)
    weight_inc : float
        Weight per unit rated power
        (default 0, kg/W)
    weight_base : float
        Base weight
        (default 0, kg)
    cost_inc : float
        Nonrecurring cost per unit power
        (default 0, USD/W)
    cost_base : float
        Base cost
        (default 0 USD)
    """

    def initialize(self):
        # define control rules
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("num_em_per_nac", default=1, desc="Number of electric motors in the nacelle")
        self.options.declare("rule", default="fraction", desc="Control strategy - fraction or fixed power")
        self.options.declare("bias", default="gt", desc="Determines which component the power split rule is with respect to")

        self.options.declare("prop_gt_gb_efficiency", default=1.0, desc="Transmission efficiency from gas turbine to propeller (dimensionless)")
        self.options.declare("prop_em_gb_efficiency", default=1.0, desc="Transmission efficiency from electric motor to propeller (dimensionless)")
        self.options.declare("weight_inc", default=0.0, desc="kg per input watt")
        self.options.declare("weight_base", default=0.0, desc="kg base weight")
        self.options.declare("cost_inc", default=0.0, desc="$ cost per input watt")
        self.options.declare("cost_base", default=0.0, desc="$ cost base")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("mech_power_out", units="W", desc="Total mechanical power produced by the nacelle", shape=(nn,))
        self.add_input("power_rating_gt", val=99999999, units="W", desc="Gas turbine power rating")
        self.add_input("power_rating_em", val=99999999, units="W", desc="Electric motor power rating")

        rule = self.options["rule"]
        bias = self.options["bias"]
        if rule == "fraction":
            if bias == "gt":
                self.add_input("power_split_fraction_gt", val=0.5, desc="Fraction of power from gas turbine(s)", shape=(nn,))
            elif bias == "em":
                self.add_input("power_split_fraction_em", val=0.5, desc="Fraction of power from electric motor(s)", shape=(nn,))
            else:
                msg = 'Specify either "gt" or "em" as power split bias'
                raise ValueError(msg)
        elif rule == "fixed":
            if bias == "gt":
                self.add_input("power_split_amount_gt", units="W", desc="Raw amount of power from gas turbine(s)", shape=(nn,))
            elif bias == "em":
                self.add_input("power_split_amount_em", units="W", desc="Raw amount of power from electric motor(s)", shape=(nn,))
            else:
                msg = 'Specify either "gt" or "em" as power split bias'
                raise ValueError(msg)
        else:
            msg = 'Specify either "fraction" or "fixed" as power split control rule between turbo and electric motor'
            raise ValueError(msg)

        weight_inc = self.options["weight_inc"]
        cost_inc = self.options["cost_inc"]

        self.add_output("unit_mech_power_in_gt", units="W", desc="Unit output power from gas turbine(s) shaft after gearbox", shape=(nn,))
        self.add_output("unit_mech_power_in_em", units="W", desc="Unit output power from electric motor(s) shaft after gearbox", shape=(nn,))
        self.add_output("component_cost_gt", units="USD", desc="Gas turbine component cost")
        self.add_output("component_cost_em", units="USD", desc="Electric motor component cost")
        self.add_output("component_weight_gt", units="kg", desc="Gas turbine component weight")
        self.add_output("component_weight_em", units="kg", desc="Electric motor component weight")
        self.add_output("component_sizing_margin_gt", desc="Fraction of rated power", shape=(nn,))
        self.add_output("component_sizing_margin_em", desc="Fraction of rated power", shape=(nn,))

        if rule == "fraction":
            if bias == "gt":
                self.declare_partials(
                    ["unit_mech_power_in_gt", "unit_mech_power_in_em"], ["mech_power_out", "power_split_fraction_gt"], rows=range(nn), cols=range(nn)
                )
            elif bias == "em":
                self.declare_partials(
                    ["unit_mech_power_in_gt", "unit_mech_power_in_em"], ["mech_power_out", "power_split_fraction_em"], rows=range(nn), cols=range(nn)
                )
            # end
        elif rule == "fixed":
            if bias == "gt":
                self.declare_partials(
                    ["unit_mech_power_in_gt", "unit_mech_power_in_em"], ["mech_power_out", "power_split_amount_gt"], rows=range(nn), cols=range(nn)
                )
            elif bias == "em":
                self.declare_partials(
                    ["unit_mech_power_in_gt", "unit_mech_power_in_em"], ["mech_power_out", "power_split_amount_em"], rows=range(nn), cols=range(nn)
                )
            # end

        self.declare_partials("component_cost_gt", "power_rating_gt", val=cost_inc)
        self.declare_partials("component_cost_em", "power_rating_em", val=cost_inc)
        self.declare_partials("component_weight_gt", "power_rating_gt", val=weight_inc)
        self.declare_partials("component_weight_em", "power_rating_em", val=weight_inc)
        self.declare_partials("component_sizing_margin_gt", "mech_power_out", rows=range(nn), cols=range(nn))
        self.declare_partials("component_sizing_margin_em", "mech_power_out", rows=range(nn), cols=range(nn))
        self.declare_partials("component_sizing_margin_gt", "power_rating_gt")
        self.declare_partials("component_sizing_margin_em", "power_rating_em")

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        nm = self.options["num_em_per_nac"]
        rule = self.options["rule"]
        bias = self.options["bias"]
        weight_inc = self.options["weight_inc"]
        weight_base = self.options["weight_base"]
        cost_inc = self.options["cost_inc"]
        cost_base = self.options["cost_base"]
        eta_gt_prop = self.options["prop_gt_gb_efficiency"]
        eta_em_prop = self.options["prop_em_gb_efficiency"]

        if rule == "fraction":
            if bias == "gt":
                unit_mech_power_in_gt = inputs["mech_power_out"] * inputs["power_split_fraction_gt"]  / eta_gt_prop
                unit_mech_power_in_em = inputs["mech_power_out"] * (1 - inputs["power_split_fraction_gt"]) / nm / eta_em_prop
            elif bias == "em":
                unit_mech_power_in_gt = inputs["mech_power_out"] * (1 - inputs["power_split_fraction_em"]) / eta_gt_prop
                unit_mech_power_in_em = inputs["mech_power_out"] * inputs["power_split_fraction_em"] / nm / eta_em_prop
            else:
                msg = 'Specify either "gt" or "em" as power split bias'
                raise ValueError(msg)


        elif rule == "fixed":

            if bias == "gt":
                # Case where power requested from gas turbine is greater than total power needed from propeller
                # if inputs['mech_power_out'] < inputs['power_split_amount']:
                over_supply_idx = np.where(inputs["mech_power_out"] < inputs["power_split_amount"])
                unit_mech_power_in_gt = np.zeros(nn)
                unit_mech_power_in_em = np.zeros(nn)
                unit_mech_power_in_gt[over_supply_idx] = inputs["mech_power_out"][over_supply_idx] / eta_gt_prop
                unit_mech_power_in_em[over_supply_idx] = np.zeros(nn)[over_supply_idx]

                # Case where power requested from gas turbine is less than total power needed from propeller
                # if inputs['mech_power_out'] >= inputs['power_split_amount']:
                partial_supply_idx = np.where(inputs["mech_power_out"] >= inputs["power_split_amount"])
                unit_mech_power_in_gt[partial_supply_idx] = inputs["power_split_amount"][partial_supply_idx] / eta_gt_prop
                unit_mech_power_in_em[partial_supply_idx] = (inputs["mech_power_out"][partial_supply_idx] - inputs["power_split_amount"][partial_supply_idx]) / nm / eta_em_prop
            elif bias == "em":
                # Case where power requested from electric motor is greater than total power needed from propeller
                # if inputs['mech_power_out'] < inputs['power_split_amount']:
                over_supply_idx = np.where(inputs["mech_power_out"] < inputs["power_split_amount"])
                unit_mech_power_in_gt = np.zeros(nn)
                unit_mech_power_in_em = np.zeros(nn)
                unit_mech_power_in_em[over_supply_idx] = inputs["mech_power_out"][over_supply_idx] / nm / eta_em_prop
                unit_mech_power_in_gt[over_supply_idx] = np.zeros(nn)[over_supply_idx]

                # Case where power requested from electric motor is less than total power needed from propeller
                # if inputs['mech_power_out'] >= inputs['power_split_amount']:
                partial_supply_idx = np.where(inputs["mech_power_out"] >= inputs["power_split_amount"])
                unit_mech_power_in_em[partial_supply_idx] = inputs["power_split_amount"][partial_supply_idx] / nm / eta_em_prop
                unit_mech_power_in_gt[partial_supply_idx] = (inputs["mech_power_out"][partial_supply_idx] - inputs["power_split_amount"][partial_supply_idx]) / eta_gt_prop
        # end
        outputs["unit_mech_power_in_gt"] = unit_mech_power_in_gt
        outputs["unit_mech_power_in_em"] = unit_mech_power_in_em

        outputs["component_cost_gt"] = inputs["power_rating_gt"] * cost_inc + cost_base
        outputs["component_cost_em"] = inputs["power_rating_em"] * cost_inc + cost_base
        outputs["component_weight_gt"] = inputs["power_rating_gt"] * weight_inc + weight_base
        outputs["component_weight_em"] = inputs["power_rating_em"] * weight_inc + weight_base
        outputs["component_sizing_margin_gt"] = inputs["mech_power_out"]  / inputs["power_rating_gt"] / eta_gt_prop
        outputs["component_sizing_margin_em"] = inputs["mech_power_out"] / inputs["power_rating_em"] / eta_em_prop

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]
        nm = self.options["num_em_per_nac"]
        rule = self.options["rule"]
        bias = self.options["bias"]
        eta_gt_prop = self.options["prop_gt_gb_efficiency"]
        eta_em_prop = self.options["prop_em_gb_efficiency"]

        if rule == "fraction":
            if bias == "gt":
                J["unit_mech_power_in_gt", "mech_power_out"] = inputs["power_split_fraction_gt"] / eta_gt_prop
                J["unit_mech_power_in_em", "mech_power_out"] = (1 - inputs["power_split_fraction_gt"]) / nm / eta_em_prop
                J["unit_mech_power_in_gt", "power_split_fraction_gt"] = inputs["mech_power_out"] / eta_gt_prop
                J["unit_mech_power_in_em", "power_split_fraction_gt"] = -inputs["mech_power_out"] / nm / eta_em_prop
            elif bias == "em":
                J["unit_mech_power_in_gt", "mech_power_out"] = (1 - inputs["power_split_fraction_em"]) / eta_gt_prop
                J["unit_mech_power_in_em", "mech_power_out"] = inputs["power_split_fraction_em"] / nm / eta_em_prop
                J["unit_mech_power_in_gt", "power_split_fraction_em"] = -inputs["mech_power_out"] / eta_gt_prop
                J["unit_mech_power_in_em", "power_split_fraction_em"] = inputs["mech_power_out"] / nm / eta_em_prop
        elif rule == "fixed":
            if bias == "gt":
                over_supply_idx = np.where(inputs["mech_power_out"] < inputs["power_split_amount"])
                enough_idx = np.where(inputs["mech_power_out"] >= inputs["power_split_amount"])
                
                Jpo_gt_pi = np.zeros(nn)
                Jpo_gt_ps = np.zeros(nn)
                Jpo_em_pi = np.zeros(nn)
                Jpo_em_ps = np.zeros(nn)
                Jpo_gt_pi[over_supply_idx] = np.ones(nn)[over_supply_idx] / eta_gt_prop
                Jpo_gt_ps[over_supply_idx] = np.zeros(nn)[over_supply_idx]
                Jpo_em_pi[over_supply_idx] = np.zeros(nn)[over_supply_idx]
                Jpo_em_ps[over_supply_idx] = np.zeros(nn)[over_supply_idx] 
            # else:
                Jpo_gt_ps[partial_supply_idx] = np.ones(nn)[partial_supply_idx] / eta_gt_prop
                Jpo_gt_pi[partial_supply_idx] = np.zeros(nn)[partial_supply_idx]
                Jpo_em_ps[partial_supply_idx] = -np.ones(nn)[partial_supply_idx] / nm / eta_em_prop
                Jpo_em_pi[partial_supply_idx] = np.ones(nn)[partial_supply_idx] / nm / eta_em_prop
            if bias == "em":
                over_supply_idx = np.where(inputs["mech_power_out"] < inputs["power_split_amount"])
                partial_supply_idx = np.where(inputs["mech_power_out"] >= inputs["power_split_amount"])
                
                Jpo_gt_pi = np.zeros(nn)
                Jpo_gt_ps = np.zeros(nn)
                Jpo_em_pi = np.zeros(nn)
                Jpo_em_ps = np.zeros(nn)
                Jpo_em_pi[over_supply_idx] = np.ones(nn)[over_supply_idx] / nm / eta_em_prop
                Jpo_em_ps[over_supply_idx] = np.zeros(nn)[over_supply_idx]
                Jpo_gt_pi[over_supply_idx] = np.zeros(nn)[over_supply_idx]
                Jpo_gt_ps[over_supply_idx] = np.zeros(nn)[over_supply_idx]

                Jpo_gt_ps[partial_supply_idx] = np.ones(nn)[partial_supply_idx] / eta_gt_prop
                Jpo_gt_pi[partial_supply_idx] = np.zeros(nn)[partial_supply_idx]
                Jpo_em_ps[partial_supply_idx] = -np.ones(nn)[partial_supply_idx] / nm / eta_em_prop
                Jpo_em_pi[partial_supply_idx] = np.ones(nn)[partial_supply_idx] / nm / eta_em_prop
            # end

            J["unit_mech_power_in_gt", "mech_power_out"] = Jpo_gt_pi
            J["unit_mech_power_in_gt", "power_split_amount"] = Jpo_gt_ps
            J["unit_mech_power_in_em", "mech_power_out"] = Jpo_em_pi
            J["unit_mech_power_in_em", "power_split_amount"] = Jpo_em_ps
        J["component_sizing_margin_gt", "mech_power_out"] = 1 / inputs["power_rating_gt"]
        J["component_sizing_margin_em", "mech_power_out"] = 1 / inputs["power_rating_em"]
        J["component_sizing_margin_gt", "power_rating_gt"] = -(inputs["mech_power_out"] / inputs["power_rating_gt"] ** 2)
        J["component_sizing_margin_em", "power_rating_em"] = -(inputs["mech_power_out"] / inputs["power_rating_em"] ** 2)


def test_power_split_nacelle():
    """
    Test function for PowerSplitNacelle component
    """
    import openmdao.api as om
    
    print("Testing PowerSplitNacelle component...")
    
    # Set up the test problem
    nn = 5
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables
    ivc.add_output('mech_power_out', val=np.array([1000, 1500, 2000, 2500, 3000]), units='W', desc='Total power')
    ivc.add_output('power_rating_gt', val=2000.0, units='W', desc='Gas turbine rating')
    ivc.add_output('power_rating_em', val=1000.0, units='W', desc='Electric motor rating')
    ivc.add_output('power_split_fraction_gt', val=np.array([0.7, 0.6, 0.5, 0.4, 0.3]), desc='GT power fraction')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    
    # Test fraction rule with GT bias
    model.add_subsystem('power_split_gt', PowerSplitNacelle(
        num_nodes=nn, 
        num_em_per_nac=2, 
        rule='fraction', 
        bias='gt',
        prop_gt_gb_efficiency=0.95,
        prop_em_gb_efficiency=0.98
    ), promotes=['*'])
    
    prob = om.Problem(model)
    prob.setup()
    
    # Run the model
    prob.run_model()
    
    # Get results
    gt_power = prob.get_val('unit_mech_power_in_gt')
    em_power = prob.get_val('unit_mech_power_in_em')
    gt_margin = prob.get_val('component_sizing_margin_gt')
    em_margin = prob.get_val('component_sizing_margin_em')
    
    print(f"\nResults (Fraction rule, GT bias):")
    print(f"Total Power (W): {prob.get_val('mech_power_out')}")
    print(f"GT Fraction: {prob.get_val('power_split_fraction_gt')}")
    print(f"GT Unit Power (W): {gt_power}")
    print(f"EM Unit Power (W): {em_power}")
    print(f"GT Sizing Margin: {gt_margin}")
    print(f"EM Sizing Margin: {em_margin}")

    prob.check_partials(compact_print=True)
    


if __name__ == "__main__":
    test_power_split_nacelle()
