import openmdao.api as om
import numpy as np
import random
import math 
random.seed(2)

#Adapted from canonical example openMDAO https://openmdao.org/newdocs/versions/latest/basic_user_guide/command_line/check_setup.html

#what is the lever? trust and police fairness
# Why framing it as an MDO problem is essential, 
    #increased analysis and more avenue of research. its not static anymore.
class Police(om.ExplicitComponent):
    """
    Component containing 
    """

    def setup(self):

        # Global Design Variable is z1 = police features/characterstics and z2 = neighborhood features/characterstics
        self.add_input('z1', val=np.zeros(1))

        # Coupling parameter how much do you trust police?  0<=y2<=1 ..idea is if people see cops to often they hate it but crime increases if they don't see cops
        self.add_input('y2', val=1.0)

        # Coupling output - How often to do patrolling? 0<=y1<=1..idea is how often to do patrolling taking into account that some neighborhood might not want to see cops more often.
        self.add_output('y1', val=1.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the rergression equation gotten from a survey
        This could be any surrogate model
       
        """
        z1 = inputs['z1'][0] #only police features/characterstics
        y2 = inputs['y2']

        def func_police(z1):
            y1 = z1**2
            return y1
        outputs['y1'] = (1 - y2)*func_police(z1) 



class Neighborhood(om.ExplicitComponent):
    """
    Neighborhood related details and their relation with police 
    """

    def setup(self):

        # Global Design Variable
        self.add_input('z2', val=np.zeros(1))

        # # Local Design Variable
        # self.add_input('x', val=0.)

        # Coupling parameter
        self.add_input('y1', val=1.0)

        # Coupling output
        self.add_output('y2', val=1.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the surrogate equation
        """
        z2 = inputs['z2'][0] #Neighborhood characterstics..
        y1 = inputs['y1']
        #add an random int to account for other stuff

        def func_neighbor(z2):
            y2 = 2*math.log(z2)
            return y2
        outputs['y2'] = (1 - y1)*func_neighbor(z2) 


class SocioTechnical(om.Group):
    """
  
    """

    def setup(self):
        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('d1', Police(), promotes_inputs=['z1', 'y2'],
                            promotes_outputs=['y1'])
        cycle.add_subsystem('d2', Neighborhood(), promotes_inputs=['z2', 'y1'],
                            promotes_outputs=['y2'])

        cycle.set_input_defaults('z1', np.array([0.5]))
        cycle.set_input_defaults('z2', np.array([0.5]))

        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = om.NonlinearBlockGS()

        # fairness objective for the decision STS :Hoovers Concentration index
        # Since we assume 1 neighborhood now..
        self.add_subsystem('obj_cmp', om.ExecComp('obj = 1/0.5)*(y1-y2)'), promotes = ['*'])



if __name__ == "__main__":
    prob = om.Problem()
    prob.model = SocioTechnical()

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['maxiter'] = 100
    prob.driver.options['tol'] = 1e-8

    prob.model.add_design_var('z1', lower=0, upper=1)
    prob.model.add_design_var('z2', lower=0, upper=1)
    prob.model.add_objective('obj')
    # prob.model.add_constraint('con1', upper=0)
    # prob.model.add_constraint('con2', upper=0)

    # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
    prob.model.approx_totals()

    prob.setup()
    prob.set_solver_print(level=0)

    prob.run_driver()

    print('minimum found at')
    print(prob.get_val('z1'))
    print(prob.get_val('z2'))

    print('minumum objective')
    print(prob.get_val('obj')[0])