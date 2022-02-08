# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:07:40 2021

@author: Adrian Ramos Perez
"""
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import DiscreteFactor

def var_elimination(factors, query_variables, evidence=None, elimination_order=None):
    
    """
    This function takes as inputs:
     - The set of factors $\bar{\Phi}$ that model the problem.
     - The variables that won't be eliminated Y (query variables).
     - The evidence (E=e).
     - The elimination order.
     
    And returns the inferred probability P(Y|E=e).
    
    :param list[DiscreteFactor] factors: List of factors that model the problem.
    :param list[str] query_variables: Query variables.
    :param dict{str: int} evidence: Evidence in the form of a dictionary. For example evidence={'D': 2, 'E':0}
                                    means that D=d² and E=e⁰.
    :param list[str] elimination_order: Specification of the order in which the variables will be eliminated.
    :return: DiscreteFactor corresponding the inferred probability.
    """
    # --------------------------------------------- Parameter check ---------------------------------------------
    if not isinstance(factors, list) and not factors:
        raise ValueError(f"The parameter factors: {factors} must be a nonempty list of DiscreteFactor objects.")
    
    if not isinstance(query_variables, list) and not query_variables:
        raise ValueError(f"The parameter query_variables: {query_variables} must be a nonempty list of str objects.")
    
    if evidence is not None and (not isinstance(evidence, dict) and not evidence):
        raise ValueError(f"The parameter evidence: {evidence} must be a nonempty dict.")
    
    if elimination_order is not None and (not isinstance(elimination_order, list) and not elimination_order):
        raise ValueError(f"The parameter elimination_order: {elimination_order} must be a nonempty list of str objects.")
    # --------------------------------------------- End parameter check -----------------------------------------
    
    # Initial parameters
    # Number of factors
    m = len(factors)
    # Get variables
    variables = []
    for i in range(m):
        variables.append(factors[i].variables)
    variables = list(set(variables))
    # Number of variables
    n = len(variables)
    # Evidence variables
    evidence_variables = list(evidence.keys())
    
    # 1. If evidence is not None, we must reduce all the factors according to the evidence
    if evidence is not None:
        # For each factor
        for i in range(m):
            # Find intersection of variables between evidence and factor scope
            intersection = set(evidence_variables).intersection(set(factor[i].variables))
            # If intersection is not empty, we must reduce this factor
            if intersection:
                ev = {var: evidence[var] for var in intersection if var in evidence}
                factor[i] = factor[i].reduce(ev.items(), inplace=False)
                
    # Variables to eliminate
    variables_to_eliminate = set(variables).difference(set(query_variables))
    if evidence is not None:
        variables_to_eliminate = variables_to_eliminate.difference(set(evidence_variables))
    variables_to_eliminate = list(variables_to_eliminate)
    
    # If elimination_order is not None, we must check if the variables in elimination_order are right.
    # If the variables in elimination_order are right, then they should be set as variables_to_eliminate.
    if elimination_order is not None:
        if not set(elimination_order).difference(set(variables_to_eliminate)):
            raise ValueError(f"The parameter elimination_order: {elimination_order} does not contain the right variables.")
        else:
            variables_to_eliminate = elimination_order
        
    # 2. Eliminate Var-Z
    factors_update = factors.copy()
    for var in variables_to_eliminate:
        # ---------------------------- Your code goes here! ----------------------------------
        # Determine the set of factors that involve var
        
        # Compute the product of these factors
        
        # Marginalize var
        
        # Overwrite factors_update
        
        # ------------------------------------------------------------------------------------
        
    # 3. Multiply the remaining factors
        pass
    
#%% 2. PERFORM INFERENCE OVER STUDENT EXAMPLE
# Bayesina Network definition:
student_model = BayesianModel([("D","C"),
                               ("I","C"),
                               ("I","E"),
                               ("C","R")])
# Conditional Probability Distributions:
cpd_D = TabularCPD(variable="D", variable_card=2, values=[[0.6], [0.4]])
cpd_I = TabularCPD(variable="I", variable_card=2, values=[[0.7], [0.3]])
cpd_C = TabularCPD(variable="C", variable_card=3, evidence=['I','D'], evidence_card=[2,2], values=[[0.3, 0.7, 0.02, 0.2],
                                                                                                   [0.4, 0.25, 0.08, 0.3],
                                                                                                   [0.3, 0.05, 0.9, 0.5]    ])
cpd_E = TabularCPD(variable="E", variable_card=2, evidence=['I'], evidence_card=[2], values=[[0.95, 0.2],
                                                                                             [0.05, 0.8]    ])
cpd_R = TabularCPD(variable="R", variable_card=2, evidence=['C'], evidence_card=[3], values=[[0.99, 0.4, 0.1],
                                                                                             [0.01, 0.6, 0.9]    ])
student_model.add_cpds(cpd_D, cpd_I, cpd_C, cpd_E, cpd_R)
# Verify local independencies:
print("B. Network Independencies:\n",student_model.local_independencies(['I','D','C','E','R']))

inference = VariableElimination(student_model)
#   1. Causal reasoning:
P_R = inference.query(variables=["R"] )
P_R_given_I0 = inference.query(variables=["R"], evidence={"I":0} )
P_R_given_I0_D0 = inference.query(variables=["R"], evidence={"I":0, "D":0} )
print("\nProbability of R: \n",P_R)
print("\nProbability of R given I = 0: \n",P_R_given_I0)
print("\nProbability of R given I = 0, D = 0: \n",P_R_given_I0_D0)

#   2. Evidential reasoning:
P_D = inference.query(variables=["D"] )
P_D_given_C0 = inference.query(variables=["D"], evidence={"C":0} )
P_I = inference.query(variables=["I"] )
P_I_given_C0 = inference.query(variables=["I"], evidence={"C":0} )
print("\nProbability of D: \n",P_D)
print("\nProbability of D given C = 0: \n",P_D_given_C0)
print("\nProbability of I: \n",P_I)
print("\nProbability of I given C = 0: \n",P_I_given_C0)

#   3. Intercausal reasoning:
P_I = inference.query(variables=["I"] )
P_I_given_C0 = inference.query(variables=["I"], evidence={"C":0} )
P_I_given_C0_D1 = inference.query(variables=["I"], evidence={"C":0, "D":1} )
print("\nProbability of I: \n",P_I)
print("\nProbability of I given C = 0: \n",P_I_given_C0)
print("\nProbability of I given C = 0, D = 1: \n",P_I_given_C0_D1)