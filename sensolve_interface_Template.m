function [nlp_f, nlp_g, nlp_jac_grad_gamma_x_p, nlp_jac_g_p, nlp_hess_lag_x, nlp_jac_g_x] = sensolve_interface_Template(nlp_solver)
%SENSOLVE_INTERFACE_TEMPLATE Computes interface functions for
%sensolve from CasADi solver of type specified above

%% check for solver type
callstack = dbstack;
fcn_name_parts = split(callstack.name, '_');
expected_solver_name = fcn_name_parts{end};
assert(strcmp(nlp_solver.class_name, expected_solver_name), 'wrong solver of type %s passed, this function only supports %s', nlp_solver.class_name, expected_solver_name);

%% get interface functions
w = casadi.MX.sym('w', nlp_solver.size1_in(0));
lam_g = casadi.MX.sym('lam_g', nlp_solver.size1_in(4));
p = casadi.MX.sym('p', nlp_solver.size1_in(1));


% provide the following CasADi functions from the nlp_solver object
% retrieve functions via nlp_solver.get_function(function_name)
% list available functions with nlp_solver.get_functions()
nlp_f
nlp_g
nlp_jac_grad_gamma_x_p 
nlp_jac_g_p 
nlp_hess_lag_x 
nlp_jac_g_x

end

