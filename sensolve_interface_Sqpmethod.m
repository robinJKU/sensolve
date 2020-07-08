function [nlp_f, nlp_g, nlp_jac_grad_gamma_x_p, nlp_jac_g_p, nlp_hess_lag_x, nlp_jac_g_x] = sensolve_interface_Sqpmethod(nlp_solver)
%SENSOLVE_INTERFACE_SQPMETHOD Computes interface functions for
%sensolve from CasADi solver of type specified above

%% check for solver type
callstack = dbstack;
fcn_name_parts = split(callstack(1).name, '_');
expected_solver_name = fcn_name_parts{end};
assert(strcmp(nlp_solver.class_name, expected_solver_name), 'wrong solver of type %s passed, this function only supports %s', nlp_solver.class_name, expected_solver_name);

%% get interface functions
w = casadi.MX.sym('w', nlp_solver.size1_in(0));
lam_g = casadi.MX.sym('lam_g', nlp_solver.size1_in(4));
p = casadi.MX.sym('p', nlp_solver.size1_in(1));

% nlp_fg:(x,p)->(f,g)
nlp_fg = nlp_solver.get_function('nlp_fg');
% nlp_grad:(x,p,lam_f,lam_g)->(f,g,grad_gamma_x,grad_gamma_p)
nlp_grad = nlp_solver.get_function('nlp_grad');
% nlp_hess_l:(x,p,lam_f,lam_g)->(sym_hess_gamma_x_x)
nlp_hess_l = nlp_solver.get_function('nlp_hess_l');
% nlp_jac_fg:(x,p)->(f,grad_f_x,g,jac_g_x)
nlp_jac_fg = nlp_solver.get_function('nlp_jac_fg');
[f, g] = nlp_fg(w, p);
nlp_f = casadi.Function('nlp_f', {w, p}, {f});
nlp_g = casadi.Function('nlp_g', {w, p}, {g});
[~,~,grad_gamma_x,~] = nlp_grad(w, p, [], lam_g);
jac_grad_gamma_x_p = jacobian(grad_gamma_x, p) + jacobian(jacobian(f,w),p);
nlp_jac_grad_gamma_x_p = casadi.Function('nlp_jac_grad_gamma_x_p', {w, p, lam_g}, {jac_grad_gamma_x_p});
[~,~,~,jac_g_x] = nlp_jac_fg(w, p);
nlp_jac_g_x = casadi.Function('nlp_jac_g_x', {w, p} , {jac_g_x});
jac_g_p = jacobian(g, p);
nlp_jac_g_p = casadi.Function('nlp_jac_g_p', {w, p}, {jac_g_p});
hess_lag_x = nlp_hess_l(w, p, [], lam_g) + hessian(f, w);
nlp_hess_lag_x = casadi.Function('nlp_hess_lag_x', {w, p, lam_g} , {hess_lag_x});

end

