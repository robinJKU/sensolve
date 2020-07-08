clear
clc

addpath('../..'); % sensolve

%% PNLP
w = casadi.SX.sym('w', 2);
p = casadi.SX.sym('p', 2);
cost = 1/2*(w(1) - p(1))^2 + 1/2*(w(2) - p(1))^2 + 1/2*w(1)*w(2);
g = w'*diag([1,5])*w - p(2);
g_lb = -inf;
g_ub = 0;
g_mask_eq = false;
nlp = struct('f', cost, 'x', w, 'g', g, 'p', p);

%% NLP solver
sqp = casadi.nlpsol('osqp', 'sqpmethod', nlp, struct('qpsol', 'osqp'));
ipopt = casadi.nlpsol('ipopt', 'ipopt', nlp);

%% sensolve
s = sensolve(sqp);

%% exact solution nominal case
p_nom = [0.25; 0.25];
sol_nom_sqp = sqp('lbg', g_lb, 'ubg', g_ub, 'p', p_nom);
sqp_stats = sqp.stats;
sol_nom_ipopt = ipopt('lbg', g_lb, 'ubg', g_ub, 'p', p_nom);
ipopt_stats = ipopt.stats;
w_nom = full(sol_nom_sqp.x);
lam_g_nom = full(sol_nom_sqp.lam_g);

%% exact solution perturbed case
p_pert =  [0.25;0];
sol_pert_sqp = sqp('lbg', g_lb, 'ubg', g_ub, 'p', p_pert, 'x0', sol_nom_sqp.x, 'lam_g0', sol_nom_sqp.lam_g);
sol_pert_ipopt = ipopt('lbg', g_lb, 'ubg', g_ub, 'p', p_pert, 'x0', sol_nom_sqp.x, 'lam_g0', sol_nom_sqp.lam_g);

%% approximate solution perturbed case
[w_approx, lam_g_approx, g_mask_act_approx, sensolve_stats] = s.solve(w_nom, p_nom, p_pert, lam_g_nom, g_mask_eq);

