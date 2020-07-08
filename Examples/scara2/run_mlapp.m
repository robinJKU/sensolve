clear
close
clc

addpath('../..'); % sensolve

% load nominal solution
load('nom.mat');

% run mlapp
scara2(w_nom, p_nom, lam_g_nom);
