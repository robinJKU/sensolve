# sensolve
A solver for approximate NLP solutions based on parametric sensitivities

## Download & Installation
* Clone (or download) this repo
* add the sensolve folder to your MATLAB path
    ```
    addpath('path-to-sensolve')
    ```
* also add the CasADi path to your MATLAB path
    ```
    addpath('path-to-CasADi')
    ```

## Usage
```
% nlp_solver: CasADi solver instance of supported type (built-in support for SQP and Ipopt)
% g_mask_eq: equality constraint mask (= always active constraints)
% w_nom: nominal solution: optimization variables
% lam_g_nom: nominal solution: multipliers
% p_nom: nominal parameters
% p: perturbed parameters
s = sensolve(nlp_solver); % create solver instance
[w, lam_g, g_mask_act, stats] = s.solve(w_nom, p_nom, p, lam_g_nom, g_mask_eq);
% w: approximation of perturbed solution: optimization variables
% lam_g: approximation of perturbed solution: multipliers
% g_mask_act: mask of active set of constraints
% stats: computational statistics
```

## How to cite sensolve
If you successfully used sensolve in an academic research application and would like to include a reference to our work, please cite our following paper:

*TODO*