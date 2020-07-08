classdef sensolve < handle
    %SENSOLVE Solver for approximate NLP solution based on parametric
    %sensitivities
    
    properties
        nlp_solver
        dim
        nlp_f
        nlp_g
        nlp_jac_grad_gamma_x_p
        nlp_jac_g_p
        nlp_hess_lag_x
        nlp_jac_g_x
    end
    
    methods
        function obj = sensolve(nlp_solver)
            %SENSOLVE Construct an instance of this class
            % nlp_solver: CasADi solver object of supported class
            
            % store solver object
            obj.nlp_solver = nlp_solver;
            % solver:(x0,p,lbx,ubx,lbg,ubg,lam_x0,lam_g0)->(x,f,g,lam_x,lam_g,lam_p)
            
            % get variable sizes
            obj.dim = struct('w', nlp_solver.size1_in(0), ...
                             'g', nlp_solver.size1_in(4), ...
                             'p', nlp_solver.size1_in(1));
                         
            % set function required for sensitivity analysis
            obj.set_sens_funs();
        end
        
        function set_sens_funs(obj)
            %set_sens_funs sets internal functions required for computing
            %parametric sensivities
            
            sensolve_interface_fcn_name = ['sensolve_interface_', obj.nlp_solver.class_name];
            assert(exist([sensolve_interface_fcn_name,'.m'], 'file') == 2, 'solver %s not interfaced', obj.nlp_solver.class_name);
            sensolve_interface_fcn = eval(['@', sensolve_interface_fcn_name]);
            [obj.nlp_f, obj.nlp_g, obj.nlp_jac_grad_gamma_x_p, obj.nlp_jac_g_p, obj.nlp_hess_lag_x, obj.nlp_jac_g_x] = sensolve_interface_fcn(obj.nlp_solver);
            
        end
        
        function codegen(obj, filename)            
            % codegen Code-generate functions required for computing
            %parametric sensivities
            % filename: filename of target C file
            CG = casadi.CodeGenerator(filename);
            CG.add(obj.nlp_f);
            CG.add(obj.nlp_g);
            CG.add(obj.nlp_jac_grad_gamma_x_p);
            CG.add(obj.nlp_jac_g_p);
            CG.add(obj.nlp_hess_lag_x);
            CG.add(obj.nlp_jac_g_x);
            CG.generate();           
        end
        
        function use_external(obj, filename)
            % use_external Use external code-generated functions
            % filename: filename of DLL
            % get external function
            obj.nlp_f = casadi.external('nlp_f', filename);
            obj.nlp_g = casadi.external('nlp_g', filename);
            obj.nlp_jac_grad_gamma_x_p = casadi.external('nlp_jac_grad_gamma_x_p', filename);
            obj.nlp_jac_g_p = casadi.external('nlp_jac_g_p', filename);
            obj.nlp_hess_lag_x = casadi.external('nlp_hess_lag_x', filename);
            obj.nlp_jac_g_x = casadi.external('nlp_jac_g_x', filename); 
        end
        
        function [g_idxvec_act, g_mask_act] = get_activeset(obj, g)
            %get_activeset Compute active set from constraint residue
            %g: constraint residue vector
            %g_idxvec_act: index vector of active constraints
            %g_mask_act: binary mask of active constraints
            
            act_thres = -1e-6; % constant activation threshold
            g_mask_act = full(g) > act_thres;
            g_idxvec_act = idxvec(g_mask_act);
        end
        
        function [jac_w_p, jac_lam_p, g_idxvec_act, g_mask_act] = get_sens(obj, w, p, lam_g)
            %get_sens compute sensitivities
            hess_lag_w = sparse(obj.nlp_hess_lag_x(w, p, lam_g));
            g = obj.nlp_g(w, p);
            [g_idxvec_act, g_mask_act] = get_activeset(obj, g);
%             if isempty(g_idxvec_act)
%                 jac_w_p = zeros(0,length(p));
%                 jac_lam_p = zeros(0,length(p));
%                 return
%             end
            jac_g_w = sparse(obj.nlp_jac_g_x(w, p));
            jac_g_w_act = jac_g_w(g_idxvec_act,:);
            jac_grad_gamma_x_p = sparse(obj.nlp_jac_grad_gamma_x_p(w, p, lam_g));
            jac_g_p = sparse(obj.nlp_jac_g_p(w, p));
            jac_g_p_act = jac_g_p(g_idxvec_act,:);
            KKT_mat = [hess_lag_w, jac_g_w_act'; 
                       jac_g_w_act, sparse(zeros(length(g_idxvec_act)))];
            rhs_mat = [jac_grad_gamma_x_p; jac_g_p_act];
            sens_mat = full(-KKT_mat\rhs_mat);
            jac_w_p = sens_mat(1:obj.dim.w,:);
            jac_lam_p = sens_mat(obj.dim.w+1:end,:);
        end
        
        function [p_range, p_range_idx_g, p_range_event] = get_adm_p_range(obj, w, p, lam_g)
            %get_adm_p_range Compute first-order approximation of admissible
            %parameter range
            %w: optimization variables
            %p: parameters
            %lam_g: multipliers
            %p_range: approx. admissible parameter range
            %p_range_idx_g: index matrix of constraint with status change
            %p_range_event: event occuring - 0 = leave AS, 1 = enter AS
            
            g = sparse(obj.nlp_g(w, p));
            [~, g_mask_act] = obj.get_activeset(g);
            [jac_w_p, jac_lam_p] = obj.get_sens(w, p, lam_g);
            jac_g_p = sparse(obj.nlp_jac_g_p(w, p));
            g_jac_p_nact = jac_g_p(~g_mask_act, :);
            jac_g_w = sparse(obj.nlp_jac_g_x(w, p));
            jac_g_w_nact = jac_g_w(~g_mask_act, :);
            p_range = repmat([-inf, inf], [obj.dim.p, 1]);
            p_range_idx_g = nan(obj.dim.p, 2);
            p_range_event = nan(obj.dim.p, 2);
            g_pjac_p_nact = g_jac_p_nact + jac_g_w_nact*jac_w_p;
            for i = 1:obj.dim.p
                p_enter = [];
                p_leave = [];
                p_enter_smaller = [];
                p_enter_larger = [];
                p_leave_smaller = [];
                p_leave_larger = [];
                for j = 1:obj.dim.g
                    if ~g_mask_act(j) % enter?
                        p_enter = p(i) - g(j)./g_pjac_p_nact(sum(~g_mask_act(1:j)),i);
                        if p_enter < p(i)
                            p_enter_smaller = p_enter;
                        elseif p_enter > p(i)
                            p_enter_larger = p_enter;
                        end
                    else % leave?
                        p_leave = p(i) - lam_g(j)./jac_lam_p(sum(g_mask_act(1:j)),i);
                        if p_leave < p(i)
                            p_leave_smaller = p_leave;
                        elseif p_leave > p(i)
                            p_leave_larger = p_leave;
                        end
                    end
                    % enter: 1, leave: 0
                    [p_range(i,1), idx] = max([p_range(i,1), p_enter_smaller]);
                    if idx ~= 1
                        p_range_idx_g(i,1) = j;
                        p_range_event(i,1) = 1;
                    end
                    [p_range(i,1), idx] = max([p_range(i,1), p_leave_smaller]);
                    if idx ~= 1
                        p_range_idx_g(i,1) = j;
                        p_range_event(i,1) = 0;
                    end
                    [p_range(i,2), idx] = min([p_range(i,2), p_enter_larger]);
                    if idx ~= 1
                        p_range_idx_g(i,2) = j;
                        p_range_event(i,2) = 1;
                    end
                    [p_range(i,2), idx] = min([p_range(i,2), p_leave_larger]);
                    if idx ~= 1
                        p_range_idx_g(i,2) = j;
                        p_range_event(i,2) = 0;
                    end
                end
            end
        end
        
        function [w, lam_g, g_mask_act, stats] = solve(obj, w, p_nom, p, lam_g, g_mask_eq)
            %solve: Compute solution using iterative feedback scheme
            %w: optimization variables
            %p_nom: nominal parameters
            %p: parameters
            %lam_g: multipliers
            %g_mask_eq: equality constraint mask
            %g_mask_act: active set mask
            %stats: computational statistics
            
            error_flag = false;
            [last_warn_msg, last_warn_ID] = lastwarn();
            if strcmp(last_warn_ID, 'MATLAB:singularMatrix') || strcmp(last_warn_ID, 'MATLAB:nearlySingularMatrix')
                lastwarn('', '');
            end
            
            % settings, convert to varargin options struct?
            cnt_it_max = 100;
            g_act_thresh = 1e-6;
            
            % init stats
            stats.iter_count = 0;
            stats.activeset_changes_count = 0;
            stats.t_wall_total = 0;
            stats.t_iter = [];
            
            % first-order approximation
            timer_1st = tic;
            [jac_w_p, jac_lam_p, ~, g_mask_act] = obj.get_sens(w, p_nom, lam_g);
            dp = p_nom - p;
            w = w - jac_w_p*dp;
            if any(g_mask_act)
                lam_g(g_mask_act) = lam_g(g_mask_act) - jac_lam_p*dp;   
            end
            stats.t_first_order = toc(timer_1st);
            stats.t_wall_total = stats.t_wall_total + stats.t_first_order;
            [~, warn_ID] = lastwarn();
            if strcmp(warn_ID, 'MATLAB:singularMatrix') || strcmp(warn_ID, 'MATLAB:nearlySingularMatrix')
                error_flag = true;
                stats.return_status = 'Singular_KKT_First_Order';
            end
            
            g = sparse(obj.nlp_g(w, p));
            [g_idxvec_act, g_mask_act] = obj.get_activeset(g);
            g_res = g(g_mask_act);
            while any([abs(g(g_mask_eq)); g(~g_mask_eq)] > g_act_thresh) && ~error_flag % norm(g_res) > ident   
                timer_iter = tic;
                % KKT system of sensitivities
                hess_lag_w = full(obj.nlp_hess_lag_x(w, p, lam_g));
                jac_g_w = obj.nlp_jac_g_x(w, p);
                jac_g_w_act = full(jac_g_w(g_idxvec_act,:));
                KKT_mat = [hess_lag_w, jac_g_w_act'; 
                           jac_g_w_act, zeros(length(g_idxvec_act))];
                rhs_mat = [zeros(obj.dim.w, length(g_idxvec_act)); -eye(length(g_idxvec_act))];
                
                % parameter sensitivities wrt. residue of constraints
%                 KKT_mat = full(KKT_mat); rhs_mat = full(rhs_mat);
                sens_mat = -KKT_mat\rhs_mat;
%                 sens_mat = full(-KKT_mat\rhs_mat);
                jac_w_g_res = sens_mat(1:obj.dim.w,:);
                jac_lam_g_res = sens_mat(obj.dim.w+1:end,:);
                [~, warn_ID] = lastwarn();
                if strcmp(warn_ID, 'MATLAB:singularMatrix') || strcmp(warn_ID, 'MATLAB:nearlySingularMatrix')
                    error_flag = true;
                    stats.return_status = 'Singular_KKT_Iterations';
                end
                
                % update primal and dual variables
                w = w - jac_w_g_res*g_res;
                lam_g(g_mask_act) = lam_g(g_mask_act) - jac_lam_g_res*g_res;
                              
                % update active set
                g = sparse(obj.nlp_g(w, p));
                g_mask_act_prev = g_mask_act;
                [~, g_mask_act] = obj.get_activeset(g);
                g_mask_act = g_mask_act | g_mask_eq;
                g_idxvec_act = idxvec(g_mask_act);
                lam_g(~g_mask_act) = 0;
                g_res = g(g_mask_act);
                
                % update stats
                if any(g_mask_act_prev ~= g_mask_act)
                    stats.activeset_changes_count = stats.activeset_changes_count + 1;
                end
                stats.iter_count = stats.iter_count + 1;
                if stats.iter_count > cnt_it_max
                    stats.return_status = 'Maximum_Iterations_Reached';
                    error_flag = true;
                end
                stats.t_iter(stats.iter_count) = toc(timer_iter);
            end
            stats.success = ~error_flag;
            stats.t_wall_total = stats.t_wall_total + sum(stats.t_iter);
            if stats.success
                stats.return_status = 'Solve_Succeeded';
                % return warning status
                if ~isempty(last_warn_msg)
                    lastwarn(last_warn_msg, last_warn_ID);
                end
            end            
        end
    end
end

