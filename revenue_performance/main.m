clc; clear;

n = 10;
m = 15;
tau_train = 500;
tau_valid = 0.25 * tau_train;
tau_test = 0.5 * tau_train;

num_ground_choice_model = 10;
whole_set = 1 : n;
vec_log_lik_train = zeros(4, num_ground_choice_model);
vec_log_lik_test = zeros(4, num_ground_choice_model);
vec_test_rmse = zeros(4, num_ground_choice_model);
vec_customer_type_list = cell(m, 2 * num_ground_choice_model);
vec_lambda_list = zeros(m, num_ground_choice_model);
vec_v_mnl = zeros(n, num_ground_choice_model);
vec_v_gam = zeros(n, num_ground_choice_model);
vec_w_gam = zeros(n, num_ground_choice_model);
vec_focal_idx = zeros(num_ground_choice_model, 1);
vec_ranking = zeros(n, num_ground_choice_model);
vec_v_flm = zeros(n, num_ground_choice_model);
vec_delta_flm = zeros(num_ground_choice_model, 1);
vec_mmnl_gamma = cell(num_ground_choice_model, 1);
vec_mmnl_mnl = cell(num_ground_choice_model, 1);
vec_purchase_hist_train = cell(num_ground_choice_model, 1);
vec_purchase_hist_validation = cell(num_ground_choice_model, 1);
vec_purchase_hist_test = cell(num_ground_choice_model, 1);

for sample_idx = 1 : num_ground_choice_model

[log_lik_train, rec_exitflag, customer_type_list, lambda_list, ...
     purchase_hist_train, purchase_hist_validation, v_mnl, v_gam, w_gam, ...
      focal_idx, ranking, v_flm, delta_flm, purchase_hist_test] = estimate_model(m, n, tau_train, tau_valid, tau_test, sample_idx);

vec_log_lik_train(1:3, sample_idx) = log_lik_train;
vec_customer_type_list(:, (2*sample_idx-1) : 2*sample_idx) = customer_type_list;
vec_lambda_list(:, sample_idx) = lambda_list;
vec_v_mnl(:, sample_idx) = v_mnl;
vec_v_gam(:, sample_idx) = v_gam;
vec_w_gam(:, sample_idx) = w_gam;
vec_focal_idx(sample_idx) = focal_idx;
vec_ranking(:, sample_idx) = ranking;
vec_v_flm(:, sample_idx) = v_flm;
vec_delta_flm(sample_idx) = delta_flm;
vec_purchase_hist_train{sample_idx} = purchase_hist_train;
vec_purchase_hist_validation{sample_idx} = purchase_hist_validation;
vec_purchase_hist_test{sample_idx} = purchase_hist_test;

end
%% 
% for sample_idx = 1 : num_ground_choice_model
%     fo = strcat('./ground_mmnl_', num2str(tau_train), '_', num2str(sample_idx), '_gamma', '.txt');
%     mmnl_gamma = load(fo);
%     fo = strcat('./ground_mmnl_', num2str(tau_train), '_', num2str(sample_idx), '_mnl', '.txt');
%     mmnl_mnl = load(fo);
%     mmnl_log = mmnl_gamma(end);
%     mmnl_gamma = mmnl_gamma(1:end-1);
%     vec_log_lik_train(4, sample_idx) = mmnl_log;
%     vec_mmnl_gamma{sample_idx} = mmnl_gamma;
%     vec_mmnl_mnl{sample_idx} = mmnl_mnl;
% end

%% compute prediction performance

for sample_idx = 1 : num_ground_choice_model
    
    v_mnl = vec_v_mnl(:, sample_idx);
    v_gam = vec_v_gam(:, sample_idx);
    w_gam = vec_w_gam(:, sample_idx);
    focal_idx = vec_focal_idx(sample_idx);
    ranking = vec_ranking(:, sample_idx);
    v_flm = vec_v_flm(:, sample_idx);
    delta_flm = vec_delta_flm(sample_idx);
    % mmnl_gamma = vec_mmnl_gamma{sample_idx};
    % mmnl_mnl = vec_mmnl_mnl{sample_idx};
    
    x_mnl = log(v_mnl);
    vec_log_lik_test(1, sample_idx) = mnl_log_lik(x_mnl, n, tau_test, purchase_hist_test);
    
    x_gam = log(v_gam);
    x_gam = [x_gam; log(w_gam)];
    vec_log_lik_test(2, sample_idx) = gam_log_lik(x_gam, n, tau_test, purchase_hist_test);
    
    params = log(v_flm);
    params = [params; delta_flm];
    vec_log_lik_test(3, sample_idx) = -flm_total_loglikelihood(params, n, tau_test, ranking, focal_idx, purchase_hist_test);
    % vec_log_lik_test(4, sample_idx) = mmnl_log_lik(mmnl_gamma, mmnl_mnl, tau_test, purchase_hist_test);
    mmnl_gamma = 0;
    mmnl_mnl = 0;
    
    test_rmse = compute_rmse(n, x_mnl, x_gam, params, ranking, focal_idx, mmnl_gamma, mmnl_mnl, tau_test, purchase_hist_test);
    vec_test_rmse(:, sample_idx) = test_rmse;
    
end


%%
num_revenue_sample = 10;
vec_revenue = zeros(num_ground_choice_model, num_revenue_sample, 4); % add offering everything
for sample_idx = 1 : num_ground_choice_model
    
    customer_type_list = vec_customer_type_list(:, (2*sample_idx-1) : 2*sample_idx);
    lambda_list = vec_lambda_list(:, sample_idx) ;
    v_mnl = vec_v_mnl(:, sample_idx);
    v_gam = vec_v_gam(:, sample_idx);
    w_gam = vec_w_gam(:, sample_idx);
    focal_idx = vec_focal_idx(sample_idx);
    ranking = vec_ranking(:, sample_idx);
    v_flm = vec_v_flm(:, sample_idx);
    delta_flm = vec_delta_flm(sample_idx);
    % mmnl_gamma = vec_mmnl_gamma{sample_idx};
    % mmnl_mnl = vec_mmnl_mnl{sample_idx};
    
    for jj = 1 : num_revenue_sample
        
        
        % seed = 100 + jj * 888;
        seed = 100 + jj * 888;
        rng(seed);
        r = sort(rand(n, 1)*100, 'ascend'); % price vector; high to low 
        
        % OPT
        [S_opt, rev_opt] = gsp_opt_assort(customer_type_list, m, n, r, lambda_list);
        % mnl
        cvx_begin quiet
            cvx_solver mosek
            variable x(n,1) nonnegative
            variable x0 nonnegative
            obj = r'*x;
            maximize obj
            subject to 
                sum(x)+x0 == 1;
                x./v_mnl <= x0;
        cvx_end
        S_mnl = whole_set(x>eps);

        % gam
        cvx_begin quiet
            cvx_solver mosek
            variable x(n,1) nonnegative
            variable x0 nonnegative
            obj = r'*x;
            maximize obj
            subject to 
                sum((v_gam-w_gam)./v_gam.*x)+(1+sum(w_gam))*x0 == 1;
                x./v_gam <= x0;
        cvx_end
        S_gam = whole_set(x>eps);

        % flm
        S_flm = flm_opt_assort(r, n, focal_idx, ranking, v_flm, delta_flm);
        S_flm = find(S_flm == 1);

        % mmnl
        % S_mmnl = mmnl_opt_assort(r, n, mmnl_gamma, mmnl_mnl);

        revenue_opt = 0;
        revenue_mnl = 0;
        revenue_gam = 0;
        revenue_flm = 0;
        % revenue_mmnl = 0;

        for cus_type = 1:m % equal probability
            temp = customer_type_list{cus_type, 1};
            gsp_focal = customer_type_list{cus_type, 2};

            % OPT
            
            idx = find(sum(temp==S_opt',1));
            if(~isempty(idx)) % if buying something
                choice = temp(idx(min( length(idx), gsp_focal )));
                revenue_opt = revenue_opt + r(choice) * lambda_list(cus_type);
            end



            % MNL
            idx = find(sum(temp==S_mnl',1));
            if(~isempty(idx)) % if buying something
                choice = temp(idx(min( length(idx), gsp_focal )));
                revenue_mnl = revenue_mnl + r(choice) * lambda_list(cus_type);
            end
            % GAM
            idx = find(sum(temp==S_gam',1));
            if(~isempty(idx)) % if buying something
                choice = temp(idx(min( length(idx), gsp_focal )));
                revenue_gam = revenue_gam + r(choice) * lambda_list(cus_type);
            end    
            % FLM
            idx = find(sum(temp==S_flm',1));
            if(~isempty(idx)) % if buying something
                choice = temp(idx(min( length(idx), gsp_focal )));
                revenue_flm = revenue_flm + r(choice) * lambda_list(cus_type);
            end  
            % MMNL
%             idx = find(sum(temp==S_mmnl',1));
%             if(~isempty(idx)) % if buying something
%                 choice = temp(idx(min( length(idx), gsp_focal )));
%                 revenue_mmnl = revenue_mmnl + r(choice) * lambda_list(cus_type);
%             end  
        end

        vec_revenue(sample_idx,jj,1) = revenue_mnl;
        vec_revenue(sample_idx,jj,2) = revenue_gam;
        vec_revenue(sample_idx,jj,3) = revenue_flm;
        % vec_revenue(sample_idx,jj,4) = revenue_mmnl;
        vec_revenue(sample_idx,jj,5) = revenue_opt;


    end
end

vec_ratio = vec_revenue(:,:,1:4)./vec_revenue(:,:,5);
vec_avg_ratio = squeeze(mean(vec_ratio, 2));

mean(vec_avg_ratio, 1)
