function test_rmse = compute_rmse(n, x_mnl, x_gam, params, ranking, focal_idx, mmnl_gamma, mmnl_mnl, tau, purchase_hist)


utility = params(1:n);
delta = params(end);

rmse_flm = 0;
rmse_mnl = 0;
rmse_mmnl = 0;
rmse_gam = 0;

len = tau;


for i = 1 : tau
    
    choice_set = purchase_hist{i, 1};   
    len = len + length(choice_set);
    choice = purchase_hist{i, 2}; 
    true_prob = zeros(1, n+1);
    
    if choice == 0
        true_prob(n+1) = 1;
    else
        true_prob(choice) = 1;
    end

    assortment = zeros(1, n);    
    assortment(choice_set) = 1;
    
    % for FLM
    temp = [];
    for ii = 1 : n
        if assortment(ranking(ii)) == 1
            temp = [temp, ranking(ii)];
        end
    end
    focal_idx = min(length(temp), focal_idx);
    focal_item = temp(focal_idx);
    flm_prob = zeros(1, n+1);
    flm_domi = 1 + sum(exp(utility) .* assortment') + delta * exp(utility(focal_item));
    
    for ii = 1 : length(choice_set)
        choice_temp = choice_set(ii);
        flm_prob(choice_temp) = exp(utility(choice_temp));
    end
    flm_prob(focal_item) = (1+delta) * flm_prob(focal_item);
    flm_prob(n+1) = 1;
    flm_prob = flm_prob / flm_domi;
        
    rmse_flm = rmse_flm + norm(flm_prob - true_prob)^2;
    
    % for MNL
    v_mnl = exp(x_mnl);
    mnl_prob = zeros(1, n+1);
    mnl_domi = 1 + sum(v_mnl .* assortment');
    for ii = 1 : length(choice_set)
        choice_temp = choice_set(ii);
        flm_prob(choice_temp) = v_mnl(choice_temp);
    end
    mnl_prob(n+1) = 1;
    mnl_prob = mnl_prob / mnl_domi;
    rmse_mnl = rmse_mnl + norm(mnl_prob - true_prob)^2;
    
    % for MMNL
%     mmnl_prob = zeros(1, n+1);
%     for ii = 1 : length(choice_set)
%         choice_temp = choice_set(ii);
%         prob_choice_temp = 0;
%         for jj = 1 : length(mmnl_gamma)
%             prob_choice_temp = prob_choice_temp + mmnl_gamma(jj) * mmnl_mnl(jj, choice_temp) / (1 + sum(mmnl_mnl(jj, choice_set)));
%         end
%         mmnl_prob(choice_temp) = prob_choice_temp;
%     end
%     
%     prob_choice_temp = 0;
%     for jj = 1 : length(mmnl_gamma)
%         prob_choice_temp = prob_choice_temp + mmnl_gamma(jj) * 1 / (1 + sum(mmnl_mnl(jj, choice_set)));
%     end
%     mmnl_prob(n+1) = prob_choice_temp;
%         
%     rmse_mmnl = rmse_mmnl + norm(mmnl_prob - true_prob)^2;
  
end

rmse_mnl = sqrt(rmse_mnl / len);
rmse_flm = sqrt(rmse_flm / len);
rmse_mmnl = sqrt(rmse_mmnl / len);


test_rmse = [rmse_mnl; rmse_gam; rmse_flm; rmse_mmnl];




end