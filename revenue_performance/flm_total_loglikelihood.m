function loglikelihood = flm_total_loglikelihood(params, n, tau, ranking, focal_idx, purchase_hist)

loglikelihood = 0;

for i = 1 : tau
    
    choice_set = purchase_hist{i, 1};   
    choice = purchase_hist{i, 2}; 
    assortment = zeros(1, n);    
    assortment(choice_set) = 1;
    
    single = flm_single_assortment_loglikelihood(params, n, ranking, focal_idx, assortment, choice);
    loglikelihood = loglikelihood - single;

end