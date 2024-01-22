function log_lik = mmnl_log_lik(mmnl_gamma, mmnl_mnl, tau, purchase_hist)

    log_lik = 0;
    for i = 1:tau
        choice_set = purchase_hist{i,1};
        choice = purchase_hist{i,2};
        total = 0;
        
        if (choice==0)
            for jj = 1 : length(mmnl_gamma)
                total = total + mmnl_gamma(jj) * ( 1 / (1+sum(mmnl_mnl(jj, choice_set))));
            end
        else
            for jj = 1 : length(mmnl_gamma)
                total = total + mmnl_gamma(jj) * ( mmnl_mnl(jj, choice) / (1+sum(mmnl_mnl(jj, choice_set))));
            end
            
        end
        log_lik = log_lik + log(total);
    end
end