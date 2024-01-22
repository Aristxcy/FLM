function [S_opt, rev_opt] = gsp_opt_assort(customer_type_list, m, n, r, lambda_list)


idx_set = linspace(1, n, n);
rev_opt = 0;

for cardi_iter = 1 : n
    possible_S = nchoosek(idx_set, cardi_iter);
    for k = 1:size(possible_S, 1)
        revenue = 0;
        S_without_no = possible_S(k, :);
        for cus_type = 1 : m
            rank = customer_type_list{cus_type, 1};
            gsp_focal = customer_type_list{cus_type, 2};
            idx = find(sum(rank==S_without_no',1));
            if(~isempty(idx)) % if buying something
                choice = rank( idx( min( length(idx), gsp_focal ) ) );
                revenue = revenue + r(choice) * lambda_list;
            end  
        end
        
        if revenue > rev_opt
            rev_opt = revenue;
            S_opt = S_without_no;
            
        end
    end
        
end



end