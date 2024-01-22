function S_opt = mmnl_opt_assort(r, n, mmnl_gamma, mmnl_mnl)

idx_set = linspace(1, n, n);
rev_opt = 0;

for cardi_iter = 1 : n
    possible_S = nchoosek(idx_set, cardi_iter);
    for k = 1:size(possible_S, 1)
        revenue = 0;
        S_without_no = possible_S(k, :);
        
%         for jj = 1 : length(mmnl_gamma)
%             type_revenue = sum(r(S_without_no)' .* mmnl_mnl(jj, S_without_no)) ...
%                 / (1 + sum(mmnl_mnl(jj, S_without_no)));
%             revenue = revenue + type_revenue * mmnl_gamma(jj);
%         end

        for jj = 1 : cardi_iter
            prob = 0;
            for ii = 1 : length(mmnl_gamma)
                prob = prob + mmnl_gamma(ii) * mmnl_mnl(ii, S_without_no(jj)) / (1 + sum(mmnl_mnl(ii, S_without_no)));
            end
            revenue = revenue + prob * r(S_without_no(jj));
        end
        
        if revenue > rev_opt
            rev_opt = revenue;
            S_opt = S_without_no;
        end
    end       
end

end
