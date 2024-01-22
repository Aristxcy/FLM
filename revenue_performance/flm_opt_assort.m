function S_opt = flm_opt_assort(r, n, focal_idx, ranking, v_flm, delta_flm)

opt_rev = 0;

eps = 1e-6;

S_list = [];
for last_focal = 1 : n
    idx = find(ranking == last_focal);
    if idx > 1
        N_1 = ranking(1:idx-1);
        N_2 = ranking(idx+1:end);
        
        % case 1
        n_temp = length(N_1);
        v_flm_temp = v_flm(N_1);
       
        r_temp = r(N_1);
        
        C1 = (1+delta_flm) * r(last_focal) * v_flm(last_focal);
        C2 = 1 + (1+delta_flm) * v_flm(last_focal);
        cvx_begin quiet
            cvx_solver mosek
            variable x(n_temp, 1) nonnegative
            variable x0 nonnegative
            obj = r_temp' * x + C1 * x0;
            maximize obj
            subject to 
                sum(x) + C2 * x0 == 1;
                x./v_flm_temp <= x0;
                sum(x./v_flm_temp) <= (focal_idx - 1) * x0;
        cvx_end
        assortment = zeros(1, n);
        assortment(N_1(x>eps)) = 1;
        assortment(last_focal) = 1;
        S_list = [S_list; assortment];    
        
        % case 2
        if idx >= focal_idx
            N_2_sort = sort(N_2, 'ascend');

            for j = 1 : length(N_2)
                guess = N_2_sort(1:j);
                C1 = (1+delta_flm) * r(last_focal) * v_flm(last_focal) + sum(r(guess) .* v_flm(guess));
                C2 = 1 + (1+delta_flm) * v_flm(last_focal) + sum(v_flm(guess));
                cvx_begin quiet
                    cvx_solver mosek
                    variable x(n_temp, 1) nonnegative
                    variable x0 nonnegative
                    obj = r_temp' * x + C1 * x0;
                    maximize obj
                    subject to 
                        sum(x) + C2 * x0 == 1;
                        x./v_flm_temp <= x0;
                        sum(x./v_flm_temp) == (focal_idx - 1) * x0;
                cvx_end
                assortment = zeros(1, n);
                assortment(guess) = 1;
                assortment(last_focal) = 1;
                assortment(N_1(x>eps)) = 1;
                S_list = [S_list; assortment]; 
            end
        end
    else
        assortment = zeros(1, n);
        assortment(last_focal) = 1;
        S_list = [S_list; assortment];        
    end

end

S_list = unique(S_list, 'rows', 'stable');

for j = 1 : size(S_list, 1)
    assortment = S_list(j, :);
    rev = flm_assort_rev(assortment, r, focal_idx, ranking, v_flm, delta_flm);
    if rev > opt_rev
        opt_rev = rev;
        S_opt = assortment;
    end
end

% S_opt
% opt_rev

%% enumeration 
% whole_set = linspace(1, n, n);
% opt_rev = 0;
% 
% for cardi_iter = 1 : n
%     possible_S = nchoosek(whole_set, cardi_iter);
%     for k = 1:size(possible_S, 1)
%         S_without_no = possible_S(k, :);
%         assortment = zeros(1, n);
%         assortment(S_without_no) = 1;   
%         rev = flm_assort_rev(assortment, r, focal_idx, ranking, v_flm, delta_flm);
%         if rev > opt_rev
%             opt_rev = rev;
%             S_opt = assortment;
%         end
%     end
% end
% S_opt
% opt_rev

end
