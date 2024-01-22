function [focal_idx, ranking, fval, exitflag, x] = cali_tau_model(n, tau_train, tau_valid, purchase_hist_train, purchase_hist_validation, warm_start, choice_list)

idxs = linspace(1, n, n);
all_permutation = perms(idxs);
[total_permutation, permutation_length] = size(all_permutation);

% permutation_list = [];
% for i = 1 : total_permutation
%     permutation = all_permutation(i, :);
%     for j = 2 : permutation_length
%         % identify the number of inversions
%         s = find(permutation(1:j-1) > permutation(j));
%         u(j) = length(s);
%         result = sum(u); 
%     end   
%     if result <= 4
%         permutation_list = [permutation_list; permutation];
%     end
% end

[~, choice_ranking] = sort(histcounts(choice_list), 'descend');
choice_ranking

if length(choice_ranking) < n
    difference_set = setdiff(1:n, choice_ranking);
    random_order = difference_set(randperm(length(difference_set)));
    choice_ranking = [choice_ranking, random_order];
end
    

% ranking_list = [];
% for i = 1 : size(permutation_list, 1)
%     permutation = permutation_list(i, :);
%     ranking = choice_ranking(permutation);
%     ranking_list = [ranking_list; ranking];
% end

ranking_list = [choice_ranking];
for i = 1 : total_permutation
    permutation = all_permutation(i, :);
    distance = kendall_tau_distance(permutation, choice_ranking);
    if distance <= 1
        ranking_list = [ranking_list; permutation];
    end
end
    

flm_params = zeros(n*2 + 2, 1);
valid_best = inf;
corresponding_train = inf;
options = optimset('Algorithm','sqp', 'MaxFunEvals', 1e4, 'Display', 'off');
for focal_idx = 2
    for rank_idx = 1 : size(ranking_list, 1)
        ranking = ranking_list(rank_idx, :);
        % params include the utility vector and the distortion constant
        f_flm = @(params) flm_total_loglikelihood(params, n, tau_train, ranking, focal_idx, purchase_hist_train);
        f_flm_valid = @(params) flm_total_loglikelihood(params, n, tau_valid, ranking, focal_idx, purchase_hist_validation);

        A = [zeros(n, n+1); zeros(1, n), -1];
        b = zeros(n+1, 1);

        lb = [];
        ub = [];

        [esti_flm_params, fvalTau, exitflag] = fmincon(f_flm, [warm_start; 1], A, b, [], [], lb, ub, [], options);
        fvalidflm_log = f_flm_valid(esti_flm_params);

        if fvalidflm_log < valid_best
            valid_best = fvalidflm_log;
            flm_params(1:n+1) = esti_flm_params;
            flm_params(n+2) = focal_idx;
            flm_params(n+3:end) = ranking';
            exitflag_final = exitflag;
            corresponding_train = fvalTau;
        end
    end
end

x = flm_params(1:n+1);
focal_idx = flm_params(n+2);
ranking = flm_params(n+3:end);
fval = corresponding_train;
exitflag = exitflag_final;

end