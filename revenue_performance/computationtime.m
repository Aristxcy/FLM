clc; clear; close all;

focal_idx = 2;

%%
T_list_flm = zeros(1, 10);
iter = 0;
for n = 10:10:100
    r = sort(rand(n, 1), 'descend');
    ranking = randperm(n);
    v_flm = exp(rand(n, 1));
    delta_flm = 0.8;
    tic
    S_opt = flm_opt_assort(r, n, focal_idx, ranking, v_flm, delta_flm);
    T = toc;
    iter = iter + 1;
    T_list_flm(iter) = T;
end

%%
T_list_mnl = zeros(1, 10);
iter = 0;
for n = 10:10:100
    r = sort(rand(n, 1), 'descend');
    v_mnl = exp(rand(n, 1));
    whole_set = 1 : n;
    tic
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
    T = toc;
    iter = iter + 1;
    T_list_mnl(iter) = T;
end

%% 
n_list = 10:10:100;
loglog(n_list, T_list ./ T_list_mnl, '-', 'LineWidth', 2)
hold on
loglog(n_list, n_list .^ 2 , '-.', 'LineWidth', 2)

lgd = legend('Time Ratio', '$n^2$', 'Location', 'northwest', 'Box', 'Off', 'Interpreter', 'latex');
lgd.FontSize = 18;

xlabel('Number of Items', 'FontSize', 20, 'FontName', 'Times New Roman');

saveas(gcf, 'uniform_2.png');