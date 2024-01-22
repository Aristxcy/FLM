function [log_lik_train, rec_exitflag, customer_type_list, lambda_list, ...
     purchase_hist_train, purchase_hist_validation, v_mnl, v_gam, w_gam, ...
      focal_idx, ranking, v_flm, delta_flm, purchase_hist_test] = estimate_model(m, n, tau_train, tau_valid, tau_test, sample_idx)

    % stands for mnl, gam, and flm
    log_lik_train = zeros(3, 1);
    rec_exitflag = zeros(3, 1); 
 
    lambda_list = rand(m, 1);
    lambda_list = lambda_list / sum(lambda_list);

    %% generate customer types 
    temp_list = zeros(m, 1);
    iter = 1;
    
    % choice_index_prob_list = rand(2, 1);
    % choice_index_prob_list = choice_index_prob_list / sum(choice_index_prob_list);
    choice_index_set = 1:2; 
    choice_index_prob_list = [0.9; 0.1];
    customer_type_list = cell(m, 2);
    while(iter <= m)
        temp = randperm(n);
        len = length(temp);
        temp = fliplr(temp);
        temp_abbr = sum(temp.*(n+1).^(0:1:len-1)); % identifier of customer's type

        if( sum(temp_abbr==temp_list)==0 )
            customer_type_list{iter, 1} = temp;
            sample_choice_index = randsrc(1, 1, [choice_index_set; choice_index_prob_list']);    
            customer_type_list{iter, 2} = sample_choice_index; % index
            temp_list(iter) = temp_abbr;
            iter = iter + 1;
        else
            continue; % skip repeated customer type
        end
    end

   
    purchase_hist_train = cell(tau_train, 2);
    a =  1 : m;

    iter = 1;
    choice_list = [];
    while iter <= tau_train
        sample_m = randsrc(1, 1, [a; lambda_list']);
        cardi = randi(n);
        choice_set = 1:cardi;
        temp = customer_type_list{sample_m, 1};
        idx = find(sum(temp==choice_set', 1));
        if(isempty(idx))
            choice = 0;
        else
            choice = temp(idx( min(customer_type_list{sample_m, 2}, length(idx))));
        end
        purchase_hist_train{iter, 1} = choice_set;
        purchase_hist_train{iter, 2} = choice;
        choice_list = [choice_list; choice];
        iter = iter + 1; 
    end


    [A, ~] = check_compromise(purchase_hist_train, tau_train, n);

    A = A(:, 1:n) ./ A(:, n+1)

    purchase_hist_validation = cell(tau_valid, 2);
    iter = 1;
    while iter <= tau_valid
        sample_m = randsrc(1, 1, [a; lambda_list']);
        cardi = randi(n);
        choice_set = 1:cardi;
        temp = customer_type_list{sample_m, 1};
        idx = find(sum(temp==choice_set', 1));
        if(isempty(idx))
            choice = 0;
        else
            choice = temp(idx( min(customer_type_list{sample_m, 2}, length(idx))));
        end
        purchase_hist_validation{iter, 1} = choice_set;
        purchase_hist_validation{iter, 2} = choice;
        iter = iter + 1; 
    end
    
    purchase_hist_test = cell(tau_test, 2);
    iter = 1;
    while iter <= tau_test
        sample_m = randsrc(1, 1, [a; lambda_list']);
        cardi = randi(n);
        choice_set = 1:cardi;
        temp = customer_type_list{sample_m, 1};
        idx = find(sum(temp==choice_set', 1));
        if(isempty(idx))
            choice = 0;
        else
            choice = temp(idx( min(customer_type_list{sample_m, 2}, length(idx))));
        end
        purchase_hist_test{iter, 1} = choice_set;
        purchase_hist_test{iter, 2} = choice;
        iter = iter + 1; 
    end

    %% calibrate choice models
    % MNL
    options = optimoptions('fminunc', 'MaxFunctionEvaluations', 5e3);
    % x is the utility vector, set no purchase be 0
    [x, fval, exitflag, ~] = fminunc(@(x) -mnl_log_lik(x, n, tau_train, purchase_hist_train),...
                                        zeros(n,1), options);
    warm_start = x;
    v_mnl = exp(x);
    log_lik_train(1) = -fval;
    rec_exitflag(1) = exitflag;

    % gam model
    options = optimoptions('fminunc', 'MaxFunctionEvaluations', 5e3);
    [x, fval, exitflag, ~] = fminunc(@(x) -gam_log_lik(x, n, tau_train, purchase_hist_train),...
                                        [warm_start; zeros(n, 1)], options);
    v_gam = exp(x(1:n));
    w_gam = exp(x(n+1:end));
    log_lik_train(2) = -fval;
    rec_exitflag(2) = exitflag;

    % focal luce model - compromise effect
    [focal_idx, ranking, fval, exitflag, x] = cali_tau_model(n, tau_train, tau_valid, ...
                                                            purchase_hist_train, purchase_hist_validation, warm_start, choice_list);
    v_flm = exp(x(1:n));
    delta_flm = x(end);
    log_lik_train(3) = -fval;
    rec_exitflag(3) = exitflag;

    % MMNL model
    in_sample_transactions = [];
    for i = 1 : tau_train
        choice_set = purchase_hist_train{i, 1};
        choice = purchase_hist_train{i, 2};
        s.product = choice;
        s.offered_products = [0, choice_set];
        in_sample_transactions = [in_sample_transactions, s];
    end
    transactions.in_sample_transactions = in_sample_transactions;
    out_of_sample_transactions = []; 
    transactions.out_of_sample_transactions = out_of_sample_transactions;
    Q.transactions = transactions;

    numProds = n + 1;
    ground_model.products = linspace(1, numProds, numProds) - 1;

    ground_model.code = 'mnl';
    ground_model.etas = zeros(numProds-1, 1);

    Q.ground_model = ground_model;
    output = jsonencode(Q);
    fo = strcat('./ground_', num2str(tau_train), '_', num2str(sample_idx), '_train', '.json');
    fileID = fopen(fo, 'w');
    fprintf(fileID, output);
    fclose('all');
    
    %% 
    in_sample_transactions = [];
    for i = 1 : tau_valid
        choice_set = purchase_hist_validation{i, 1};
        choice = purchase_hist_validation{i, 2};
        s.product = choice;
        s.offered_products = [0, choice_set];
        in_sample_transactions = [in_sample_transactions, s];
    end
    transactions.in_sample_transactions = in_sample_transactions;
    out_of_sample_transactions = []; 
    transactions.out_of_sample_transactions = out_of_sample_transactions;
    Q.transactions = transactions;

    numProds = n + 1;
    ground_model.products = linspace(1, numProds, numProds) - 1;

    ground_model.code = 'mnl';
    ground_model.etas = zeros(numProds-1, 1);

    Q.ground_model = ground_model;
    output = jsonencode(Q);
    fo = strcat('./ground_', num2str(tau_valid), '_', num2str(sample_idx), '_validation', '.json');
    fileID = fopen(fo, 'w');
    fprintf(fileID, output);
    fclose('all');
end

