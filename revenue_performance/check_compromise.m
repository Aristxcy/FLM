function [A, unique_choice_set_list]  = check_compromise(purchase_history, tau, n)



choice_set_list = zeros(tau, n);
for i = 1 : tau
    choice_set = purchase_history{i, 1};
    choice_set_list(i, choice_set) = 1;
end

unique_choice_set_list = unique(choice_set_list, 'rows');
unique_total = size(unique_choice_set_list, 1);

A = zeros(unique_total, n+1);


for i = 1 : tau
    choice_set = choice_set_list(i, :);
    [~, b] = ismember(choice_set, unique_choice_set_list, 'rows');
    choice = purchase_history{i, 2};
    A(b, choice) = A(b, choice) + 1;
    A(b, n+1) = A(b, n+1) + 1;
end
    
    





