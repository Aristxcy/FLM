function out = flm_single_assortment_loglikelihood(params, n, ranking, focal_idx, assortment, choice)


utility = params(1:n);
delta = params(end);

temp = [];
for ii = 1 : n
    if assortment(ranking(ii)) == 1
        temp = [temp, ranking(ii)];
    end
end

focal_idx = min(length(temp), focal_idx);
focal_item = temp(focal_idx);

if choice ~= 0
    purchase = zeros(1, n);
    purchase(choice) = 1;
    out = sum(purchase' .* utility) +  log(1+delta) * (focal_item == choice) ...
        - log(1 + sum(exp(utility) .* assortment') + delta * exp(utility(focal_item)));
else
    out = - log(1 + sum(exp(utility) .* assortment') + delta * exp(utility(focal_item)));
            
end