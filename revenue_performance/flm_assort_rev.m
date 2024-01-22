function rev = flm_assort_rev(assortment, r, focal_idx, ranking, v_flm, delta_flm)

temp = [];
for ii = 1 : length(assortment)
    if assortment(ranking(ii)) == 1
        temp = [temp, ranking(ii)];
    end
end

focal_idx = min(size(temp, 2), focal_idx);

S = find(assortment == 1);
focal_item = temp(focal_idx);
numerator = sum(v_flm(S) .* r(S)) ...
    + delta_flm * v_flm(focal_item) * r(focal_item) ; % 分子
domi = 1 + sum(v_flm(S)) + delta_flm * v_flm(focal_item);
rev = numerator / domi;

end