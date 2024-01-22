function tau_distance = kendall_tau_distance(x, y)
    len = length(x);
    v = 0;

    for i = 1:len
        for j = i+1:len
            a = x(i) < x(j) && y(i) > y(j);
            b = x(i) > x(j) && y(i) < y(j);

            if a || b
                v = v + 1;
            end
        end
    end

    tau_distance = abs(v);
   
end
