function y = regu(x)
    c=2;
    y= 1-exp(-0.5*c*norm(x)^2);
    %y=0.5*norm(x)^2;
end