function y = Dregu(x)
    c=2;
    y=c*exp(-0.5*c*norm(x)^2)*x;
    %y=x;
end