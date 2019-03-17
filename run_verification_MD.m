
function res=run_verification_MD(m,d,eta)
%% tuning parameters
%eta= 1e-2; %learning rate
n=5000; %number of datapoints
%m = 350; % pivot update frequency
nit = 2000; % number or iterations
%d = 10; % problem dimension
sigma=0.1; %linear regression variance

%number of experiments to run
nrun=1000;



%% initialization

x = {};
tilde_x = {};
f_val ={};
f_val_pivot ={};

x0=randn(d,1);

for j = 1:nrun
    x{end+1} = zeros(d,nit);
    tilde_x{end+1} = zeros(d,nit);
    f_val{end+1} = zeros(1,nit);
    f_val_pivot{end+1} = zeros(1,nit);
    x{end}(:,1)=x0;
    tilde_x{end}(:,1)=x0;
end


%% getting data/problem definition
x_star = randn(d,1);
A = randn(n,d);
noise = sigma*randn(n,1);
y = A*x_star+noise;
x_sol = A\y; %getting the solution to the problem

%% learning
for r = 1:nrun
    for i = 2:nit
        index = randi([1,n]);
        grad_f_est = 2*A(index,:)'*(A(index,:)*x{r}(:,i-1)-y(index));
        grad_f_est_pivot = 2*A(index,:)'*(A(index,:)*tilde_x{r}(:,i-1)-y(index));
        grad_f_pivot = (2/n)*A'*(A*tilde_x{r}(:,i-1)-y); 
        x{r}(:,i) = x{r}(:,i-1)-eta*(grad_f_est-grad_f_est_pivot+grad_f_pivot);
        f_val{r}(i) = (1/n)*norm(x{r}(:,i)-x_sol)^2;
        %pivot update
        if mod(i,m)==0
            tilde_x{r}(:,i)=x{r}(:,i);
        else
            tilde_x{r}(:,i)=tilde_x{r}(:,i-1);
        end 
        f_val_pivot{r}(i) = (1/n)*norm(tilde_x{r}(:,i)-x_sol)^2;
    end
end

%% averaging
f_val_aver = zeros(1,nit);
f_val_pivot_aver = zeros(1,nit);

for i = 1:nit
    tmp1 = 0;
    tmp2 = 0;
    for r = 1:nrun
        tmp1 = tmp1+f_val{r}(:,i);
        tmp2 = tmp2+f_val_pivot{r}(:,i);
    end
    f_val_aver(i)=tmp1/nrun;
    f_val_pivot_aver(i)=tmp2/nrun;
end
res=max(f_val_pivot_aver./f_val_aver);


%% plotting averaged run
figure
for r =1:nrun
    semilogy(1:nit,f_val{r},'Linewidth',1,'Color',[91, 207, 244] / 255);hold on
end

semilogy(1:nit,f_val_aver,'Linewidth',2,'Color','b');hold on
xlabel('iteration')
ylabel('cost')

%% plotting staleness
figure
plot(1:nit,f_val_pivot_aver./f_val_aver,'Linewidth',2,'Color','b');hold on
title(max(f_val_pivot_aver./f_val_aver))
xlabel('iteration')
ylabel('cost')
end
