clear all
close all
clc

%% tuning parameters
eta= 1e-2; %learning rate
n =1000; %number of datapoints
m1 = 50; % pivot update frequency
m2 = 80; % pivot update frequency 2
nit = 700; % number or iterations
d = 10; % problem dimension
sigma = 0.1; %linear regression variance
nruns = 15; %number of runs
x0 = 10*randn(d,1); %x0

%% initialization
x1={};x2={};x3={};tilde_x3={};x4={};tilde_x4={};
f_val1={}; f_val2={}; f_val3={};f_val4={};
f1_avg = zeros(1,nit);
f2_avg = zeros(1,nit);
f3_avg = zeros(1,nit);
f4_avg = zeros(1,nit);

for i = 1:nruns
    x1{end+1} = zeros(d,nit);
    x2{end+1} = zeros(d,nit);
    x3{end+1} = zeros(d,nit);
    x4{end+1} = zeros(d,nit);
    tilde_x3{end+1} = zeros(d,nit);
    tilde_x4{end+1} = zeros(d,nit);
    f_val1{end+1} = zeros(1,nit);
    f_val2{end+1} = zeros(1,nit);
    f_val3{end+1} = zeros(1,nit); 
    f_val4{end+1} = zeros(1,nit);     
    x1{end}(:,1) = x0;
    x2{end}(:,1) = x0;
    x3{end}(:,1) = x0;   
    x4{end}(:,1) = x0; 
    tilde_x3{end}(:,1) = x0;   
    tilde_x4{end}(:,1) = x0; 
end

%% getting data/problem definition
x_star = randn(d,1); %solution to the problem
A = randn(n,d); %random data points
y = A*x_star+sigma*randn(n,1); %target variable
x_sol = A\y; %getting the numerical correct solution to the problem

disp(max(eig(A'*A)));

%% learning
for r = 1:nruns
    for i = 2:nit
        %GD
        grad_f = (2/n)*A'*(A*x1{r}(:,i-1)-y)+Dregu(x1{r}(:,i-1)-x_sol); 
        x1{r}(:,i) = x1{r}(:,i-1)-eta*grad_f;
        f_val1{r}(:,i) = (1/n)*(norm(A*x1{r}(:,i)-y)^2+regu(x1{r}(:,i)-x_sol))-(1/n)*(norm(A*x_sol-y)^2+regu(x_sol-x_sol));
        %f_val1{r}(:,i) = norm(x1{r}(:,i)-x_sol)^2;        

        %SGD
        index = randi([1,n]);
        grad_f_est = 2*A(index,:)'*(A(index,:)*x2{r}(:,i-1)-y(index))+Dregu(x2{r}(:,i-1)-x_sol); 
        x2{r}(:,i) = x2{r}(:,i-1)-eta*grad_f_est;
        f_val2{r}(:,i) = (1/n)*(norm(A*x2{r}(:,i)-y)^2+regu(x2{r}(:,i)-x_sol))-(1/n)*(norm(A*x_sol-y)^2+regu(x_sol-x_sol));
        %f_val2{r}(:,i) = norm(x2{r}(:,i)-x_sol)^2;        

        
        %SVRG
        index = randi([1,n]);
        grad_f_est = 2*A(index,:)'*(A(index,:)*x3{r}(:,i-1)-y(index))+Dregu(x3{r}(:,i-1)-x_sol);
        grad_f_est_pivot = 2*A(index,:)'*(A(index,:)*tilde_x3{r}(:,i-1)-y(index))+Dregu(tilde_x3{r}(:,i-1)-x_sol);
        grad_f_pivot = (2/n)*A'*(A*tilde_x3{r}(:,i-1)-y)+Dregu(tilde_x3{r}(:,i-1)-x_sol); 
        x3{r}(:,i) = x3{r}(:,i-1)-eta*(grad_f_est-grad_f_est_pivot+grad_f_pivot);
        f_val3{r}(:,i) = (1/n)*(norm(A*x3{r}(:,i)-y)^2+regu(x3{r}(:,i)-x_sol))-(1/n)*(norm(A*x_sol-y)^2+regu(x_sol-x_sol));
        %f_val3{r}(:,i) = norm(x3{r}(:,i)-x_sol)^2;        
        %pivot update
        if mod(i,m1)==0
            tilde_x3{r}(:,i)=x3{r}(:,i);
        else
            tilde_x3{r}(:,i)=tilde_x3{r}(:,i-1);
        end
        
        %SVRG 2
        index = randi([1,n]);
        grad_f_est = 2*A(index,:)'*(A(index,:)*x4{r}(:,i-1)-y(index))+Dregu(x4{r}(:,i-1)-x_sol); 
        grad_f_est_pivot = 2*A(index,:)'*(A(index,:)*tilde_x4{r}(:,i-1)-y(index))+Dregu(tilde_x4{r}(:,i-1)-x_sol); 
        grad_f_pivot = (2/n)*A'*(A*tilde_x4{r}(:,i-1)-y)+Dregu(tilde_x4{r}(:,i-1)-x_sol); 
        x4{r}(:,i) = x4{r}(:,i-1)-eta*(grad_f_est-grad_f_est_pivot+grad_f_pivot);
        f_val4{r}(:,i) = (1/n)*(norm(A*x4{r}(:,i)-y)^2+regu(x4{r}(:,i)-x_sol))-(1/n)*(norm(A*x_sol-y)^2+regu(x_sol-x_sol));
        %f_val4{r}(:,i) = norm(x4{r}(:,i)-x_sol)^2;        
        if mod(i,m2)==0
            tilde_x4{r}(:,i)=x4{r}(:,i);
        else
            tilde_x4{r}(:,i)=tilde_x4{r}(:,i-1);
        end              
        
    end 
end

%% averaging
for i = 1:nit
    for r = 1:nruns
        f1_avg(i)=f1_avg(i)+(1/nruns)*f_val1{r}(i);
        f2_avg(i)=f2_avg(i)+(1/nruns)*f_val2{r}(i);
        f3_avg(i)=f3_avg(i)+(1/nruns)*f_val3{r}(i);
        f4_avg(i)=f4_avg(i)+(1/nruns)*f_val4{r}(i);
    end
end


%% plotting
for r=1:nruns
    semilogy(1:nit,abs(f_val1{r}),'Linewidth',1,'Color','b');hold on
    semilogy(1:nit,abs(f_val2{r}),'Linewidth',1,'Color',[0.945 0.569 1.0]);hold on
    semilogy(1:nit,abs(f_val3{r}),'Linewidth',1,'Color',[0.52 0.73 0.572]);hold on
    semilogy(1:nit,abs(f_val4{r}),'Linewidth',1,'Color',[0.752 0.555 0.604]);hold on
end
h2=semilogy(1:nit,abs(f2_avg),'Linewidth',2,'Color','m');hold on
h3=semilogy(1:nit,abs(f3_avg),'Linewidth',2,'Color',[0.228 0.6 0.287]);hold on
h4=semilogy(1:nit,abs(f4_avg),'Linewidth',2,'Color',[0.541 0.137 0.102]);hold on
h1=semilogy(1:nit,abs(f1_avg),'Linewidth',2,'Color','b');hold on

xlabel('iteration','Fontsize',16)
ylabel('suboptimality','Fontsize',16)
l=legend([h1,h2,h3,h4],{'GD','MB-SGD, constant \eta','SVRG, m = 50','SVRG, m = 80'});
ylim([1e-10,1e4])
l.FontSize = 16;


