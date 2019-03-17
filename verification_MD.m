clear all
close all
clc

d=1:20:81;

res1 = d;
for i = 1:length(d)
    res1(i)=run_verification_MD(d(i),10,1*1e-2);
end

res2 = d;
for i = 1:length(d)
    res2(i)=run_verification_MD(d(i),10,0.6*1e-2);
end

res3 = d;
for i = 1:length(d)
    res3(i)=run_verification_MD(d(i),10,0.2*1e-2);
end


figure;plot(d,res1,'-o','Linewidth',2);hold on
plot(d,res2,'-o','Linewidth',2);hold on
plot(d,res3,'-o','Linewidth',2);hold on

l=legend('\eta = 0.01','\eta = 0.006','\eta = 0.002','Location','Best');
l.FontSize = 16;
ylim([0.9,13])

xlabel('m','Fontsize',16)
