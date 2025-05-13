function [C acc acc_perclass kappa kappaperclass]=kappaaccuracy(x,y)
%input:
%   x : ture label, should be a row vector
%   y : predict label, should be a row vector
%   
% output:
%   C: confusion matrix 
%   acc : testing sample accuracy
%   acc_perclass : testing sample accuracy per class
%   kappa : kappa coefficient of all class
%   kappaperclass : kappa coefficient of individual classes


% x1=[ 2 2 2 1 1 1 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7];
% y1=[ 2 2 2 2 1 1 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7];
% x=x1';
% y=y1';

%% confusion matrix
truelabel=max(x);
for i=1:truelabel
    mis{i}=y(find(x==i));
     for j=1:truelabel
         C(i,j)=length(find(mis{i}==j));
    end
end
%% testing sample accuracy 
acc=sum(diag(C))/length(y);
%% testing sample accuracy per class
for i=1:truelabel
    numberperclass(i,1)=length(find(x==i));
end
acc_perclass=diag(C)./numberperclass;
%% kappa coefficient of all class
NormalziedC=C./length(y);
p0=sum(diag(NormalziedC));
c1=sum(NormalziedC,1);
r=sum(NormalziedC,2);
pc=sum(c1.*r');
kappa=(p0-pc)/(1-pc);     

%% kappa coefficient of individual classes
for i=1:truelabel
    kappaperclass(1,i)=(NormalziedC(i,i)-r(i)*c1(i))/(r(i)-r(i)*c1(i));
end

       