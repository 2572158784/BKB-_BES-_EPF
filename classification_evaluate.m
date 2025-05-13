function [Kappa,PCC,Ca,A,x2]=classification_evaluate(confusion_matrix);
NN=sum(sum(confusion_matrix));%分类总数，各列相加求总和
x1=sum(confusion_matrix);
x2=sum(confusion_matrix');
x3=x1*x2';
[row,col]=size(confusion_matrix);
for i=1:row
    A(i)=confusion_matrix(i,i);%对角线元素即正确分类的个数
    Ca(i)=A(i)/x2(1,i);%单类别精度
end
B=sum(A);
fz=NN*B-x3;
fm=NN^2-x3;
Kappa=fz/fm;%根据Kappa系数的定义
PCC=B/NN;%分类精度