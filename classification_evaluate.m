function [Kappa,PCC,Ca,A,x2]=classification_evaluate(confusion_matrix);
NN=sum(sum(confusion_matrix));%��������������������ܺ�
x1=sum(confusion_matrix);
x2=sum(confusion_matrix');
x3=x1*x2';
[row,col]=size(confusion_matrix);
for i=1:row
    A(i)=confusion_matrix(i,i);%�Խ���Ԫ�ؼ���ȷ����ĸ���
    Ca(i)=A(i)/x2(1,i);%����𾫶�
end
B=sum(A);
fz=NN*B-x3;
fm=NN^2-x3;
Kappa=fz/fm;%����Kappaϵ���Ķ���
PCC=B/NN;%���ྫ��