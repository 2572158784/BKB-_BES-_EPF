
% % %%%%%%%%%%%%%大学数据集  1尺度
tic
clc
close all
clear all
load paviaU.mat
yi=paviaU;HSI_image=paviaU;numband=103;bandnum=numband;s1=1;NP=9;

for h=1:numband
img=yi(:,:,s1);img=double(img);max0=max(max(img));Im=img/max0;ngrays=Im;Rimg=img(:,:,1);Gimg=img(:,:,1);
n=10; N_IterTotalR=2;r=rand;
rngrays=ngrays(:,:,1);nd=2;
LbR(1:1)=0.5;LbR(1:2)=0.9; UbR(1:1)=3;UbR(1:2)=20; 
nestR=zeros(n,2); 
fitnessR=1*ones(n,1); 
XFit=fitnessR;
nest11 = zeros(1, n);
% % 循环生成n个随机数
for f= 1:n
   nest11(f) = 0.5 + (0.9-0.5) * rand;
end
nest22= randperm(n)';nestR=[nest11',nest22];   
T= N_IterTotalR;XPos=nestR;
% %% ------------ Select the optimal fitness value--------------%
             for i=1:size(nestR,1),
                yr=fobj_BKA(rngrays,nestR(i,1),nestR(i,2));
                y(:,:,1)=yr; fnew=Calc_MSE(Im,y);
                if fnew<=fitnessR(i),
                    fitnessR(i)=fnew;
                    nest(i,:)=nestR(i,:);
                end
            [fmaxR,K]=min(fitnessR);bestnestR=nestR(K,:);
            XPosNew=nestR;XPos=nest;  XFit_New=fnew;XFit=fitnessR;  
            XLeader_Pos=bestnestR; XLeader_Fit=fmaxR; 
%% -------------------Migration behavior-------------------%
       m=2*sin(r+pi/2); s = randi([1,n],1); r_XFitness=XFit(s);dim=n;
       ori_value = rand(1,n);cauchy_value = tan((ori_value-0.5)*pi);
       lb= LbR;ub=UbR;
        if XFit(i)< r_XFitness
            XPosNew(i,:)=XPos(i,:)+cauchy_value(:,dim).* (XPos(i,:)-XLeader_Pos);
        else
            XPosNew(i,:)=XPos(i,:)+cauchy_value(:,dim).* (XLeader_Pos-m.*XPos(i,:));
        end
        new_nestR=XPosNew;
 %% --------------  Select the optimal fitness value---------    
                yr=fobj_BKA(rngrays,new_nestR(i,1),new_nestR(i,2));
                y(:,:,1)=yr;fnew=Calc_MSE(Im,y);
                if fnew<=fitnessR(i),
                    fitnessR(i)=fnew;
                    nest(i,:)=new_nestR(i,:);
                end
                [fmax1R,K]=min(fitnessR);bestnestR=nestR(K,:);
                XPosNew=nestR;XPos=nest;  XFit_New=fnew;XFit=fitnessR;  
             end             
%% -------Update the optimal Black-winged Kite----------%
    if(fmax1R<fmaxR)
        Best_Fitness_BKA=fmax1R;
        Best_Pos_BKA=bestnestR;
    end
denosing_Im=fobj_BEEPS(rngrays,bestnestR(1,1),bestnestR(1,2));
BestnestR05(s1,:)=bestnestR;
Y05_PU(:,:,s1)=denosing_Im;
s1=s1+1;
end

load paviaU_gt.mat;X=paviaU_gt;
for n=1:numband
pattern= Y05_PU(:,:,n);
pattern1=pattern(find(X(:,:)==1));%玉米、大豆、干草、林地、牧场、草地
pattern2=pattern(find(X(:,:)==2));%玉米、大豆、干草、林地、牧场、草地
pattern3=pattern(find(X(:,:)==3));%玉米、大豆、干草、林地、牧场、草地
pattern4=pattern(find(X(:,:)==4));%玉米、大豆、干草、林地、牧场、草地
pattern5=pattern(find(X(:,:)==5));%玉米、大豆、干草、林地、牧场、草地
pattern6=pattern(find(X(:,:)==6));%玉米、大豆、干草、林地、牧场、草地
pattern7=pattern(find(X(:,:)==7));%玉米、大豆、干草、林地、牧场、草地
pattern8=pattern(find(X(:,:)==8));%玉米、大豆、干草、林地、牧场、草地
pattern9=pattern(find(X(:,:)==9));%玉米、大豆、干草、林地、牧场、草地
patterns_in(n,:)=[pattern1',pattern2',pattern3',pattern4',pattern5',pattern6',pattern7',pattern8',pattern9'];
end
Feature=patterns_in';
% % % % % define the matrix for the coordinates of the patterns that are to be classified
[R1,C1]=find(X(:,:)==1);co1=[R1,C1]';[R2,C2]=find(X(:,:)==2);co2=[R2,C2]';[R3,C3]=find(X(:,:)==3);co3=[R3,C3]';
[R4,C4]=find(X(:,:)==4);co4=[R4,C4]';[R5,C5]=find(X(:,:)==5);co5=[R5,C5]';[R6,C6]=find(X(:,:)==6);co6=[R6,C6]';
[R7,C7]=find(X(:,:)==7);co7=[R7,C7]';[R8,C8]=find(X(:,:)==8);co8=[R8,C8]';[R9,C9]=find(X(:,:)==9);co9=[R9,C9]';
coordinates=[co1,co2,co3,co4,co5,co6,co7,co8,co9]; %存放位置坐标，两行六列，第一行各类样本的行，第二行为列
% %%%%%%%%%%准备将每一类样本集单独提取出来便于选取训练样本
[L1,P]=size(pattern1);[L2,P]=size(pattern2);[L3,P]=size(pattern3);[L4,P]=size(pattern4);[L5,P]=size(pattern5);[L6,P]=size(pattern6);
[L7,P]=size(pattern7);[L8,P]=size(pattern8);[L9,P]=size(pattern9);
b1=1*ones(L1,1);b2=2*ones(L2,1);b3=3*ones(L3,1);b4=4*ones(L4,1);b5=5*ones(L5,1);
b6=6*ones(L6,1);b7=7*ones(L7,1);b8=8*ones(L8,1);b9=9*ones(L9,1);
Label=[b1;b2;b3;b4;b5;b6;b7;b8;b9];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%LBP特征
% % % % % %%%%%%%%%%BLS
patterns_in1=patterns_in(:,1:L1);patterns_in2=patterns_in(:,L1+1:L1+L2);patterns_in3=patterns_in(:,L1+L2+1:L1+L2+L3);
patterns_in4=patterns_in(:,L1+L2+L3+1:L1+L2+L3+L4);patterns_in5=patterns_in(:,L1+L2+L3+L4+1:L1+L2+L3+L4+L5);
patterns_in6=patterns_in(:,L1+L2+L3+L4+L5+1:L1+L2+L3+L4+L5+L6);
patterns_in7=patterns_in(:,L1+L2+L3+L4+L5+L6+1:L1+L2+L3+L4+L5+L6+L7);
patterns_in8=patterns_in(:,L1+L2+L3+L4+L5+L6+L7+1:L1+L2+L3+L4+L5+L6+L7+L8);
patterns_in9=patterns_in(:,L1+L2+L3+L4+L5+L6+L7+L8+1:L1+L2+L3+L4+L5+L6+L7+L8+L9);
% % % %%%%%%%%%%%%%%均匀选取测试样本
 nn=0.01; 
nu1=ceil(L1*nn);nu2=ceil(L2*nn);nu3=ceil(L3*nn);nu4=ceil(L4*nn);nu5=ceil(L5*nn);nu6=ceil(L6*nn);nu7=ceil(L7*nn);nu8=ceil(L8*nn); 
nu9=ceil(L9*nn); 

a1 = randperm(L1);train_patterns1 = patterns_in1(:,a1(1:nu1));test_patterns1 = patterns_in1(:,a1(1:end));
a2 = randperm(L2);train_patterns2 = patterns_in2(:,a2(1:nu2));test_patterns2 = patterns_in2(:,a2(1:end));
a3 = randperm(L3);train_patterns3 = patterns_in3(:,a3(1:nu3));test_patterns3= patterns_in3(:,a3(1:end));
a4 = randperm(L4);train_patterns4 = patterns_in4(:,a4(1:nu4));test_patterns4= patterns_in4(:,a4(1:end));
a5 = randperm(L5);train_patterns5 = patterns_in5(:,a5(1:nu5));test_patterns5 = patterns_in5(:,a5(1:end));
a6 = randperm(L6);train_patterns6 = patterns_in6(:,a6(1:nu6));test_patterns6= patterns_in6(:,a6(1:end));
a7 = randperm(L7);train_patterns7 = patterns_in7(:,a7(1:nu7));test_patterns7 = patterns_in7(:,a7(1:end));
a8 = randperm(L8);train_patterns8 = patterns_in8(:,a8(1:nu8));test_patterns8 = patterns_in8(:,a8(1:end));
a9 = randperm(L9);train_patterns9 = patterns_in9(:,a9(1:nu9));test_patterns9 = patterns_in9(:,a9(1:end));
train_x=[train_patterns1,train_patterns2,train_patterns3,train_patterns4,train_patterns5,train_patterns6,train_patterns7,train_patterns8,train_patterns9]';
[hang0,lie0]=size(train_x);
[z,k0]=size(train_patterns1);[z,k1]=size(train_patterns2);[z,k2]=size(train_patterns3);[z,k3]=size(train_patterns4);
[z,k4]=size(train_patterns5);[z,k5]=size(train_patterns6);[z,k6]=size(train_patterns7);[z,k7]=size(train_patterns8);
[z,k8]=size(train_patterns9);
train_y1=zeros(hang0,1);train_y1(1:k0,1)=1;train_y2=zeros(hang0,1);train_y2(k0+1:k0+k1,1)=1;
train_y3=zeros(hang0,1);train_y3(k0+k1+1:k0+k1+k2,1)=1;train_y4=zeros(hang0,1);train_y4(k0+k1+k2+1:k0+k1+k2+k3,1)=1;
train_y5=zeros(hang0,1);train_y5(k0+k1+k2+k3+1:k0+k1+k2+k3+k4,1)=1;train_y6=zeros(hang0,1);train_y6(k0+k1+k2+k3+k4+1:k0+k1+k2+k3+k4+k5,1)=1;
train_y7=zeros(hang0,1);train_y7(k0+k1+k2+k3+k4+k5+1:k0+k1+k2+k3+k4+k5+k6,1)=1;
train_y8=zeros(hang0,1);train_y8(k0+k1+k2+k3+k4+k5+k6+1:k0+k1+k2+k3+k4+k5+k6+k7,1)=1;
train_y9=zeros(hang0,1);train_y9(k0+k1+k2+k3+k4+k5+k6+k7+1:k0+k1+k2+k3+k4+k5+k6+k7+k8,1)=1;
train_y=[train_y1,train_y2,train_y3,train_y4,train_y5,train_y6,train_y7,train_y8,train_y9];
% % %%%%%%%%%%%%%%%%%%%%%%%%%%选取测试样本
test_x=[test_patterns1,test_patterns2,test_patterns3,test_patterns4,test_patterns5,test_patterns6,test_patterns7,test_patterns8,test_patterns9]';
% test_x=[patterns_in1,patterns_in2,patterns_in3,patterns_in4,patterns_in5,patterns_in6,patterns_in7,patterns_in8,patterns_in9,patterns_in10,patterns_in11,patterns_in12,patterns_in13,patterns_in14,patterns_in15,patterns_in16]';
[hang,lie]=size(test_x);ntest=hang;
[z,z0]=size(test_patterns1);[z,z1]=size(test_patterns2);[z,z2]=size(test_patterns3);[z,z3]=size(test_patterns4);
[z,z4]=size(test_patterns5);[z,z5]=size(test_patterns6);[z,z6]=size(test_patterns7);[z,z7]=size(test_patterns8);
[z,z8]=size(test_patterns9);
test_y1=zeros(hang,1);test_y1(1:z0,1)=1;test_y2=zeros(hang,1);test_y2(z0+1:z0+z1,1)=1;
test_y3=zeros(hang,1);test_y3(z0+z1+1:z0+z1+z2,1)=1;test_y4=zeros(hang,1);test_y4(z0+z1+z2+1:z0+z1+z2+z3,1)=1;
test_y5=zeros(hang,1);test_y5(z0+z1+z2+z3+1:z0+z1+z2+z3+z4,1)=1;test_y6=zeros(hang,1);test_y6(z0+z1+z2+z3+z4+1:z0+z1+z2+z3+z4+z5,1)=1;
test_y7=zeros(hang,1);test_y7(z0+z1+z2+z3+z4+z5+1:z0+z1+z2+z3+z4+z5+z6,1)=1;
test_y8=zeros(hang,1);test_y8(z0+z1+z2+z3+z4+z5+z6+1:z0+z1+z2+z3+z4+z5+z6+z7,1)=1;
test_y9=zeros(hang,1);test_y9(z0+z1+z2+z3+z4+z5+z6+z7+1:z0+z1+z2+z3+z4+z5+z6+z7+z8,1)=1;
test_y=[test_y1,test_y2,test_y3,test_y4,test_y5,test_y6,test_y7,test_y8,test_y9];
%%%%%%%%%%%%%%%%%%%shot structrue with fine tuning under BP algorithm%%%%%%%%%%%%%%%%%%%%%%%%
% 
C = 2^-20; s = 0.8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
N11=18 ;%feature nodes  per window
N2=6;% number of windows of feature nodes
N33=400;% number of enhancement nodes
epochs=1;
train_err=zeros(1,epochs);test_err=zeros(1,epochs);
train_time=zeros(1,epochs);test_time=zeros(1,epochs);
N1=N11; N3=N33;  
train_x = zscore(train_x')';
H1 = [train_x .1 * ones(size(train_x,1),1)];y=zeros(size(train_x,1),N2*N1);
for i=1:N2
    we=2*rand(size(train_x,2)+1,N1)-1;
    We{i}=we;
    A1 = H1 * we;A1 = mapminmax(A1);
%     clear we;
beta1  =  sparse_bls(A1,H1,1e-3,100)';
beta11{i}=beta1;
T1 = H1 * beta1;
fprintf(1,'Feature nodes in window %f: Max Val of Output %f Min Val %f\n',i,max(T1(:)),min(T1(:)));
[T1,ps1]  =  mapminmax(T1',0,1);T1 = T1';
ps(i)=ps1;
y(:,N1*(i-1)+1:N1*i)=T1;
end
%%%%%%%%%%%%%enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H2 = [y .1 * ones(size(y,1),1)];
if N1*N2>=N3
     wh=orth(2*rand(N2*N1+1,N3)-1);
else
    wh=orth(2*rand(N2*N1+1,N3)'-1)'; 
end
T2 = H2 *wh;
l2 = max(max(T2));
l2 = s/l2;
fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',l2,min(T2(:)));
T2 = tansig(T2 * l2);
T3=[y T2];
% clear H2;clear T2;
beta = (T3'  *  T3+eye(size(T3',1)) * (C)) \ ( T3'  *  train_y);
Training_time = toc;
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
%%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%
xx = T3 * beta;
yy = result(xx);
train_yy = result(train_y);
TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);
disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
%%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
test_x = zscore(test_x')';
HH1 = [test_x .1 * ones(size(test_x,1),1)];
%clear test_x;
yy1=zeros(size(test_x,1),N2*N1);
for i=1:N2
    beta1=beta11{i};ps1=ps(i);
    TT1 = HH1 * beta1;
    TT1  =  mapminmax('apply',TT1',ps1)';
yy1(:,N1*(i-1)+1:N1*i)=TT1;
end
HH2 = [yy1 .1 * ones(size(yy1,1),1)]; 
TT2 = tansig(HH2 * wh * l2);TT3=[Feature TT2];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ratio = nn;
[Xtrain Ytrain Xtest Ytest Train_co Test_co] = splitData_ratio119(TT3, Label, ratio, coordinates);
% % % % % % % % % %% SVM classifier%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CVC =2.^(8); CVSigma=2.^(2);
[SVMoutlabel, dvalue, SVM_Time] = SVMclassifier(Xtrain, Ytrain, Xtest, Ytest, CVC, CVSigma);   
[SVM.CM SVM.acc SVM.accperclass SVM.kappa SVM.kappaperclass] = kappaaccuracy(SVMoutlabel,Ytest);
[Kappa,PCC,Ca,A,x2]=classification_evaluate(SVM.CM);
Kappa=Kappa*100; Kappa=roundn(Kappa,-2);
PCC=PCC*100;PCC=roundn(PCC,-2);
Ca=Ca*100;Ca=roundn(Ca,-2);AA=sum(Ca)/NP;
s=[Ca,Kappa,AA,PCC]; 
ss=[Kappa,AA,PCC]; 
CLASS=zeros(610,340,NP);CLASS_1=zeros(610,340); 
for k=1:NP
fff=dvalue(:,k); fff= fff';
 for i=1:L1
  CLASS_1(R1(i,1),C1(i,1))=fff(:,i);
 end
 for i=1:L2
  FFFF=fff(:,L1+1:L1+L2);  
  CLASS_1(R2(i,1),C2(i,1))=FFFF(:,i);
 end
  for i=1:L3
   FFFF=fff(:,L1+L2+1:L1+L2+L3);     
  CLASS_1(R3(i,1),C3(i,1))=FFFF(:,i);   
 end
  for i=1:L4
  FFFF=fff(:,L1+L2+L3+1:L1+L2+L3+L4);     
  CLASS_1(R4(i,1),C4(i,1))=FFFF(:,i);    
 end
  for i=1:L5
   FFFF=fff(:,L1+L2+L3+L4+1:L1+L2+L3+L4+L5);     
  CLASS_1(R5(i,1),C5(i,1))=FFFF(:,i);    
  end
  for i=1:L6
   FFFF=fff(:,L1+L2+L3+L4+L5+1:L1+L2+L3+L4+L5+L6);      
  CLASS_1(R6(i,1),C6(i,1))=FFFF(:,i);   
 end
  for i=1:L7
  FFFF=fff(:,L1+L2+L3+L4+L5+L6+1:L1+L2+L3+L4+L5+L6+L7); 
  CLASS_1(R7(i,1),C7(i,1))=FFFF(:,i); 
  end
  for i=1:L8
  FFFF=fff(:,L1+L2+L3+L4+L5+L6+L7+1:L1+L2+L3+L4+L5+L6+L7+L8); 
  CLASS_1(R8(i,1),C8(i,1))=FFFF(:,i);    
 end
  for i=1:L9
  FFFF=fff(:,L1+L2+L3+L4+L5+L6+L7+L8+1:L1+L2+L3+L4+L5+L6+L7+L8+L9); 
  CLASS_1(R9(i,1),C9(i,1))=FFFF(:,i);    
  end    
CLASS(:,:,k)=CLASS_1;
end
s2=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s1=1;
for h=1:NP
img=CLASS(:,:,h);
% img=double(img); max0=max(max(img));img=img/max0;
Rimg=img(:,:,1);Gimg=img(:,:,1);
  % 初始化参数
  [row,column]=size(Rimg);
  tempimg1=zeros(row,column);  tempimg2=zeros(row,column);
  tempimg3=zeros(row,column);  tempimg4=zeros(row,column);
  tempimg5=zeros(row,column);  tempimg6=zeros(row,column);
  len=numel(Rimg(:));
  Psi1=zeros(1,len); Phi1=zeros(1,len); Psi2=zeros(1,len); Phi2=zeros(1,len);
  X1=zeros(1,len);X2=zeros(1,len);Y1=zeros(1,len);Y2=zeros(1,len);
 lambda=0.6;
 sigma=4;
 for i=1:row
     for j=1:column
         X1((i-1)*column+j)=Rimg(i,j);
         X2((i-1)*column+j)=Gimg(i,j);
      end
 end
 Psi1(1)=X1(1); Phi1(len)=X1(len);  Psi2(1)=X2(1);Phi2(len)=X2(len);
  for i=2:len
     Psi1(i)=(1-lambda*exp((-(X1(i)-Psi1(i-1))^2)/(2*sigma^2)))*X1(i)+lambda*exp((-(X1(i)-Psi1(i-1))^2)/(2*sigma^2))*Psi1(i-1);
     Psi2(i)=(1-lambda*exp((-(X2(i)-Psi2(i-1))^2)/(2*sigma^2)))*X2(i)+lambda*exp((-(X2(i)-Psi2(i-1))^2)/(2*sigma^2))*Psi2(i-1);
 end
  for i=(len-1):-1:1
      Phi1(i)=(1-lambda*exp((-(X1(i)-Phi1(i+1))^2)/(2*sigma^2)))*X1(i)+lambda*exp((-(X1(i)-Phi1(i+1))^2)/(2*sigma^2))*Phi1(i+1);
      Phi2(i)=(1-lambda*exp((-(X2(i)-Phi2(i+1))^2)/(2*sigma^2)))*X2(i)+lambda*exp((-(X2(i)-Phi2(i+1))^2)/(2*sigma^2))*Phi2(i+1);
  end
  for i=1:len
      Y1(i)=(Psi1(i)-(1-lambda)*X1(i)+Phi1(i))/(1+lambda);
      Y2(i)=(Psi2(i)-(1-lambda)*X2(i)+Phi2(i))/(1+lambda);
  end
  for i=1:row
      for j=1:column
          tempimg1(i,j)=Y1((i-1)*column+j);
          tempimg3(i,j)=Y2((i-1)*column+j);    
      end
 end
 % 2.vertical processing
  for j=1:column
     for i=1:row
        X1((j-1)*row+i)=tempimg1(i,j);
        X2((j-1)*row+i)=tempimg3(i,j);
     end
  end
  Psi1(1)=X1(1); Phi1(len)=X1(len); Psi2(1)=X2(1); Phi2(len)=X2(len);
  for i=2:len
     Psi1(i)=(1-lambda*exp((-(X1(i)-Psi1(i-1))^2)/(2*sigma^2)))*X1(i)+lambda*exp((-(X1(i)-Psi1(i-1))^2)/(2*sigma^2))*Psi1(i-1);
     Psi2(i)=(1-lambda*exp((-(X2(i)-Psi2(i-1))^2)/(2*sigma^2)))*X2(i)+lambda*exp((-(X2(i)-Psi2(i-1))^2)/(2*sigma^2))*Psi2(i-1);
 end
 for i=(len-1):-1:1
    Phi1(i)=(1-lambda*exp((-(X1(i)-Phi1(i+1))^2)/(2*sigma^2)))*X1(i)+lambda*exp((-(X1(i)-Phi1(i+1))^2)/(2*sigma^2))*Phi1(i+1);
    Phi2(i)=(1-lambda*exp((-(X2(i)-Phi2(i+1))^2)/(2*sigma^2)))*X2(i)+lambda*exp((-(X2(i)-Phi2(i+1))^2)/(2*sigma^2))*Phi2(i+1);
 end
 for i=1:len
    Y1(i)=(Psi1(i)-(1-lambda)*X1(i)+Phi1(i))/(1+lambda);
    Y2(i)=(Psi2(i)-(1-lambda)*X2(i)+Phi2(i))/(1+lambda);
 end
 for j=1:column
     for i=1:row
         tempimg1(i,j)=Y1((j-1)*row+i);
         tempimg3(i,j)=Y2((j-1)*row+i);
     end
 end
 % vertical-horizon
 for j=1:column
    for i=1:row
         X1((j-1)*row+i)=Rimg(i,j);
         X2((j-1)*row+i)=Gimg(i,j);
     end
 end
 Psi1(1)=X1(1); Phi1(len)=X1(len);Psi2(1)=X2(1); Phi2(len)=X2(len);
 for i=2:len
     Psi1(i)=(1-lambda*exp((-(X1(i)-Psi1(i-1))^2)/(2*sigma^2)))*X1(i)+lambda*exp((-(X1(i)-Psi1(i-1))^2)/(2*sigma^2))*Psi1(i-1);
     Psi2(i)=(1-lambda*exp((-(X2(i)-Psi2(i-1))^2)/(2*sigma^2)))*X2(i)+lambda*exp((-(X2(i)-Psi2(i-1))^2)/(2*sigma^2))*Psi2(i-1);
 end
 for i=(len-1):-1:1
     Phi1(i)=(1-lambda*exp((-(X1(i)-Phi1(i+1))^2)/(2*sigma^2)))*X1(i)+lambda*exp((-(X1(i)-Phi1(i+1))^2)/(2*sigma^2))*Phi1(i+1);
     Phi2(i)=(1-lambda*exp((-(X2(i)-Phi2(i+1))^2)/(2*sigma^2)))*X2(i)+lambda*exp((-(X2(i)-Phi2(i+1))^2)/(2*sigma^2))*Phi2(i+1);
 end
 for i=1:len
    Y1(i)=(Psi1(i)-(1-lambda)*X1(i)+Phi1(i))/(1+lambda);
    Y2(i)=(Psi2(i)-(1-lambda)*X2(i)+Phi2(i))/(1+lambda);
 end
 for j=1:column
    for i=1:row
         tempimg2(i,j)=Y1((j-1)*row+i);
         tempimg4(i,j)=Y2((j-1)*row+i);
     end
 end 
 % 2.horizon processing
 for i=1:row
     for j=1:column
         X1((i-1)*column+j)=tempimg2(i,j);
         X2((i-1)*column+j)=tempimg4(i,j);
     end
 end
 Psi1(1)=X1(1);Phi1(len)=X1(len); Psi2(1)=X2(1); Phi2(len)=X2(len);
 for i=2:len
     Psi1(i)=(1-lambda*exp((-(X1(i)-Psi1(i-1))^2)/(2*sigma^2)))*X1(i)+lambda*exp((-(X1(i)-Psi1(i-1))^2)/(2*sigma^2))*Psi1(i-1);
     Psi2(i)=(1-lambda*exp((-(X2(i)-Psi2(i-1))^2)/(2*sigma^2)))*X2(i)+lambda*exp((-(X2(i)-Psi2(i-1))^2)/(2*sigma^2))*Psi2(i-1);
 end
 for i=(len-1):-1:1
     Phi1(i)=(1-lambda*exp((-(X1(i)-Phi1(i+1))^2)/(2*sigma^2)))*X1(i)+lambda*exp((-(X1(i)-Phi1(i+1))^2)/(2*sigma^2))*Phi1(i+1);
     Phi2(i)=(1-lambda*exp((-(X2(i)-Phi2(i+1))^2)/(2*sigma^2)))*X2(i)+lambda*exp((-(X2(i)-Phi2(i+1))^2)/(2*sigma^2))*Phi2(i+1);
 end
 for i=1:len
     Y1(i)=(Psi1(i)-(1-lambda)*X1(i)+Phi1(i))/(1+lambda);
     Y2(i)=(Psi2(i)-(1-lambda)*X2(i)+Phi2(i))/(1+lambda);
 end
 for i=1:row
     for j=1:column
         tempimg2(i,j)=Y1((i-1)*column+j);
         tempimg4(i,j)=Y2((i-1)*column+j);
     end
 end
tempimg7=(tempimg1+tempimg2)/2;tempimg8=(tempimg3+tempimg4)/2;
img119=tempimg7; 
I_enhanced119(:,:,s1)=img119;
s1=s1+1;
end
clear X;load paviaU_gt.mat;X=paviaU_gt;XX=X;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%read ground truth from the file
for n=1:NP
pattern00=I_enhanced119(:,:,n);
pa1=pattern00(find(XX(:,:)==1));%玉米、大豆、干草、林地、牧场、草地
pa2=pattern00(find(XX(:,:)==2));
pa3=pattern00(find(XX(:,:)==3));
pa4=pattern00(find(XX(:,:)==4));%玉米、大豆、干草、林地、牧场、草地
pa5=pattern00(find(XX(:,:)==5));
pa6=pattern00(find(XX(:,:)==6));
pa7=pattern00(find(XX(:,:)==7));%玉米、大豆、干草、林地、牧场、草地
pa8=pattern00(find(XX(:,:)==8));
pa9=pattern00(find(XX(:,:)==9));
patterns_in1111(n,:)=[pa1',pa2',pa3',pa4',pa5',pa6',pa7',pa8',pa9'];
end
[n,ntest]=size(patterns_in1111);patterns_in1111=patterns_in1111';
for m=1:length(patterns_in1111)
    MAXL(m,:)=max(patterns_in1111(m,:));
    [hang,lie]=find(patterns_in1111(m,:)== MAXL(m,:));
    fenlei011(m,1)=min(lie);
end
fenlei=fenlei011';
label1=fenlei(1:z0); label2=fenlei(z0+1:z0+z1);label3=fenlei(z0+z1+1:z0+z1+z2);
 label4=fenlei(z0+z1+z2+1:z0+z1+z2+z3); label5=fenlei(z0+z1+z2+z3+1:z0+z1+z2+z3+z4);
 label6=fenlei(z0+z1+z2+z3+z4+1:z0+z1+z2+z3+z4+z5);
 label7=fenlei(z0+z1+z2+z3+z4+z5+1:z0+z1+z2+z3+z4+z5+z6);
 label8=fenlei(z0+z1+z2+z3+z4+z5+z6+1:z0+z1+z2+z3+z4+z5+z6+z7);
 label9=fenlei(z0+z1+z2+z3+z4+z5+z6+z7+1:z0+z1+z2+z3+z4+z5+z6+z7+z8);%  
% %%%%%%%%%Compute the Confusion Matrix
 M=zeros(9,9);
 M(1,:)=[length(find(label1==1)) length(find(label1==2)) length(find(label1==3)) length(find(label1==4)) length(find(label1==5)) length(find(label1==6)) length(find(label1==7)) length(find(label1==8)) length(find(label1==9))  ];
 M(2,:)=[length(find(label2==1)) length(find(label2==2)) length(find(label2==3)) length(find(label2==4)) length(find(label2==5)) length(find(label2==6)) length(find(label2==7)) length(find(label2==8)) length(find(label2==9))  ];
 M(3,:)=[length(find(label3==1)) length(find(label3==2)) length(find(label3==3)) length(find(label3==4)) length(find(label3==5)) length(find(label3==6)) length(find(label3==7)) length(find(label3==8)) length(find(label3==9))  ];
 M(4,:)=[length(find(label4==1)) length(find(label4==2)) length(find(label4==3)) length(find(label4==4)) length(find(label4==5)) length(find(label4==6)) length(find(label4==7)) length(find(label4==8)) length(find(label4==9)) ];
 M(5,:)=[length(find(label5==1)) length(find(label5==2)) length(find(label5==3)) length(find(label5==4)) length(find(label5==5)) length(find(label5==6)) length(find(label5==7)) length(find(label5==8)) length(find(label5==9))  ];
 M(6,:)=[length(find(label6==1)) length(find(label6==2)) length(find(label6==3)) length(find(label6==4)) length(find(label6==5)) length(find(label6==6)) length(find(label6==7)) length(find(label6==8)) length(find(label6==9)) ];
 M(7,:)=[length(find(label7==1)) length(find(label7==2)) length(find(label7==3)) length(find(label7==4)) length(find(label7==5)) length(find(label7==6)) length(find(label7==7)) length(find(label7==8)) length(find(label7==9))  ];
 M(8,:)=[length(find(label8==1)) length(find(label8==2)) length(find(label8==3)) length(find(label8==4)) length(find(label8==5)) length(find(label8==6)) length(find(label8==7)) length(find(label8==8)) length(find(label8==9))   ];
 M(9,:)=[length(find(label9==1)) length(find(label9==2)) length(find(label9==3)) length(find(label9==4)) length(find(label9==5)) length(find(label9==6)) length(find(label9==7)) length(find(label9==8)) length(find(label9==9)) ];
[Kappa,PCC,Ca,A,x2]=classification_evaluate(M);
Kappa=Kappa*100; Kappa=roundn(Kappa,-2);
PCC=PCC*100;PCC=roundn(PCC,-2);
Ca=Ca*100;Ca=roundn(Ca,-2);AA=sum(Ca)/9;
s=[Ca,Kappa,AA,PCC]
toc
% % %%%%%%%%%%将分类结果图片显示
cluster1=coordinates(:,find(fenlei==1));cluster2=coordinates(:,find(fenlei==2));
cluster3=coordinates(:,find(fenlei==3));cluster4=coordinates(:,find(fenlei==4));
cluster5=coordinates(:,find(fenlei==5));cluster6=coordinates(:,find(fenlei==6));
cluster7=coordinates(:,find(fenlei==7));cluster8=coordinates(:,find(fenlei==8));
cluster9=coordinates(:,find(fenlei==9));
clusters=[cluster1,cluster2,cluster3,cluster4,cluster5,cluster6,cluster7,cluster8,cluster9];
CLASS_6=zeros(610,340);
for d=1:length(cluster1(1,:))
   CLASS_6(cluster1(1,d),cluster1(2,d))=1;
end

for d=1:length(cluster2(1,:))
   CLASS_6(cluster2(1,d),cluster2(2,d))=2;
end 

for d=1:length(cluster3(1,:))
   CLASS_6(cluster3(1,d),cluster3(2,d))=3;
end

for d=1:length(cluster4(1,:))
   CLASS_6(cluster4(1,d),cluster4(2,d))=4;
end

for d=1:length(cluster5(1,:))
   CLASS_6(cluster5(1,d),cluster5(2,d))=5;
end 
for d=1:length(cluster6(1,:))
   CLASS_6(cluster6(1,d),cluster6(2,d))=6;
end

for d=1:length(cluster7(1,:))
   CLASS_6(cluster7(1,d),cluster7(2,d))=7;
end

for d=1:length(cluster8(1,:))
   CLASS_6(cluster8(1,d),cluster8(2,d))=8;
end 
for d=1:length(cluster9(1,:))
   CLASS_6(cluster9(1,d),cluster9(2,d))=9;
end

imagesc(CLASS_6,[0 16]);axis image;axis off;
load color_1
colormap(color_1);




