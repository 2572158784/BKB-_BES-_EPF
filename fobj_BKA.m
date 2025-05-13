function y = fobj_BKA(X,lambda,sigma)
% dim=size(x,2);
% o=sum((1:dim).*(x.^4))+rand;


img=X;
% %img=double(img);max0=max(max(img));img=img/max0;
  Rimg=img(:,:,1);Gimg=img(:,:,1);
  % 初始化参数
  [row,column]=size(Rimg);
  tempimg1=zeros(row,column);  tempimg2=zeros(row,column);
  tempimg3=zeros(row,column);  tempimg4=zeros(row,column);
  tempimg5=zeros(row,column);  tempimg6=zeros(row,column);
  len=numel(Rimg(:));
  Psi1=zeros(1,len); Phi1=zeros(1,len); Psi2=zeros(1,len); Phi2=zeros(1,len);
  X1=zeros(1,len);X2=zeros(1,len);Y1=zeros(1,len);Y2=zeros(1,len);
% 注：参数sigma的值越大，图像越模糊，如果需要较大程度地磨皮，应该在保持sigma值较小的前提下，逐渐增大lambda的值
% 这样才能使图像不会变得太模糊
 % 对RGB三个通道分别处理
 % horizon-vertical processing
 % 1.horizon processing
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
 % 1.vertical processing
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
y=img119;                                              


end
