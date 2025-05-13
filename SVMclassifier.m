function [outlabel, f, Time] = SVMclassifier(trainfeature, trainlabel, testfeature, testlabel, C, sigma)

    L=max(trainlabel);
    TrainTime = 0; Train_Time = 0; TestTime = 0; Test_Time = 0;
    for i=1:L
        label=(trainlabel==i);
%         tic;
        %% %%%%%%%%%%%%%%%%%%%
        svmStruct=svmtrain(trainfeature,label,...
            'Kernel_Function','rbf','RBF_Sigma',sigma,'showplot',0, ...
            'Autoscale',0,'BoxConstraint',C,'Method','LS');
        Train_Time = toc;
        TrainTime = TrainTime + Train_Time;
        
%         tic;
        classes=rbf_kernel(testfeature,svmStruct.SupportVectors,sigma);        
        classes=classes.*repmat((svmStruct.Alpha'),length(testlabel),1);
        f(:,i)=-(sum(classes,2)+svmStruct.Bias);
        Test_Time = toc;
        TestTime =  TestTime +  Test_Time; 

    end
    
    [mf,outlabel]=max(f,[],2);  
    
    Time.train = TrainTime; 
    Time.test = TestTime;    
end