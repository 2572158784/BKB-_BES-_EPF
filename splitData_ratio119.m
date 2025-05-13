function [XTrain yTrain XTest yTest Train_co Test_co] = splitData_ratio119(X, y, ratio, coordinates)

% Splits the data into training and testing
% Useful Variables
totalEgs = size(X,1);
numLabels = unique(y); % how many unique labels present

for i = 1:length(numLabels)
    temp = (y==numLabels(i,1)); % find out where each label is present
    idx = find(temp==1); % get the index of that row where the lable is present
    Xtemp = X(idx(:,1),:); % get values of X stored at that particular index
    idxTemp = (randperm(size(Xtemp,1)))'; % randomly choose rows to put into train
    TrainPerLable = ceil(size(Xtemp,1)*ratio);
    if TrainPerLable == 0
        TrainPerLable = 10;
    else
        TrainPerLable = TrainPerLable;
    end
    XTrainTemp = Xtemp(idxTemp(1:TrainPerLable,1),:);
    yTrainTemp = (ones(TrainPerLable,1))*i; 
    XTestTemp = Xtemp(idxTemp(1:end),:); % select remaining rows for testing
    TestPerLable = size(Xtemp,1); 
    yTestTemp = (ones(TestPerLable,1))*i;
    
    co_Temp = coordinates(:, idx(:,1));
    Train_co_Temp = co_Temp(:, idxTemp(1:TrainPerLable,1));
    Test_co_Temp = co_Temp(:, idxTemp(1:end,1));
    
    if i==1 % set up first input to train and test
        XTrain = XTrainTemp;
        yTrain = yTrainTemp;
        XTest = XTestTemp;
        yTest = yTestTemp;
        
        Train_co = Train_co_Temp;
        Test_co = Test_co_Temp;
    else % keep adding new egs to the training and testing sets
        XTrain = [XTrain; XTrainTemp];
        yTrain = [yTrain; yTrainTemp];
        XTest = [XTest; XTestTemp];
        yTest = [yTest; yTestTemp];
        
        Train_co = [Train_co, Train_co_Temp];
        Test_co = [Test_co, Test_co_Temp];
    end
end