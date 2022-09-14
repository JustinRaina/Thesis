%%
clc;
clear all;

%% load data
load("..\BestLags\estiNum100SeriesNum400AR\5m\PACF.mat");
load("..\BestLags\estiNum100SeriesNum400AR\5m\ARLagsPACFMatrix.mat");
load("..\BestLags\estiNum100SeriesNum400AR\5m\nextExhaustive\ARLagsMaxExhaustiveMatrix.mat");
load("..\BestLags\estiNum100SeriesNum400AR\5m\PassQMatrix.mat");
load("..\BestLags\estiNum100SeriesNum400AR\5m\PassQ2Matrix.mat");
load("..\BestLags\estiNum100SeriesNum400AR\5m\PassJBMatrix.mat");
load("..\BestLags\estiNum100SeriesNum400AR\Combination.mat");

%% 
resultQ = zeros(1,3); % Q檢定通過否(平均)(RF模型個數*三種方法)
resultQ2 = zeros(1,3); % Q2檢定通過否(平均)(RF模型個數*三種方法)
resultJB = zeros(1,3); % JB檢定通過否(平均)(RF模型個數*三種方法)

%% 所有資料訓練一個模型
seriesNum = 400; % 時間序列樣本數
trainNum = seriesNum*0.8; % 訓練樣本數
testNum = seriesNum*0.2-1; % 測試樣本數
trainX = PACF(end-trainNum:end-1, :); % 訓練X
testX = PACF(1:testNum, :); % 訓練Y
trainY = ARLagsMaxMatrix(end-trainNum:end-1, :); % 測試X
testY = ARLagsMaxMatrix(1:testNum, :); % 測試Y

%% Random Forest
rng("default");
NumTrees = 1000; % 決策樹的數量
predictResuslt = zeros(testNum, 5); % 測試結果
B = cell(5,1); % 隨機森林模型
T = zeros(1,5);
for i = 1:5
    tic % time start
    B{i} = TreeBagger(NumTrees,trainX,trainY(:, i));
    T(i) = toc; % time stop
    Yfit = predict(B{i},testX);
    predictResuslt(:, i) = str2num(cell2mat(Yfit));
end
tMul = sum(T) % time accumulate

%% Boolean matrix to lags 
RFlags = cell(testNum, 1); % RF
for i = 1:length(RFlags)
    lags = [];
    for j = 1:5
        if predictResuslt(i, j) == 1
            lags = [lags j];
        end
    end
    RFlags{i} = lags;
end

PACFlags = cell(testNum, 1); % 閾值法
for i = 1:length(PACFlags)
    lags = [];
    for j = 1:5
        if ARLagsPACFMatrix(i, j) == 1
            lags = [lags j];
        end
    end
    PACFlags{i} = lags;
end

Maxlags = cell(testNum, 1); % 窮舉法
for i = 1:length(Maxlags)
    lags = [];
    for j = 1:5
        if ARLagsMaxMatrix(i, j) == 1
            lags = [lags j];
        end
    end
    Maxlags{i} = lags;
end

%% check AvgProb
RFQPass = zeros(testNum, 1);
RFQ2Pass = zeros(testNum, 1);
RFJBPass = zeros(testNum, 1);
for i = 1:length(RFQPass)
    for j = 1:length(checkMatrix)
        if isequal(RFlags{i}, checkMatrix{j})
            RFQPass(i) = PassQMatrix(i,j);
            RFQ2Pass(i) = PassQ2Matrix(i,j);
            RFJBPass(i) = PassJBMatrix(i,j);
            break;
        end
    end
end
resultQ(1) = mean(RFQPass);
resultQ2(1) = mean(RFQ2Pass);
resultJB(1) = mean(RFJBPass);

PACFQPass = zeros(testNum, 1);
PACFQ2Pass = zeros(testNum, 1);
PACFJBPass = zeros(testNum, 1);
for i = 1:length(PACFQPass)
    for j = 1:length(checkMatrix)
        if isequal(PACFlags{i}, checkMatrix{j})
            PACFQPass(i) = PassQMatrix(i,j);
            PACFQ2Pass(i) = PassQ2Matrix(i,j);
            PACFJBPass(i) = PassJBMatrix(i,j);
            break;
        end
    end
end
resultQ(2) = mean(PACFQPass);
resultQ2(2) = mean(PACFQ2Pass);
resultJB(2) = mean(PACFJBPass);

MaxQPass = zeros(testNum, 1);
MaxQ2Pass = zeros(testNum, 1);
MaxJBPass = zeros(testNum, 1);
for i = 1:length(MaxQPass)
    for j = 1:length(checkMatrix)
        if isequal(Maxlags{i}, checkMatrix{j})
            MaxQPass(i) = PassQMatrix(i,j);
            MaxQ2Pass(i) = PassQ2Matrix(i,j);
            MaxJBPass(i) = PassJBMatrix(i,j);
            break;
        end
    end
end
resultQ(3) = mean(MaxQPass);
resultQ2(3) = mean(MaxQ2Pass);
resultJB(3) = mean(MaxJBPass);