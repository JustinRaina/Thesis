function [RFlags] = Function_RF20ShufflePass(testX, testNum)
% 隨機森林法
% testX：First five PACF
% testNum：Series number

load(".\RF\RF_model\estiNum100SeriesNum400ARGARCH\RMSENextExhaustive\new\RF_model.mat");
predictResuslt = zeros(testNum, 5);
for i = 1:5
    Yfit = predict(B{i},testX.'); %(1,5)
    predictResuslt(:, i) = str2num(cell2mat(Yfit)); %(1,5)
end

%% Boolean matrix to lags 
RFlags = cell(testNum, 1);
for i = 1:length(RFlags)
    lags = [];
    for j = 1:5
        if predictResuslt(i, j) == 1
            lags = [lags j];
        end
    end
    RFlags{i} = lags;
end

end