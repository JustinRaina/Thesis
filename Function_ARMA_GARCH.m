function predOutVec = Function_ARMA_GARCH(rollDif,estiNum,predOutNum,RF,GARCH,Exlags)
% AR-GARCH模型
% estiNum: data number for model estimation
% predOutNum: data number for prediction
trainX = rollDif(1:estiNum); % training data
testNum = 1;

% PACF, ACF
lagsNum = 5;
bounds = 1.96/(estiNum)^0.5;
ARLags = []; % considered AR terms
PACFTbl = parcorr(trainX,lagsNum);
%PACFTbl = parcorr(trainX,NumAR=lagsNum);
if RF == 1 % 使用RF
    RFlags = Function_RF20ShufflePass(PACFTbl(2:6), testNum);
    %RFlags = Function_RF20ShufflePass(abs(PACFTbl(2:6)), testNum);
    ARLags = RFlags{1};
elseif RF == 2 % 使用閾值法
    for i = 1:lagsNum
        if abs(PACFTbl(i+1))>bounds
            ARLags = [ARLags i];
        end
    end
else
    ARLags = Exlags;
    %ARLags = [];
end

% MALags = []; % considered MA terms
% ACFTbl = autocorr(trainX,NumLags=lagsNum+1,NumMA=lagsNum);
% for i = 1:lagsNum
%     if abs(ACFTbl(i+1))>bounds
%         MALags = [MALags i];
%     end
% end
%% composite model
if GARCH == 1
    VarMdl = garch(1,1); % variance equation
    Mdl = arima('ARLags',ARLags,'Variance',VarMdl); % mean equation
else
    % Mdl = arima('ARLags',ARLags,'MALags',MALags,'Variance',VarMdl); % mean equation
    Mdl = arima('ARLags',ARLags); % mean equation
end

% model estimating
EstMdl = estimate(Mdl,trainX);

% % residual, variance, log likelihood
% [res,v,logL] = infer(EstMdl,trainX);

% In-sample prediction(Static Forcast Method)
% predInNum = estiNum-max(Mdl.P, Mdl.Q); % data number for prediction
% predInVec = zeros(predInNum,1);
% for i = 1:predInNum
%     preSampleInY = rollDif(i:i+max(Mdl.P, Mdl.Q)-1);
%     predInVec(i) = forecast(EstMdl,1,preSampleInY);
% end

%% residual test(lags = 20)
% % Q test
% res = predInVec-rollDif(max(Mdl.P, Mdl.Q)+1:estiNum);
% hQ = lbqtest(res);
% [hQ,pValue,stat,cValue] = lbqtest(res)
% % Q^2 test
% sampleVar = (res.'*res)/length(res);
% resSquare = res.^2-sampleVar;
% hQ2 = lbqtest(resSquare);
% [hQ2,pValue,stat,cValue] = lbqtest(resSquare)
% % JB test
% [hJB,p] = jbtest(res);
% [hJB,pValue,stat,cValue] = jbtest(res)

%% Out-of-sample prediction(Static Forcast Method)
predOutVec = zeros(predOutNum,1);
for i = 1:predOutNum
    preSampleOutY = rollDif(estiNum-Mdl.P+i:estiNum-1+i);
    predOutVec(i) = forecast(EstMdl,1,preSampleOutY); 
end

end