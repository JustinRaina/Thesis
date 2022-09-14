%%
clc;
clear all;
close all;

%% input angle data
load("roll_5m.mat")
load("roll_5m_ord.mat") % #2-#1=#1(rollDif)
difference = rollDif;

load("LSTM.csv") % LSTM預測結果
LSTM = round(LSTM, 2);
%% 模擬參數設定
P_dBm = 0;
P_W = 10^(P_dBm/10)/1000;

n_dBm = -80; % dBm
n_var = 10^(n_dBm/10)/1000; % W

c = 3*10^8; 
fc = 28*10^9;  % 28GHz
lambda = c/fc; % 28mm

% Obkect Position
p_BS = [0,0,10].';   
p_ARIS = [-50,80,80].';  
p_UE = [0,100,1.5].'; 

% 3D distance
dBR = norm(p_BS-p_ARIS);
dRU = norm(p_UE-p_ARIS);
dBU = norm(p_BS-p_UE);

%% path loss
rho_dB = -20;
rho = 10^(rho_dB/10);

alpha_bu = 5.76; % path loss exponent between BS and UE
alpha_br = 2; % path loss exponent between BS and RIS
alpha_ru = 2.5; % path loss exponent between RIS and UE

LBU = rho*dBU^(-alpha_bu);
LBR = rho*dBR^(-alpha_br);
LRU = rho*dRU^(-alpha_ru);

%% shadowing
nlos_std = 8.5;
los_std = 4;

% K_factor
K_hRU_dB = 10;
K_hRU = 10^(K_hRU_dB/10);
K_hBR_dB = 10;
K_hBR = 10^(K_hBR_dB/10);


% The number of transmit antennas on BS, M
M = 16;

% The number of Passive Elements on ARIS, N
% N = [49, 100, 196, 400, 784];
N = [49, 100, 196, 400];

% The number of receive antennas on UE, Nr1
Nr = 1;

% set dx & dy to the half of lambda
dx = lambda/2;
dy = lambda/2;
wc = 2*pi/lambda;

result_wio = zeros(1, length(N)); % no vibration
result_noadjust = zeros(1, length(N)); % not adjust
result_adjust = zeros(1, length(N)); % adjust with 0.1s delay
result_garch100 = zeros(1, length(N)); % adjust with ARMA-GARCH, forecast 100
result_garch200 = zeros(1, length(N)); % adjust with ARMA-GARCH, forecast 200
result_garch300 = zeros(1, length(N)); % adjust with ARMA-GARCH, forecast 300
result_garch400 = zeros(1, length(N)); % adjust with ARMA-GARCH, forecast 400
result_garch500 = zeros(1, length(N)); % adjust with ARMA-GARCH, forecast 500
result_LSTM = zeros(1, length(N)); % adjust with LSTM, forecast 500
%% 基於ARMA-GARCH資料導向預測法
angleStartIndex = 101; % 姿態角預測開始的位置
Interval = 100:100:500; % 姿態角預測數
beginning = (200:200:6000)+1; % 截去前面的資料
estiNum = angleStartIndex-1; % 姿態角預測開始的位置-1
pred = cell(length(beginning), 1); % 姿態差值時間序列預測值
ang_copy = cell(length(beginning), 1); % 姿態時間序列備份
for x = 1:length(beginning)
    fprintf("x = %d\n", x);
    predOutNum = Interval(end); % 最大姿態角預測數
    difference_copy = difference(beginning(x):end); % 截去beginning以前的資料
    ang_copy{x} = rollAll(beginning(x)+estiNum:end); % 截去beginning以前的資料
    RF = 1; % 決定AR滯後期數組合的方法: 1.RF 2.閾值法
    GARCH = 1; % 決定是否採用GARCH： 0.AR 2.AR-GARCH
    pred{x} = Function_ARMA_GARCH(difference_copy, estiNum, predOutNum, RF, GARCH, []); % (序列, 訓練數, 預測數, RF?, GARCH?)
end

%% 計算接收端平均可達成傳輸率
for n = 1:length(N) % 不同ARIS天線數
    % BS天線數(x軸y軸)
    Mx = sqrt(M);  
    My = sqrt(M);
    mm = Mx*My;

    % ARIS天線數(x軸y軸)
    Nx = ceil(sqrt(N(n)));
    Ny = ceil(sqrt(N(n)));
    for x = 1:length(beginning) % 不同時間序列起始點
        times = 10; % 蒙地卡羅次數
        rng('default'); % Control random number generator
        for t=1:times % 蒙地卡羅次數
            % BU channel gain
            FBU = 10.^(-randn(1,Mx*My)*nlos_std/10);
            NLOS_hBU = sqrt(1/2)*complex(randn(1,Mx*My),randn(1,Mx*My)); % BS to UE (1xM)
            hBU = sqrt(LBU*FBU).*NLOS_hBU;
    
            % BR, RU channel gain
            FBR = 10^(-randn(1,Nr)*los_std/10);
            FRU = 10^(-randn(1,Nr)*los_std/10);
    
            NLOS_hBR = sqrt(1/2)*complex(randn(Nx*Ny,Mx*My),randn(Nx*Ny,Mx*My));
            NLOS_hRU = sqrt(1/2)*complex(randn(1,Nx*Ny),randn(1,Nx*Ny));  
    
            AFRISr = arrayfactor(Nx,Ny,lambda,dx,dy,p_BS,p_ARIS,0,0,0); % ARIS to BS (1xN)
            AFBS = arrayfactor(Mx,My,lambda,dx,dy,p_ARIS,p_BS,0,0,0); % BS to ARIS (1xM)
            AFRISt = arrayfactor(Nx,Ny,lambda,dx,dy,p_UE,p_ARIS,0,0,0); % ARIS to UE (1xN)
    
            LOS_hBR = AFRISr.'*AFBS; % (NxM)
            LOS_hRU = AFRISt;
            GBR = ricianfading(K_hBR,LOS_hBR,NLOS_hBR);
            GRU = ricianfading(K_hRU,LOS_hRU,NLOS_hRU);
            hBR = sqrt(LBR*FBR).*GBR;
            hRU = sqrt(LRU*FRU).*GRU;
            for a = 2:Interval(end)+1 % 不同的測試姿態角
                fprintf("n = %d, x = %d, t = %d, a = %d\n", n, x, t, a);
                % considering jittery perturbation channel gain with actual data (real time)
                AFRISr_now = arrayfactor(Nx,Ny,lambda,dx,dy,p_BS,p_ARIS,ang_copy{x}(a),0,0); % ARIS to BS (1xN)
                AFRISt_now = arrayfactor(Nx,Ny,lambda,dx,dy,p_UE,p_ARIS,ang_copy{x}(a),0,0); % ARIS to UE (1xN)
                LOS_hBR_now = AFRISr_now.'*AFBS; % (NxM)
                LOS_hRU_now = AFRISt_now;
                GBR_now = ricianfading(K_hBR,LOS_hBR_now,NLOS_hBR); % v
                GRU_now = ricianfading(K_hRU,LOS_hRU_now,NLOS_hRU); % v
                hBR_now = sqrt(LBR*FBR).*GBR_now;
                hRU_now = sqrt(LRU*FRU).*GRU_now;
                
                % considering jittery perturbation channel gain with actual data (delay 0.1s)
                AFRISr_delay1 = arrayfactor(Nx,Ny,lambda,dx,dy,p_BS,p_ARIS,ang_copy{x}(a-1),0,0); % ARIS to BS (1xN), delay 1s
                AFRISt_delay1 = arrayfactor(Nx,Ny,lambda,dx,dy,p_UE,p_ARIS,ang_copy{x}(a-1),0,0); % ARIS to UE (1xN), delay 1s
                LOS_hBR_delay1 = AFRISr_delay1.'*AFBS; % (NxM)
                LOS_hRU_delay1 = AFRISt_delay1;
                GBR_delay1 = ricianfading(K_hBR,LOS_hBR_delay1,NLOS_hBR); % v
                GRU_delay1 = ricianfading(K_hRU,LOS_hRU_delay1,NLOS_hRU); % v
                hBR_delay1 = sqrt(LBR*FBR).*GBR_delay1;
                hRU_delay1 = sqrt(LRU*FRU).*GRU_delay1;
    
                % considering jittery perturbation channel gain with actual data (ARMA-GARCH)
                AFRISr_garch = arrayfactor(Nx,Ny,lambda,dx,dy,p_BS,p_ARIS,ang_copy{x}(a-1)+pred{x}(a-1),0,0); % ARIS to BS (1xN), delay 1s
                AFRISt_garch = arrayfactor(Nx,Ny,lambda,dx,dy,p_UE,p_ARIS,ang_copy{x}(a-1)+pred{x}(a-1),0,0); % ARIS to UE (1xN), delay 1s
                LOS_hBR_garch = AFRISr_garch.'*AFBS; % (NxM)
                LOS_hRU_garch = AFRISt_garch;
                GBR_garch = ricianfading(K_hBR,LOS_hBR_garch,NLOS_hBR); % v
                GRU_garch = ricianfading(K_hRU,LOS_hRU_garch,NLOS_hRU); % v
                hBR_garch = sqrt(LBR*FBR).*GBR_garch;
                hRU_garch = sqrt(LRU*FRU).*GRU_garch;

                % considering jittery perturbation channel gain with actual data (LSTM)
                AFRISr_LSTM = arrayfactor(Nx,Ny,lambda,dx,dy,p_BS,p_ARIS,ang_copy{x}(a-1)+LSTM(x,a-1),0,0); % ARIS to BS (1xN), delay 1s
                AFRISt_LSTM = arrayfactor(Nx,Ny,lambda,dx,dy,p_UE,p_ARIS,ang_copy{x}(a-1)+LSTM(x,a-1),0,0); % ARIS to UE (1xN), delay 1s
                LOS_hBR_LSTM = AFRISr_LSTM.'*AFBS; % (NxM)
                LOS_hRU_LSTM = AFRISt_LSTM;
                GBR_LSTM = ricianfading(K_hBR,LOS_hBR_LSTM,NLOS_hBR); % v
                GRU_LSTM = ricianfading(K_hRU,LOS_hRU_LSTM,NLOS_hRU); % v
                hBR_LSTM = sqrt(LBR*FBR).*GBR_LSTM;
                hRU_LSTM = sqrt(LRU*FRU).*GRU_LSTM;
     
                ris = zeros(Nx*Ny);
                for i=1:Nx*Ny
                    theta = angle(hBU(1,ceil(mm/2)))-angle(hRU(1,i))-angle(hBR(i,ceil(mm/2)));
                    ris(i,i) = exp(1j*theta);
                end
                h_wio = hBU + hRU*ris*hBR; % Without Vibration
                h_noadjust = hBU + hRU_now*ris*hBR_now; % Not Adjust with Vibration
    
                for i=1:Nx*Ny
                    theta = angle(hBU(1,ceil(mm/2)))-angle(hRU_delay1(1,i))-angle(hBR_delay1(i,ceil(mm/2)));
                    ris(i,i) = exp(1j*theta);
                end
                h_adjust = hBU + hRU_now*ris*hBR_now; % Adjust with real data(0.1s delay)
    
                for i=1:Nx*Ny
                    theta = angle(hBU(1,ceil(mm/2)))-angle(hRU_garch(1,i))-angle(hBR_garch(i,ceil(mm/2)));
                    ris(i,i) = exp(1j*theta);
                end
                h_garch = hBU + hRU_now*ris*hBR_now; % Adjust with real data(ARMA-GARCH)

                for i=1:Nx*Ny
                    theta = angle(hBU(1,ceil(mm/2)))-angle(hRU_LSTM(1,i))-angle(hBR_LSTM(i,ceil(mm/2)));
                    ris(i,i) = exp(1j*theta);
                end
                h_LSTM = hBU + hRU_now*ris*hBR_now; % Adjust with real data(ARMA-GARCH)
    
                w = (h_wio')./norm(h_wio').*sqrt(P_W);
                if a <= 101 % 前100個結果                    
                    result_garch100(n) = result_garch100(n) + log2(1 + (abs(h_garch*w))^2/n_var); % Adjust with real data(ARMA-GARCH)
                end
                if a <= 201 % 前200個結果
                    result_garch200(n) = result_garch200(n) + log2(1 + (abs(h_garch*w))^2/n_var); % Adjust with real data(ARMA-GARCH)
                end
                if a <= 301 % 前300個結果
                    result_garch300(n) = result_garch300(n) + log2(1 + (abs(h_garch*w))^2/n_var); % Adjust with real data(ARMA-GARCH)
                end
                if a <= 401 % 前400個結果
                    result_garch400(n) = result_garch400(n) + log2(1 + (abs(h_garch*w))^2/n_var); % Adjust with real data(ARMA-GARCH)
                end
                % 前500個結果
                result_wio(n) = result_wio(n) + log2(1 + (abs(h_wio*w))^2/n_var); % Without Vibration
                result_noadjust(n) = result_noadjust(n) + log2(1 + (abs(h_noadjust*w))^2/n_var); % Not Adjust with Vibration
                result_adjust(n) = result_adjust(n) + log2(1 + (abs(h_adjust*w))^2/n_var); % Adjust with real data(0.1s delay)
                result_garch500(n) = result_garch500(n) + log2(1 + (abs(h_garch*w))^2/n_var); % Adjust with real data(ARMA-GARCH)
                result_LSTM(n) = result_LSTM(n) + log2(1 + (abs(h_LSTM*w))^2/n_var); % Adjust with real data(ARMA-GARCH)
            end % end a
        end % end t
    end % end x
    % 求傳輸率的平均
    result_wio(n) = result_wio(n)./(500*times)./length(beginning);
    result_noadjust(n) = result_noadjust(n)./(500*times)./length(beginning);
    result_adjust(n) = result_adjust(n)./(500*times)./length(beginning);
    result_garch100(n) = result_garch100(n)./(100*times)./length(beginning);
    result_garch200(n) = result_garch200(n)./(200*times)./length(beginning);
    result_garch300(n) = result_garch300(n)./(300*times)./length(beginning);
    result_garch400(n) = result_garch400(n)./(400*times)./length(beginning);
    result_garch500(n) = result_garch500(n)./(500*times)./length(beginning);
    result_LSTM(n) = result_LSTM(n)./(500*times)./length(beginning);
end % end n

%%  繪製ARIS天線數-接收端平均可達成傳輸率圖
figure;
plot(N, result_wio, '-ko', N, result_noadjust, '-bs', N, result_adjust, '-r^', N, result_garch500, '-x', N, result_LSTM, '-+'); 
legend('Without Vibration', 'Not Adjust with Vibration', 'Adjust with 0.1s delay', 'AR-GARCH', 'LSTM', 'Location','northwest');
xlabel('The number of ARIS passive elements, N');
ylabel('Average achievable rate in bps/Hz');