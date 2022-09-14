% edit by Caro 2021.04.28
% Generate an Array Factor
function AF = arrayfactor(MR, NR, lambda, dx, dy, p_TX, p_RX, row_ang, pitch_ang, yaw_ang)

% input parameters
% MR: the antenna number in x-direction
% NR: the antenna number in y-direction
% p_TX: the position of TX (part path TX)
% p_RX: the position of RX (part path RX)
% row_ang: the rotation in x axis
% pitch_ang: the rotation in y axis
% yaw_ang: the rotation in z axis

% output array factor with size (1 x MR*NR)
AF = zeros(1,MR*NR);

for m = 1:MR
    for n = 1:NR
        % x: position vector of ith antenna
        x = [((m-1)-(MR-1)/2)*dx ((n-1)-(NR-1)/2)*dy 0];
        % row angle(x)        
        row_mat = [1 0 0; 0 cos(row_ang) sin(row_ang); 0 -sin(row_ang) cos(row_ang)].';
        % pitch angle(y)        
        pitch_mat = [cos(pitch_ang) 0 -sin(pitch_ang); 0 1 0; sin(pitch_ang) 0 cos(pitch_ang)].';
        % yaw angle(x)        
        yaw_mat = [cos(yaw_ang) sin(yaw_ang) 0; -sin(yaw_ang) cos(yaw_ang) 0; 0 0 1].';                
        total_mat = row_mat*pitch_mat*yaw_mat;
        x = x*total_mat;        
        % unit vector pointed at signal
        r = p_TX-p_RX;
        r = r/norm(r);
        AF((m-1)*NR+n) = exp(1j*2*pi/lambda*x*r);        
    end
end

end