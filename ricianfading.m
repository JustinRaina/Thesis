function RF = ricianfading(K,hLOS,hNLOS)
    RF = sqrt(K/(1+K)).*hLOS + sqrt(1/(1+K)).*hNLOS;
end