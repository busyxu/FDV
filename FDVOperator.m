function Offspring = FDVOperator(Rate,Acc,OffDec,OffVel)

%------------------------------- Reference --------------------------------
% X. Yang, J. Zou, S. Yang, J. Zheng and Y. Liu, 
% "A Fuzzy Decision Variables Framework for Large-scale Multiobjective Optimization," 
% in IEEE Transactions on Evolutionary Computation, doi: 10.1109/TEVC.2021.3118593.
% ----------------------------------------------------------------------- 
%  Copyright (C) 2021 Xu Yang
% ----------------------------------------------------------------------- 
%  Author of this Code: 
%   Xu Yang <xuyang.busyxu@qq.com> or <xuyang369369@gmail.com>
% ----------------------------------------------------------------------- 
    
    Global = GLOBAL.GetObj();
    
    %% Fuzzy Evolution Sub-stages Division
    Total = 1;
    S = floor(sqrt(2*Rate*Total/Acc));
    Step = zeros(1,S+2);  % Step(1)=0£¬Step(S+2) is the compensation step
    for i=1:S
        Step(i+1) = (S*i-i*i/2)*Acc;
    end
    Step(S+2) = Rate*Total;  % compensation step

    %% Fuzzy Operation
    R = Global.upper-Global.lower;
    iter = Global.gen/Global.maxgen;
    for i=1:S+1
        if iter>Step(i) && iter<=Step(i+1)
            gamma_a=R*10^-i.*floor(10^i*R.^-1.*(OffDec-Global.lower)) + Global.lower;
            gamma_b=R*10^-i.*ceil(10^i*R.^-1.*(OffDec-Global.lower)) + Global.lower;
%             gamma_a=R*10^-i.*floor(10^i*R.^-1.*OffDec);
%             gamma_b=R*10^-i.*ceil(10^i*R.^-1.*OffDec);
            miu1 = 1./(OffDec-gamma_a);
            miu2 = 1./(gamma_b-OffDec);
            logical = miu1-miu2>0;
            OffDec = gamma_b;
            OffDec(find(logical)) = gamma_a(find(logical));
        end
    end
    
    if nargin > 3
        Offspring = INDIVIDUAL(OffDec,OffVel);
    else
        Offspring = INDIVIDUAL(OffDec);
    end
    
end