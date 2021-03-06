function [Off_P,Off_V] = Operator_CMOPSO(Population,Parameter)
% The particle swarm optimization in CMOPSO

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

    %% Parameter setting
    if nargin > 1
        [proM,disM] = deal(Parameter{:});
    else
        [proM,disM] = deal(1,20);
    end
    P_Dec  = Population.decs;     
    [N,D]  = size(P_Dec); 
    P_Obj  = Population.objs;
    V      = Population.adds(zeros(N,D));
    Off_P  = zeros(N,D);
    Off_V  = zeros(N,D);
    Global = GLOBAL.GetObj();
    
    %% Get leaders  
    Front     = NDSort(P_Obj,inf);    
    [~,rank]  = sortrows([Front',-CrowdingDistance(P_Obj,Front)']);
    LeaderSet = rank(1:10);
    
    %% Learning
    for i = 1 : N
        % Competition according to the angle 
        winner = LeaderSet(randperm(length(LeaderSet),2));
        c1     = dot(P_Obj(i,:),P_Obj(winner(1),:))/(norm(P_Obj(i,:))*norm(P_Obj(winner(1),:)));
        angle1 = rad2deg(acos(c1));
        c2     = dot(P_Obj(i,:),P_Obj(winner(2),:))/(norm(P_Obj(i,:))*norm(P_Obj(winner(2),:)));
        angle2 = rad2deg(acos(c2));
        mask   = (angle1 > angle2);
        winner = ~mask.*winner(1) + mask.*winner(2);
        % Learning
        r1 = rand(1,D);
        r2 = rand(1,D);
        Off_V(i,:) = r1.*V(i,:) + r2.*(P_Dec(winner,:)-P_Dec(i,:));
        Off_P(i,:) = P_Dec(i,:) + Off_V(i,:);
    end

    %% Polynomial mutation
    Lower = repmat(Global.lower,N,1);
    Upper = repmat(Global.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Off_P = min(max(Off_P,Lower),Upper);
    Off_P(temp) = Off_P(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                  (1-(Off_P(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Off_P(temp) = Off_P(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                  (1-(Upper(temp)-Off_P(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
              
% 	Offspring = INDIVIDUAL(Off_P,Off_V);
end