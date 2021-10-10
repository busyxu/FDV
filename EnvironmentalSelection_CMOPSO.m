function Population = EnvironmentalSelection_CMOPSO(Population,N)
% The environmental selection of CMOPSO

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

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    
    PopObj = Population.objs;
    fmax   = max(PopObj(FrontNo==1,:),[],1);
    fmin   = min(PopObj(FrontNo==1,:),[],1);
    PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);

    %% Select the solutions in the last front
    Last = find(FrontNo==MaxFNo);
    del  = Truncation(PopObj(Last,:),length(Last)-N+sum(Next));
    Next(Last(~del)) = true;
    % Population for next generation
    Population = Population(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    N = size(PopObj,1);
    
    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,N);
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end