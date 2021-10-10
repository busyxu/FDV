function [Population,FrontNo,CrowdDis] = EnvironmentalSelection_NSGAII(Population,N)
% The environmental selection of NSGA-II

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
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);  % FrontNo存放的是每个个体所属的非支配层
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end