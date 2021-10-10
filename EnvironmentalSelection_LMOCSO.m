function Population = EnvironmentalSelection_LMOCSO(Population,V,theta)
% The environmental selection of LMOCSO

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

    Population = Population(NDSort(Population.objs,1)==1);%取非支配排序第一层
    PopObj = Population.objs;
    [N,M]  = size(PopObj);
    NV     = size(V,1);
    
    %% Translate the population
    PopObj = PopObj - repmat(min(PopObj,[],1),N,1);
    
    %% Calculate the smallest angle value between each vector and others
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma  = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);

    %% Select one solution for each reference vector
    Next = zeros(1,NV);
    %   一个向量只绑一个点，比如与向量1角度最小的有10个点，那么就选择一个APD值最小的点绑在向量上面
    for i = unique(associate)' %被选出要绑点的向量号, %%%%单引号表示转置
        current = find(associate==i);  % 选出与向量i角度最小的粒子，（可能有多个）
        % Calculate the APD value of each solution
        APD = (1+M*theta*Angle(current,i)/gamma(i)).*sqrt(sum(PopObj(current,:).^2,2));
        % Select the one with the minimum APD value
        [~,best] = min(APD); % APD数组中最小元素的索引
        Next(i) = current(best);   % APD值最小的粒子记录在next中，最后这个最小的也就是确定绑在i号向量上的粒子
    end
    %有点的向量越来越多了
    % Population for next generation
    Population = Population(Next(Next~=0));
end