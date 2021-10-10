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

    Population = Population(NDSort(Population.objs,1)==1);%ȡ��֧�������һ��
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
    %   һ������ֻ��һ���㣬����������1�Ƕ���С����10���㣬��ô��ѡ��һ��APDֵ��С�ĵ������������
    for i = unique(associate)' %��ѡ��Ҫ����������, %%%%�����ű�ʾת��
        current = find(associate==i);  % ѡ��������i�Ƕ���С�����ӣ��������ж����
        % Calculate the APD value of each solution
        APD = (1+M*theta*Angle(current,i)/gamma(i)).*sqrt(sum(PopObj(current,:).^2,2));
        % Select the one with the minimum APD value
        [~,best] = min(APD); % APD��������СԪ�ص�����
        Next(i) = current(best);   % APDֵ��С�����Ӽ�¼��next�У���������С��Ҳ����ȷ������i�������ϵ�����
    end
    %�е������Խ��Խ����
    % Population for next generation
    Population = Population(Next(Next~=0));
end