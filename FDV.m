function FDV(Global)
% <algorithm> <F> 
% Fuzzy Decision Variable Framework (FDV) with various internal optimizers.
% Rate    --- 0.8    --- Fuzzy evolution rate. Default = 0.8
% Acc     --- 0.4    --- Step acceleration. Default = 0.4
% optimiser     --- 1    --- Internal optimisation algorithm. 1 = NSGAII, 2 = NSGAIII, 3 = MOEAD, 4 = CMOPSO, 5 = LMOCSO. Default = LMOCSO.
% type --- 1 --- The type of aggregation function. If it is based on the
% MOEA/D algorithm, you need to set the aggregate function type. Default = 1


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

    %% Set the default parameters
    [Rate,Acc,optimiser,type] = Global.ParameterSet(0.8,0.4,1,1);
    
    %% NSGAII
    if optimiser==1
        % Generate random population
        Population = Global.Initialization();
        [~,FrontNo,CrowdDis] = EnvironmentalSelection_NSGAII(Population,Global.N);
        
        % Optimization
        while Global.NotTermination(Population)
            MatingPool = TournamentSelection(2,Global.N,FrontNo,-CrowdDis);
            OffDec  = GA(Population(MatingPool).decs);
            %% FDV
            iter = Global.gen/Global.maxgen;
            if iter <= Rate
                Offspring = FDVOperator(Rate,Acc,OffDec);
            else
                Offspring = INDIVIDUAL(OffDec);
            end
            %% 
            [Population,FrontNo,CrowdDis] = EnvironmentalSelection_NSGAII([Population,Offspring],Global.N);
        end
    end
    
    %% NSGAIII
    if optimiser==2
        % Generate the reference points and random population
        [Z,Global.N] = UniformPoint(Global.N,Global.M);
        Population   = Global.Initialization();
        Zmin         = min(Population(all(Population.cons<=0,2)).objs,[],1);

        % Optimization
        while Global.NotTermination(Population)
            MatingPool = TournamentSelection(2,Global.N,sum(max(0,Population.cons),2));
            OffDec  = GA(Population(MatingPool).decs);
            %% FDV
            iter = Global.gen/Global.maxgen;
            if iter <= Rate
                Offspring = FDVOperator(Rate,Acc,OffDec);
            else
                Offspring = INDIVIDUAL(OffDec);
            end
            %% 
            Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
            Population = EnvironmentalSelection_NSGAIII([Population,Offspring],Global.N,Z,Zmin);
        end
    end
    
    %% MOEA/D
    if optimiser==3
        % Generate the weight vectors
        [W,Global.N] = UniformPoint(Global.N,Global.M);
        T = ceil(Global.N/10);

        % Detect the neighbours of each solution
        B = pdist2(W,W);
        [~,B] = sort(B,2);
        B = B(:,1:T);

        % Generate random population
        Population = Global.Initialization();
        Z = min(Population.objs,[],1);

        % Optimization
        while Global.NotTermination(Population)
            % For each solution
            for i = 1 : Global.N      
                % Choose the parents
                P = B(i,randperm(size(B,2)));

                % Generate an offspring
                OffDec = GAhalf(Population(P(1:2)).decs);
                %% FDV
                iter = Global.gen/Global.maxgen;
                if iter <= Rate
                    Offspring = FDVOperator(Rate,Acc,OffDec);
                else
                    Offspring = INDIVIDUAL(OffDec);
                end
                
                % Update the ideal point
                Z = min(Z,Offspring.obj);

                % Update the neighbours
                switch type
                    case 1
                        % PBI approach
                        normW   = sqrt(sum(W(P,:).^2,2));
                        normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                        normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                        CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                        CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                        g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                        g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                    case 2
                        % Tchebycheff approach
                        g_old = max(abs(Population(P).objs-repmat(Z,T,1)).*W(P,:),[],2);
                        g_new = max(repmat(abs(Offspring.obj-Z),T,1).*W(P,:),[],2);
                    case 3
                        % Tchebycheff approach with normalization
                        Zmax  = max(Population.objs,[],1);
                        g_old = max(abs(Population(P).objs-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(P,:),[],2);
                        g_new = max(repmat(abs(Offspring.obj-Z)./(Zmax-Z),T,1).*W(P,:),[],2);
                    case 4
                        % Modified Tchebycheff approach
                        g_old = max(abs(Population(P).objs-repmat(Z,T,1))./W(P,:),[],2);
                        g_new = max(repmat(abs(Offspring.obj-Z),T,1)./W(P,:),[],2);
                end

                Population(P(g_old>=g_new)) = Offspring;
            end
        end
    end
    
    %% CMOPSO
    if optimiser == 4
        % Generate random population
        Population = Global.Initialization();

        % Optimization
        while Global.NotTermination(Population)
            [OffDec,OffVel]  = Operator_CMOPSO(Population);
            %% FDV
            iter = Global.gen/Global.maxgen;
            if iter <= Rate
                Offspring = FDVOperator(Rate,Acc,OffDec,OffVel);
            else
                Offspring = INDIVIDUAL(OffDec,OffVel);
            end
            
            %% 
            Population = EnvironmentalSelection_CMOPSO([Population,Offspring],Global.N);
        end
    end
    
    %% LMOCSO
    if optimiser == 5
         % Generate random population
        [V,Global.N] = UniformPoint(Global.N,Global.M);
        Population   = Global.Initialization();
        Population   = EnvironmentalSelection_LMOCSO(Population,V,(Global.gen/Global.maxgen)^2);

        % Optimization
        while Global.NotTermination(Population)
            % Calculate the fitness by shift-based density   SDE (the shift-based density estimation strategy)
            PopObj = Population.objs;
            N      = size(PopObj,1);
            fmax   = max(PopObj,[],1);
            fmin   = min(PopObj,[],1);
            PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
            Dis    = inf(N);
            for i = 1 : N
                SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
                for j = [1:i-1,i+1:N]
                    Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:)); 
                end
            end
            Fitness = min(Dis,[],2); 
            
            if length(Population) >= 2
                Rank = randperm(length(Population),floor(length(Population)/2)*2);
            else
                Rank = [1,1];
            end
            Loser  = Rank(1:end/2);
            Winner = Rank(end/2+1:end);
            Change = Fitness(Loser) >= Fitness(Winner);
            Temp   = Winner(Change);
            Winner(Change) = Loser(Change);
            Loser(Change)  = Temp;

            [OffDec,OffVel]      = Operator_LMOCSO(Population(Loser),Population(Winner),Rate);
            %% FDV
            iter = Global.gen/Global.maxgen;
            if iter <= Rate
                Offspring = FDVOperator(Rate,Acc,OffDec,OffVel);
            else
                Offspring = INDIVIDUAL(OffDec,OffVel);
            end
            %% 
            Population    = EnvironmentalSelection_LMOCSO([Population,Offspring],V,(Global.gen/Global.maxgen)^2);
        end
        
    end

end

