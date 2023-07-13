classdef toolboxWrapper < handle & matlab.mixin.Copyable 
    %This class makes it possible to https://github.com/ZeMA-gGmbH/LMT-ML-Toolbox
    
    properties
        pathToToolbox = 'EnterToolboxPath';
        simpleTrainingStack = [];
    end
    
    methods
        function this = toolboxWrapper(algStack,varStack)
            %Example for a Stack {@MultisensorExtractor, @Pearson, @NumFeatRanking, @LDAMahalClassifier}, {{@PCAExtractor}, {500}, {}, {}}
            run([this.pathToToolbox 'addPaths.m'])
            this.simpleTrainingStack = SimpleTrainingStack(algStack, varStack);
        end

        function dataCell = matrix4DtoCell(this,data)
            %Convert dataMatrix to Cell array to be compatible with the
            %toolbox
            dataCell = cell(1,size(data,1));
            for i = 1:size(dataCell,2)
                dataCell{1,i} = zeros(size(data,4),size(data,2));
                for j = 1:size(data,4)
                    dataCell{1,i}(j,:) = squeeze(data(i,:,1,j)); 
                end
            end
        end

        function train(this,data,target)

            dataCell = this.matrix4DtoCell(data);
            this.simpleTrainingStack.train(dataCell,target);
        end
        
        function pred = apply(this,data)
            dataCell = this.matrix4DtoCell(data);
            pred = this.simpleTrainingStack.apply(dataCell);
        end
    end
end

