 classdef fullyConected<handle & matlab.mixin.Copyable 

    properties
        lgraph = [];
        options = [];
        net = [];
       
        inputSize = [];
        outputSize = [];
        regression = true;
        builtFlag = false;

        optimVars = [
            optimizableVariable('SectionDepth',[0 3],'Type','integer')
            optimizableVariable('InitialLearnRate_Beginning',[1e-7 1e-1],'Transform','log')
            optimizableVariable('numNeurons',[100 1000],'Type','integer')
            optimizableVariable('dropOut',[0.1 0.5],'Type','real')
            optimizableVariable('l2',[0.0001 0.00011],'Transform','log')
            optimizableVariable('mini',[1200 3001],'Type','integer')
            optimizableVariable('numEpoch',[100 501],'Type','integer')
            optimizableVariable('dropRate',[0.1 0.2],'Type','real')];


        optimVarsRetrain = [
            optimizableVariable('SectionDepth',[0 3],'Type','integer')
            optimizableVariable('InitialLearnRate_Beginning',[1e-7 1e-1],'Transform','log')
            optimizableVariable('numNeurons',[100 1000],'Type','integer')
            optimizableVariable('dropOut',[0.1 0.5],'Type','real')
            optimizableVariable('l2',[0.0001 0.00011],'Transform','log')
            optimizableVariable('mini',[1200 3001],'Type','integer')
            optimizableVariable('numEpoch',[100 501],'Type','integer')
            optimizableVariable('dropRate',[0.1 0.2],'Type','real')
            optimizableVariable('MaxEpochs',[500 501],'Type','integer')
            optimizableVariable('InitialLearnRate',[1e-7 1e-2],'Transform','log')];

    end

    methods
        function this = fullyConected(InputSize,Outputsize,Regression,optim)
            this.inputSize = InputSize;
            this.outputSize = Outputsize;
            this.regression = Regression;
            if ~exist('optim','var')
                disp("No Model Built")
            else
                this.builtNet(optim);
            end
        end


        function builtNet(this,optim)
            this.builtFlag = true;


            dropOut = optim.dropOut;
            numConvs = optim.SectionDepth;
            widthFC = optim.numNeurons;
            outpuSize = this.outputSize;
            input_shape = [this.inputSize 1];
        
            layers = [
                imageInputLayer(input_shape,'Normalization','none','NormalizationDimension','auto','name','InputLayer')
                this.fullyConUnitNorm(widthFC,'First')
                batchNormalizationLayer('Name',['BN1'])
                reluLayer('Name',['First relu fc'])

            ];

            for i = 1:numConvs-1
                if i < (numConvs-1)/2
                    layers = [layers;this.fullyConUnitNorm(ceil(widthFC*(2^(i))),['FC', convertStringsToChars(num2str(i))])];

                else
                    layers = [layers;this.fullyConUnitNorm(ceil(widthFC/(2^(i))),['FC', convertStringsToChars(num2str(i))])];
                end


            end

            

            layers = [layers;dropoutLayer(dropOut,'name','DropOut1')];
            layers = [layers;this.fullyConUnit(outpuSize,'Final')];

            if this.regression == true
                layers = [layers;regressionLayer('Name','routput')];
            else
                layers = [layers;softmaxLayer('Name','softmax')];
                layers = [layers;classificationLayer('Name','classoutput')];
            end
            
            if ismember('dropRate',optim.Properties.VariableNames)
            
            else
                	optim.dropRate = 0.1;
            end
            
            lgraph = layerGraph(layers);
            this.lgraph = lgraph;
            this.options = trainingOptions('adam', ...
                'ExecutionEnvironment',"gpu",...
                'InitialLearnRate',optim.InitialLearnRate_Beginning, ...
                'MaxEpochs',optim.numEpoch, ...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropPeriod',2, ...
                'LearnRateDropFactor',(1-optim.dropRate), ...
                'MiniBatchSize',optim.mini, ...
                'L2Regularization',optim.l2, ...
                'Shuffle','every-epoch', ...
                'Verbose',false, ...
                'Plots','none', ...
                'ValidationFrequency',10);

        end
        
        
        
        function layers = fullyConUnit(this,numF,tag)
        
            layers = [
                fullyConnectedLayer(numF,'Name',[tag 'fc'])
                reluLayer('Name',[tag 'relu fc'])
            ];
        
        end

        function layers = fullyConUnitNorm(this,numF,tag)
        
            layers = [
                fullyConnectedLayer(numF,'Name',[tag 'fc'])
                batchNormalizationLayer('Name',[tag 'LayerNorm'])
                reluLayer('Name',[tag 'relu fc'])
            ];
        
        end

        function train(this,data,target)
            if this.builtFlag
                this.net = trainNetwork(data,target,this.lgraph,this.options);
            else
                disp("Model Not Built")
            end
        end

        %%%%Optimize only Base function
        function optimizefullyConected(this,paraStore,data,target,numEpochs)
            defaultParameter = [];
            defaultParameter.InputSize = this.inputSize;
            defaultParameter.OutputSize = this.outputSize;
            defaultParameter.Regression = this.regression;


            if ~exist('numEpochs','var')
                numEpochs = 50;
            end

            ObjFcn = fullyConected_OptPara(data.train,target.train,data.val,target.val,defaultParameter);
            BayesObject = bayesopt(ObjFcn,this.optimVars, ...
            'MaxObjectiveEvaluations',numEpochs,...
            'IsObjectiveDeterministic',false, ...
            'UseParallel',false);
            optim =  bestPoint(BayesObject);
            this.builtNet(optim);
            save(paraStore,"optim")

        end

        %Optimize with retraining
        function optimizefullyConectedRetrain(this,paraStore,data,target,numEpochs)
            defaultParameter = [];
            defaultParameter.InputSize = this.inputSize;
            defaultParameter.OutputSize = this.outputSize;
            defaultParameter.Regression = this.regression;


            if ~exist('numEpochs','var')
                numEpochs = 50;
            end

            ObjFcn = fullyConected_optPara_Retrain(data.train,target.train,data.val,target.val,defaultParameter,data.retrain,target.retrain,data.retrainVal,target.retrainVal);
            BayesObject = bayesopt(ObjFcn,this.optimVarsRetrain, ...
            'MaxObjectiveEvaluations',numEpochs,...
            'IsObjectiveDeterministic',false, ...
            'UseParallel',false);
            optim =  bestPoint(BayesObject);
            this.builtNet(optim);
            save(paraStore,"optim")
        end


        %retrain the entire MLP with a new learning rate
        function retrainFull(this,data,target,retrainOpt)
            fn = fieldnames(retrainOpt);
            fo = fieldnames(this.options);

            for k=1:numel(fn)
                for i = 1:numel(fo)
                    if strcmp(fo{i},fn{k})
                        this.options.(fn{k}) = retrainOpt.(fn{k});
                    end
                end
            end


            this.net = trainNetwork(data,target,layerGraph(this.net),this.options);
        end

        


        function pred = apply(this,data)
            
            %data = (data-this.normMu)./this.normSigma;
            pred = predict(this.net,data);
        end

        

    end
end

