 classdef fullyConected<handle & matlab.mixin.Copyable 

    properties
        lgraph = [];
        options = [];
        net = [];
        shapleyVal = [];
        occVals = [];
        limeVals = [];
        gradMaps = [];
        builtFlag = false;
        regDev = true
        inputSize = [];
        optimVars = [
            optimizableVariable('SectionDepth',[0 3],'Type','integer')
            optimizableVariable('InitialLearnRate',[1e-7 1e-1],'Transform','log')
            optimizableVariable('numNeurons',[100 1000],'Type','integer')
            optimizableVariable('dropOut',[0.1 0.5],'Type','real')
            optimizableVariable('l2',[0.0001 0.00011],'Transform','log')
            optimizableVariable('mini',[1200 3001],'Type','integer')
            optimizableVariable('numEpoch',[100 501],'Type','integer')];

         optimVarsAll = [
            optimizableVariable('SectionDepth',[0 3],'Type','integer')
            optimizableVariable('InitialLearnRate',[1e-7 1e-1],'Transform','log')
            optimizableVariable('numNeurons',[100 2000],'Type','integer')
            optimizableVariable('dropOut',[0 0.4],'Type','real')
            optimizableVariable('l2',[0.0001 0.01],'Transform','log')
            optimizableVariable('mini',[2000 2001],'Type','integer')
            optimizableVariable('numEpoch',[200 501],'Type','integer')
            optimizableVariable('numRed',[1e-7 1e-2],'Transform','log')
            optimizableVariable('dropRate',[1e-7 1e-1],'Transform','log')
            optimizableVariable('decay',[1e-4 1e-1],'Transform','log')];


        optimVarsRetrain = [
            optimizableVariable('l2',[0.0001 0.00011],'Transform','log')
            optimizableVariable('mini',[1200 1201],'Type','integer')
            optimizableVariable('numEpoch',[500 501],'Type','integer')
            optimizableVariable('numRed',[1e-7 1e-2],'Transform','log')];
        trans = false;
        outputSizeDev = 1;
        normMu = [];
        normSigma = [];
    end

    methods
        function this = fullyConected(InputSize,optim)
            this.inputSize = InputSize;
            if ~exist('optim','var')
                %disp("No Model Built")
            else
                this.builtNet(optim);
            end
        end


        %%%%Only Base function
        function optimizefullyConected(this,paraStore,data,target,numEpochs)
            defaultParameterIn = [];
            defaultParameterIn.InputSize = this.inputSize;
            if this.outputSizeDev ~=1
                defaultParameterIn.OutputSize = this.outputSizeDev;
            end

            if ~exist('numEpochs','var')
                ObjFcn = fullyConectedPara(data.train,target.train,data.val,target.val,defaultParameterIn);
                BayesObject = bayesopt(ObjFcn,this.optimVars, ...
                'MaxObjectiveEvaluations',50,...
                'IsObjectiveDeterministic',false, ...
                'UseParallel',false);
                optim =  BayesObject.XAtMinObjective;
                this.builtNet(optim);
                save(paraStore,"optim")
            else
                ObjFcn = fullyConectedPara(data.train,target.train,data.val,target.val,defaultParameterIn);
                BayesObject = bayesopt(ObjFcn,this.optimVars, ...
                'MaxObjectiveEvaluations',numEpochs,...
                'IsObjectiveDeterministic',false, ...
                'UseParallel',false);
                optim =  BayesObject.XAtMinObjective;
                this.builtNet(optim);
                save(paraStore,"optim")
                
            end

        end



        %%%%%FullAdvanced
        function optimizefullyConectedAllSteps(this,paraStore,data,target,numEpochs)
            defaultParameterIn = [];
            defaultParameterIn.InputSize = this.inputSize;

            if this.outputSizeDev ~=1
                defaultParameterIn.OutputSize = this.outputSizeDev;
            end


            
            if ~exist('numEpochs','var')
                ObjFcn = fullyConectedAllPara(data.train,target.train,data.val,target.val,defaultParameterIn,data.retrain,target.retrain,data.retrainTest,target.retrainTest);

                BayesObject = bayesopt(ObjFcn,this.optimVarsAll, ...
                'MaxObjectiveEvaluations',50,...
                'IsObjectiveDeterministic',false, ...
                'UseParallel',false);
                optim =  BayesObject.XAtMinObjective;
                save(paraStore,"optim")
            else
                ObjFcn = fullyConectedAllPara(data.train,target.train,data.val,target.val,defaultParameterIn,data.retrain,target.retrain,data.retrainTest,target.retrainTest);

                BayesObject = bayesopt(ObjFcn,this.optimVarsAll, ...
                'MaxObjectiveEvaluations',numEpochs,...
                'IsObjectiveDeterministic',false, ...
                'UseParallel',false);
                optim =  BayesObject.XAtMinObjective;
                save(paraStore,"optim")
                
            end

        end



        %%%%%Only Retrain
        function optimizefullyConectedRetrain(this,paraStore,data,target,optimOld,numEpochs)
            defaultParameterIn = [];
            defaultParameterIn.InputSize = this.inputSize;

            if this.outputSizeDev ~=1
                defaultParameterIn.OutputSize = this.outputSizeDev;
            end

            defaultParameterIn.trans = true;
            disp("Init Model ....")
            netTemp = fullyConected(this.inputSize);
            netTemp.outputSizeDev = size(target.train,2);
            netTemp.builtNet(optimOld)
            netTemp.train(data.train,target.train);
            defaultParameterIn.netTemp = netTemp;
            disp("Init Model")
            if ~exist('numEpochs','var')
                ObjFcn = fullyConectedPara(data.train,target.train,data.val,target.val,defaultParameterIn,data.retrain,target.retrain,data.retrainTest,target.retrainTest);

                BayesObject = bayesopt(ObjFcn,this.optimVarsRetrain, ...
                'MaxObjectiveEvaluations',50,...
                'IsObjectiveDeterministic',false, ...
                'UseParallel',false);
                optim =  BayesObject.XAtMinObjective;
                this.builtNet(optim);
                save(paraStore,"optim")
            else
                ObjFcn = fullyConectedPara(data.train,target.train,data.val,target.val,defaultParameterIn,data.retrain,target.retrain,data.retrainTest,target.retrainTest);

                BayesObject = bayesopt(ObjFcn,this.optimVarsRetrain, ...
                'MaxObjectiveEvaluations',numEpochs,...
                'IsObjectiveDeterministic',false, ...
                'UseParallel',false);
                optim =  BayesObject.XAtMinObjective;
                %this.builtNet(optim);
                save(paraStore,"optim")
                
            end

        end

        function builtNet(this,optim)
            this.builtFlag = true;


            inputshape = this.inputSize;
            dropOut = optim.dropOut;
            numConvs = optim.SectionDepth;
            widthFC = optim.numNeurons;
            outpuSize = this.outputSizeDev;
            input_shape = [inputshape 1];
        
            layers = [
                imageInputLayer(input_shape,'Normalization','none','NormalizationDimension','auto','name','InputLayer')
                this.fullyConUnitNorm(widthFC,'First')
                batchNormalizationLayer('Name',['BN1'])
                reluLayer('Name',['First relu fc'])

            ];
            %layers = [layers;batchNormalizationLayer('Name',['BN1'])];

            for i = 1:numConvs-1
                if i < (numConvs-1)/2
                    layers = [layers;this.fullyConUnitNorm(ceil(widthFC*(2^(1))),['FC', convertStringsToChars(num2str(i))])];
                    %layers = [layers;batchNormalizationLayer('Name',['BN1',convertStringsToChars(num2str(i))])];

                else
                    layers = [layers;this.fullyConUnitNorm(ceil(widthFC/(2^(1))),['FC', convertStringsToChars(num2str(i))])];
                    %layers = [layers;batchNormalizationLayer('Name',['BN1',convertStringsToChars(num2str(i))])];
                end


            end

            

            layers = [layers;dropoutLayer(dropOut,'name','DropOut1')];
            layers = [layers;this.fullyConUnit(outpuSize,'Final')];

            if this.regDev == true
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
                'InitialLearnRate',optim.InitialLearnRate, ...
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
        
        function layers = convolutionalUnit(this,numF,tag,stride,filterSize)
            decay = 0.1;
            layers = [
                convolution2dLayer([1,filterSize],numF,'Padding','same','Stride',[1,stride],'Name',[tag,'conv1'])
                batchNormalizationLayer('Name',[tag,'BN1'],'VarianceDecay',decay,'MeanDecay',decay)
                reluLayer('Name',[tag,'relu1'])
                convolution2dLayer([1,filterSize],numF,'Padding','same','Stride',[1,1],'Name',[tag,'conv2'])
                batchNormalizationLayer('Name',[tag,'BN2'],'VarianceDecay',decay,'MeanDecay',decay)
                reluLayer('Name',[tag,'relu2'])
            ];
        
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
            %[~,this.normMu,this.normSigma] = zscore(data,0,4);
            %data = (data-this.normMu)./this.normSigma;
            if this.builtFlag
                this.net = trainNetwork(data,target,this.lgraph,this.options);
            else
                disp("Model Not Built")
            end
        end



        function retrainFullOpt(this,data,target,optim)
            %[~,this.normMu,this.normSigma] = zscore(data,0,4);
            %data = (data-this.normMu)./this.normSigma;

            this.options.InitialLearnRate = optim.numRed;
            this.options.MaxEpochs = optim.numEpoch;
            this.options.MiniBatchSize = optim.mini;
            this.options.L2Regularization = optim.l2;

            netOld = layerGraph(this.net);

            this.net = trainNetwork(data,target,netOld,this.options);
        end

        function retrainFullOptDiffOutput(this,data,target,optim,outputSize)
            %[~,this.normMu,this.normSigma] = zscore(data,0,4);
            %data = (data-this.normMu)./this.normSigma;

            this.options.InitialLearnRate = optim.numRed;
            %this.options.MaxEpochs = optim.numEpoch;
            %this.options.MiniBatchSize = optim.mini;
            %this.options.L2Regularization = optim.l2;

            netOld = layerGraph(this.net);
            for j = size(netOld.Layers):-1:1
                if isa(netOld.Layers(j),'nnet.cnn.layer.FullyConnectedLayer')
                    newLayer = this.fullyConUnit(outputSize,'FinalNew');
                    if size(netOld.Layers(j).Bias,1) == outputSize
                        break
                    end
                    netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer(1));
                    break
                end
            end

            for j = size(netOld.Layers):-1:1
                if isa(netOld.Layers(j),'nnet.cnn.layer.BatchNormalizationLayer')
                    newLayer = setfield(netOld.Layers(j,1),'MeanDecay',optim.decay);
                    netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer);
                    newLayer = setfield(netOld.Layers(j,1),'VarianceDecay',optim.decay);
                    netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer);
                end
            end
            this.net = trainNetwork(data,target,netOld,this.options);
        end

        function retrainPartOpt(this,data,target,optim)
            %[~,this.normMu,this.normSigma] = zscore(data,0,4);
            %data = (data-this.normMu)./this.normSigma;

            this.options.InitialLearnRate = optim.numRed;
            this.options.MaxEpochs = optim.numEpoch;
            this.options.MiniBatchSize = optim.mini;
            this.options.L2Regularization = optim.l2;

            netOld = layerGraph(this.net);
            skip = 0;
            for j = size(netOld.Layers):-1:1
                if isa(netOld.Layers(j),'nnet.cnn.layer.FullyConnectedLayer')
                    if skip >0
                        skip = skip-1;
                    else
                        newLayer = setfield(netOld.Layers(j,1),'WeightLearnRateFactor',0);
                        netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer);
                        newLayer = setfield(netOld.Layers(j,1),'BiasLearnRateFactor',0);
                        netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer);
                    end

                end

                if isa(netOld.Layers(j),'nnet.cnn.layer.BatchNormalizationLayer')
                    if skip >0
                        skip = skip-1;
                    else
                        newLayer = setfield(netOld.Layers(j,1),'ScaleLearnRateFactor',0);
                        netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer);
                        newLayer = setfield(netOld.Layers(j,1),'OffsetLearnRateFactor',0);
                        netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer);
                        newLayer = setfield(netOld.Layers(j,1),'MeanDecay',0.00001);
                        netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer);
                        newLayer = setfield(netOld.Layers(j,1),'VarianceDecay',0.00001);
                        netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer);
                    end

                end
            end

            this.net = trainNetwork(data,target,netOld,this.options);
        end


        function pred = apply(this,data)
            
            %data = (data-this.normMu)./this.normSigma;
            pred = predict(this.net,data);
        end

        function activationMap = customOcclusion(this,dataTrain,dataTest,method)
            
            activationMap = zeros(size(dataTest));
            meanTrainSample = mean(dataTrain,4);
            meanConz = mean(activations(this.net,dataTrain,"Finalrelu fc"));

            kernelHalf = 15;
            stride = 10;

            for i = 1:size(dataTest,4)
                dataOcc = dataTest(:,:,:,i);
                aOrg = activations(this.net,dataOcc,"Finalrelu fc");
                z = 1;
                dataOccAll = zeros(size(dataTrain,1),size(dataTrain,2),1,ceil(size(dataTrain,2)/stride)*4);
                
                for j = 1:size(dataTrain,1)
                    x = [];
                    for k = 1:ceil(size(dataTrain,2)/stride)
                        dataOcc = dataTest(:,:,:,i);
                        if method == "subsensor"
                            dataOcc(j,max(1,(k-1)*stride-kernelHalf):min(size(dataTrain,2),(k-1)*stride+kernelHalf+1)) = mean(dataTest(j,:,:,i),"all"); 
                        elseif method == "custom" 
                            dataOcc(j,max(1,(k-1)*stride-kernelHalf):min(size(dataTrain,2),(k-1)*stride+kernelHalf+1)) = meanTrainSample(j,max(1,(k-1)*stride-kernelHalf):min(size(dataTrain,2),(k-1)*stride+kernelHalf+1));
                        else 
                            dataOcc(j,max(1,(k-1)*stride-kernelHalf):min(size(dataTrain,2),(k-1)*stride+kernelHalf+1)) = mean(dataTest(:,:,:,i),"all"); 
                        end
                        dataOccAll(:,:,1,z) = dataOcc;
                        x = [x,min(max(1,(k-1)*stride),size(dataTrain,2))];
                        z = z+1;
                    end
                end
                aSingleOne = (squeeze(-activations(this.net,dataOccAll,"Finalrelu fc"))+aOrg)./(meanConz);
                aSingle = zeros(size(dataTrain,1),ceil(size(dataTrain,2)/stride));
                for lv1 = 1:size(dataTrain,1)
                    aSingle(lv1,:) = aSingleOne((ceil(size(dataTrain,2)/stride)*(lv1-1)+1):ceil(size(dataTrain,2)/stride)*lv1);
                end
                %disp(i)
                aSingleUp = zeros(size(dataTrain(:,:,1,1)));
                x2 = 1:size(dataTrain,2);
                for lv1 = 1:size(dataTrain,1)
                    aSingleUp(lv1,:,1,1) = interp1(x,double(aSingle(lv1,:,1,1)),x2,'spline');
                end
                activationMap(:,:,1,i) = aSingleUp;
            end
            if isempty(this.occVals)
                this.occVals = activationMap;
            else
                this.occVals = cat(4,this.occVals,activationMap);
            end
        end

        function test = shapVal(this,trainSamples,testSamples)
            
            dataTrain = zeros(size(trainSamples,4),size(trainSamples,2)*size(trainSamples,1));
            for i = 1:size(trainSamples,4)
                dataTemp=[trainSamples(1,:,1,i),trainSamples(2,:,1,i),trainSamples(3,:,1,i),trainSamples(4,:,1,i)];
                dataTrain(i,:) = dataTemp;
            end
            dataTest = zeros(size(testSamples,4),size(trainSamples,2)*size(trainSamples,1));
            for i = 1:size(testSamples,4)
                dataTemp=[testSamples(1,:,1,i),testSamples(2,:,1,i),testSamples(3,:,1,i),testSamples(4,:,1,i)];
                dataTest(i,:) = dataTemp;
            end
            
            test = cell(size(testSamples,4),1);
            sV = cell(size(testSamples,4),1);
            j = 1;
            for i = 1:size(dataTest,1)
                disp(j)
                test{j} = shapley(@(x)predictInner(this.net,x),dataTrain(1:end,:),'QueryPoint',dataTest(i,:),'UseParallel',false,'MaxNumSubsets',15000);
                sV{j} = test{j}.ShapleyValues.ShapleyValue';
                j = j+1;
            end

            if isempty(this.shapleyVal)
                this.shapleyVal = sV;
            else
                this.shapleyVal = [this.shapleyVal;sV];
            end

            function y = predictInner(net,x)
                dataArray = zeros(4,1440,1,size(x,1));
                for iInner = 1:size(x,1)
                    dataArray(1,:,1,iInner) = x(iInner,1:1440);
                    dataArray(2,:,1,iInner) = x(iInner,1441:2880);
                    dataArray(3,:,1,iInner) = x(iInner,2881:4320);
                    dataArray(4,:,1,iInner) = x(iInner,4321:5760);
                end
                y = predict(net,dataArray);
            end
        end

        function test = limeVal(this,trainSamples,testSamples)
            
            dataTrain = zeros(size(trainSamples,4),size(trainSamples,2)*size(trainSamples,1));
            for i = 1:size(trainSamples,4)
                dataTemp=[trainSamples(1,:,1,i),trainSamples(2,:,1,i),trainSamples(3,:,1,i),trainSamples(4,:,1,i)];
                dataTrain(i,:) = dataTemp;
            end
            dataTest = zeros(size(testSamples,4),size(trainSamples,2)*size(trainSamples,1));
            for i = 1:size(testSamples,4)
                dataTemp=[testSamples(1,:,1,i),testSamples(2,:,1,i),testSamples(3,:,1,i),testSamples(4,:,1,i)];
                dataTest(i,:) = dataTemp;
            end
            
            test = cell(size(testSamples,4),1);
            sV = cell(size(testSamples,4),2);
            j = 1;
            for i = 1:size(dataTest,1)
                disp(j)
                test{j} = lime(@(x)predictInner(this.net,x),dataTrain(1:end,:),'QueryPoint',dataTest(i,:),'Type','regression','NumImportantPredictors',2000);
                sV{j,2} = test{j}.SimpleModel.PredictorNames;
                sV{j,1} = test{j}.SimpleModel.Beta;
                j = j+1;
            end

            if isempty(this.limeVals)
                this.limeVals = sV;
            else
                this.limeVals = [this.limeVals;sV];
            end

            function y = predictInner(net,x)
                dataArray = zeros(4,1440,1,size(x,1));
                for iInner = 1:size(x,1)
                    dataArray(1,:,1,iInner) = x(iInner,1:1440);
                    dataArray(2,:,1,iInner) = x(iInner,1441:2880);
                    dataArray(3,:,1,iInner) = x(iInner,2881:4320);
                    dataArray(4,:,1,iInner) = x(iInner,4321:5760);
                end
                y = predict(net,dataArray);
            end
        end


        function test = gradVal(this,testSamples)
            
            test =zeros(size(testSamples,4),size(testSamples,1)*size(testSamples,2));
            for index = 1:size(testSamples,4)
                oneOps = gradientAttribution(this.net,testSamples(:,:,:,index),1,'Finalrelu fc',"autodiff");
                test(index,:) = [oneOps(1,:),oneOps(2,:),oneOps(3,:),oneOps(4,:)];
            end
            
             
            if isempty(this.gradMaps)
                this.gradMaps = test;
            else
                this.gradMaps = [this.gradMaps;test];
            end
            
            
            function map = gradientAttribution(net,img,YPred,softmaxName,method)
                
                lgraph = layerGraph(net);
                lgraph = removeLayers(lgraph,lgraph.Layers(end).Name);
                dlnet = dlnetwork(lgraph);
                
                % To use automatic differentiation, convert the image to a dlarray.
                dlImg = dlarray(single(img),"SSC");
                
                if method == "autodiff"
                % Use dlfeval and the gradientMap function to compute the derivative. The gradientMap
                % function passes the image forward through the network to obtain the class scores
                % and contains a call to dlgradient to evaluate the gradients of the scores with respect
                % to the image.
                dydI = dlfeval(@gradientMap,dlnet,dlImg,softmaxName,YPred);
                end
                
                if method == "guided-backprop"
                
                % Use the custom layer CustomBackpropReluLayer (attached as a supporting file)  
                % with a nonstandard backward pass, and use it with automatic differentiation.
                customRelu = CustomBackpropReluLayer();
                
                % Set the BackpropMode property of each CustomBackpropReluLayer to "guided-backprop".
                customRelu.BackpropMode = "guided-backprop";
                
                % Use the supporting function replaceLayersOfType to replace all instances of reluLayer in the network with
                % instances of CustomBackpropReluLayer. 
                lgraphGB = replaceLayersOfType(lgraph, ...
                    'nnet.cnn.layer.ReLULayer',customRelu);
                
                % Convert the layer graph containing the CustomBackpropReluLayers into a dlnetwork.
                dlnetGB = dlnetwork(lgraphGB);
                dydI = dlfeval(@gradientMap,dlnetGB,dlImg,softmaxName,YPred);
                end
                
                % Sum the absolute values of each pixel along the channel dimension, then rescale
                % between 0 and 1.
                map = sum(abs(extractdata(dydI)),3);
                map = rescale(map);
            end
            
            
            function dydI = gradientMap(dlnet,dlImgs,softmaxName,classIdx)
                
                dydI = dlarray(zeros(size(dlImgs)));
                
                for i=1:size(dlImgs,4)
                    I = dlImgs(:,:,:,i);
                    scores = predict(dlnet,I,'Outputs',{softmaxName});
                    classScore = scores(classIdx);
                    dydI(:,:,:,i) = dlgradient(classScore,I);
                end
            end
            
        end

    end
end

