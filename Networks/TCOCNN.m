classdef TCOCNN<handle & matlab.mixin.Copyable 

    properties
        %Network for training
        lgraph = [];
        options = [];
        net = [];
        inputSize = [];
        outputSize = [];
        regression = true;
        builtFlag = false;

        %XAI Parameters
        shapleyVal = [];
        occVals = [];
        limeVals = [];
        gradMaps = [];
        

        %Specifies the search parameters for Bayesian optimization (Simple Hyperparameter)
        optimVars = [
            optimizableVariable('n_filter',[50 150],'Type','integer')
            optimizableVariable('SectionDepth',[3 5],'Type','integer')
            optimizableVariable('InitialLearnRate_Beginning',[1e-5 1e-3],'Transform','log')
            optimizableVariable('stride',[5 35],'Type','integer')
            optimizableVariable('kernel',[15 100],'Type','integer')
            optimizableVariable('numNeurons',[800 1500],'Type','integer')
            optimizableVariable('dropOut',[0.1 0.5],'Type','real')];
        %Specifies the search parameters for Bayesian optimization for
        %retraining
        optimVarsRetraining = [
            optimizableVariable('n_filter',[50 150],'Type','integer')
            optimizableVariable('SectionDepth',[3 5],'Type','integer')
            optimizableVariable('InitialLearnRate_Beginning',[1e-5 1e-3],'Transform','log')
            optimizableVariable('stride',[5 35],'Type','integer')
            optimizableVariable('kernel',[15 100],'Type','integer')
            optimizableVariable('numNeurons',[800 1500],'Type','integer')
            optimizableVariable('dropOut',[0.1 0.5],'Type','real')
            optimizableVariable('InitialLearnRate',[1e-7 1e-3],'Type','real')%retraining
            optimizableVariable('MaxEpochs',[50 100],'Type','integer')];
        
    end

    methods
        function this = TCOCNN(InputSize,Outputsize,Regression,optim)
            this.inputSize = InputSize;
            this.outputSize = Outputsize;
            this.regression = Regression;
            if ~exist('optim','var')
                disp("No Model Built")
            else
                this.builtNet(optim);
            end
        end

        %Builds the network based on all parametersspecified in optim
        function builtNet(this,optim)
            %Model can now be trained
            this.builtFlag = true;

            %Extract optim Params
            n_filters = optim.n_filter;
            firstKernel_size = optim.kernel;
            firstStride = optim.stride;
            dropOut = optim.dropOut;
            numConvs = optim.SectionDepth;
            widthFC = optim.numNeurons;
            
            %Default In Out
            input_shape = [this.inputSize 1];
            outpuSize = this.outputSize;
            
            %Fixed Kernel Sizes 
            kernel_size = 2;
            stride_size = 2;
            
            %InputLayer
            layers = [
                imageInputLayer(input_shape,'Normalization','none','NormalizationDimension','auto','name','InputLayer')
                this.convolutionalUnit(n_filters,'First Block',firstStride,firstKernel_size)
            ];
            %Remaining convolutional layer 
            for i = 1:numConvs-1
                layers = [layers;this.convolutionalUnit(n_filters*(i+1),['Block',int2str(i)],stride_size,kernel_size)];
            end
            
            %Final two FC Layer
            layers = [layers;this.fullyConUnit(widthFC,'First')];
            layers = [layers;dropoutLayer(dropOut,'name','DropOut1')];
            layers = [layers;this.fullyConUnit(outpuSize,'Final')];
            
            %Specify if regression or classification
            if this.regression == true
                layers = [layers;regressionLayer('Name','routput')];
            else
                layers = [layers;softmaxLayer('Name','softmax')];
                layers = [layers;classificationLayer('Name','classoutput')];
            end
            

            this.lgraph = layerGraph(layers);
            
            % training options
            this.options = trainingOptions('adam', ...
                'ExecutionEnvironment',"auto",...
                'InitialLearnRate',optim.InitialLearnRate_Beginning, ...
                'MaxEpochs',75, ...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropPeriod',2, ...
                'LearnRateDropFactor',0.9, ...
                'MiniBatchSize',50, ...
                'L2Regularization',0.0001, ...
                'Shuffle','every-epoch', ...
                'Verbose',false, ...
                'Plots','none', ...
                'ValidationFrequency',50);

        end
        
        %Used Structurefor convolutional layer
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
        
        %Used Structurefor FC Layer
        function layers = fullyConUnit(this,numF,tag)
        
            layers = [
                fullyConnectedLayer(numF,'Name',[tag 'fc'])
                reluLayer('Name',[tag 'relu fc'])
            ];
        
        end
        
        %This function is used to optimize the TCOCNN based on optimVars,
        %the training and validation data
        function optimizeTCOCNN(this,paraStore,data,target,numEpochs)
            defaultParameter = [];
            defaultParameter.InputSize = this.inputSize;
            defaultParameter.OutputSize = this.outputSize;
            defaultParameter.Regression = this.regression;


            if ~exist('numEpochs','var')
                numEpochs = 50;
            end

            ObjFcn = TCOCNN_OptPara(data.train,target.train,data.val,target.val,defaultParameter);
            BayesObject = bayesopt(ObjFcn,this.optimVars, ...
            'MaxObjectiveEvaluations',numEpochs,...
            'IsObjectiveDeterministic',false, ...
            'UseParallel',false, "ConditionalVariableFcn",@fitcdiscrCVF);
            optim =  bestPoint(BayesObject);
            this.builtNet(optim);
            save(paraStore,"optim")

        end

        function optimizeTCOCNNRetrain(this,paraStore,data,target,numEpochs)
            defaultParameter = [];
            defaultParameter.InputSize = this.inputSize;
            defaultParameter.OutputSize = this.outputSize;
            defaultParameter.Regression = this.regression;


            if ~exist('numEpochs','var')
                numEpochs = 50;
            end

            ObjFcn = TCOCNN_OptPara_Retrain(data.train,target.train,data.val,target.val,defaultParameter,data.retrain,target.retrain,data.retrainVal,target.retrainVal);
            BayesObject = bayesopt(ObjFcn,this.optimVarsRetraining, ...
            'MaxObjectiveEvaluations',numEpochs,...
            'IsObjectiveDeterministic',false, ...
            'UseParallel',false, "ConditionalVariableFcn",@fitcdiscrCVF);
            optim =  bestPoint(BayesObject);
            this.builtNet(optim);
            save(paraStore,"optim")

        end
        
        %If all hyperparameter are set it is possible to train the model
        function train(this,data,target)
            if this.builtFlag
                this.net = trainNetwork(data,target,this.lgraph,this.options);
            else
                disp("Model Not Built")
            end
        end
        
        %retrain the entire TCOCNN with a new learning rate
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

        
        %All trainable parameter in the feature extrection section freezed 
        function retrainPar(this,data,target,retrainOpt)
            fn = fieldnames(retrainOpt);
            fo = fieldnames(this.options);

            for k=1:numel(fn)
                for i = 1:numel(fo)
                    if strcmp(fo{i},fn{k})
                        this.options.(fn{k}) = retrainOpt.(fn{k});
                    end
                end
            end
            
            netOld = layerGraph(this.net);
            for j = 1:size(netOld.Layers)
                if isa(netOld.Layers(j),'nnet.cnn.layer.Convolution2DLayer')
                    newLayer = setfield(netOld.Layers(j,1),'WeightLearnRateFactor',0);
                    netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer);
                    newLayer = setfield(netOld.Layers(j,1),'BiasLearnRateFactor',0);
                    netOld = replaceLayer(netOld,netOld.Layers(j,1).Name,newLayer);

                end
                if isa(netOld.Layers(j),'nnet.cnn.layer.BatchNormalizationLayer')
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
            this.net = trainNetwork(data,target,netOld,this.options);

        end
        
        %Predict new samples
        function pred = apply(this,data)
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

