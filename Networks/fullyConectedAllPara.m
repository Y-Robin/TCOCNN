function ObjFcn = fullyConectedAllPara(XTrain,YTrain,XValidation,YValidation,defaultParameterIn,dataTransTrain,targetTransTrain,dataTransTest,targetTransTest)
    ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)

        
        if isfield(defaultParameterIn,'OutputSize')
            imageSize = defaultParameterIn.InputSize;
            cnn2DNet = fullyConected(imageSize);
            cnn2DNet.outputSizeDev = defaultParameterIn.OutputSize;
            cnn2DNet.builtNet(optVars)
            cnn2DNet.train(XTrain,YTrain);
            [valPred,label] = max(cnn2DNet.apply(XValidation),[],2);
            valError = helpers.ClassificationError.loss(categorical(label),YValidation);
            cons = [];
            fileName = num2str(valError) + ".mat";
        else
            valError = 0;
            g = gpuDevice();
            reset(g);
            for i = 1:3
                %disp(['test' convertStringsToChars(num2str(i))])
                imageSize = defaultParameterIn.InputSize;
                cnn2DNet = fullyConected(imageSize);
                cnn2DNet.outputSizeDev = size(YTrain,2);
                cnn2DNet.builtNet(optVars)
                %net.options.ValidationData = {dataTransTrain,targetTransTrain};
                %net.options.OutputNetwork = 'best-validation-loss';
                cnn2DNet.train(XTrain,YTrain);
                cnn2DNet.retrainFullOptDiffOutput(dataTransTrain,targetTransTrain,optVars,size(targetTransTrain,2))
                pred = cnn2DNet.apply(dataTransTest);
                valError = valError + helpers.RMSE.loss(pred,targetTransTest);
            end
            valError = valError/3;
            cons = [];
            fileName = num2str(valError) + ".mat";
        end

    end
end

