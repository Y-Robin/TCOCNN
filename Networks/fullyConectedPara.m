function ObjFcn = fullyConectedPara(XTrain,YTrain,XValidation,YValidation,defaultParameterIn,dataTransTrain,targetTransTrain,dataTransTest,targetTransTest)
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
            g = gpuDevice();
            reset(g);
            if isfield(defaultParameterIn,'trans')
                
                cnn2DNet = copy(defaultParameterIn.netTemp);
               
                %disp(helpers.RMSE.loss(cnn2DNet.apply(XTrain),YTrain))
                cnn2DNet.retrainFullOptDiffOutput(dataTransTrain,targetTransTrain,optVars,size(targetTransTrain,2));
                valPred = cnn2DNet.apply(dataTransTest);
                valError = helpers.RMSE.loss(valPred,targetTransTest);
                disp(valError)
                %figure()
                %scatter(valPred,targetTransTest)
                cons = [];
                fileName = num2str(valError) + ".mat";
            else
                imageSize = defaultParameterIn.InputSize;
                cnn2DNet = fullyConected(imageSize,optVars);
                cnn2DNet.train(XTrain,YTrain);
                valPred = cnn2DNet.apply(XValidation);
                valError = helpers.RMSE.loss(valPred,YValidation);
                cons = [];
                fileName = num2str(valError) + ".mat";
            end
        end

    end
end

