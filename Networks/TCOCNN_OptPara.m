function ObjFcn = TCOCNN_OptPara(XTrain,YTrain,XValidation,YValidation,defaultParameter)
    ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
        g = gpuDevice();
        reset(g)
        
        if defaultParameter.Regression == false
            cnn2DNet = TCOCNN(defaultParameter.InputSize,defaultParameter.OutputSize,defaultParameter.Regression,optVars);
            cnn2DNet.train(XTrain,YTrain);
            [valPred,label] = max(cnn2DNet.apply(XValidation),[],2);
            valError = helpers.ClassificationError.loss(categorical(label),YValidation);
            cons = [];
            fileName = num2str(valError) + ".mat";
        else
            cnn2DNet = TCOCNN(defaultParameter.InputSize,defaultParameter.OutputSize,defaultParameter.Regression,optVars);
            cnn2DNet.train(XTrain,YTrain);
            valPred = cnn2DNet.apply(XValidation);
            valError = sum(helpers.RMSE.loss(valPred,YValidation));
            cons = [];
            fileName = num2str(valError) + ".mat";
        end

    end
end

