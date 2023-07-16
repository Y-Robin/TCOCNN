function ObjFcn = fullyConected_OptPara(XTrain,YTrain,XValidation,YValidation,defaultParameter)
    ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
        if gpuDeviceCount > 0
            g = gpuDevice();
            reset(g)
        end
        
        if defaultParameter.Regression == false
            Net = fullyConected(defaultParameter.InputSize,defaultParameter.OutputSize,defaultParameter.Regression,optVars);
            Net.train(XTrain,YTrain);
            [~,label] = max(Net.apply(XValidation),[],2);
            valError = helpers.ClassificationError.loss(categorical(label),YValidation);
            cons = [];
            fileName = num2str(valError) + ".mat";
        else
            Net = fullyConected(defaultParameter.InputSize,defaultParameter.OutputSize,defaultParameter.Regression,optVars);
            Net.train(XTrain,YTrain);
            valPred = Net.apply(XValidation);
            valError = sum(helpers.RMSE.loss(valPred,YValidation));
            cons = [];
            fileName = num2str(valError) + ".mat";
        end

    end
end

