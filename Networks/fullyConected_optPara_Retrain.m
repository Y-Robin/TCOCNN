function ObjFcn = fullyConected_optPara_Retrain(XTrain,YTrain,XValidation,YValidation,defaultParameter,dataTransTrain,targetTransTrain,dataTransTest,targetTransTest)
    ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
        if gpuDeviceCount > 0
            g = gpuDevice();
            reset(g)
        end
        valError = 0;

        if defaultParameter.Regression == false
            for i = 1:3
                Net = fullyConected(defaultParameter.InputSize,defaultParameter.OutputSize,defaultParameter.Regression,optVars);
                Net.train(XTrain,YTrain);
                [~,label] = max(Net.apply(XValidation),[],2);
                valError = valError+helpers.ClassificationError.loss(categorical(label),YValidation);

            end
            valError = valError/3;
            cons = [];
            fileName = num2str(valError) + ".mat";
        else
            for i = 1:3

                Net = fullyConected(defaultParameter.InputSize,defaultParameter.OutputSize,defaultParameter.Regression,optVars);
                Net.train(XTrain,YTrain);
                Net.retrainFull(dataTransTrain,targetTransTrain,optVars)
                pred = Net.apply(dataTransTest);
                valError = valError + sum(helpers.RMSE.loss(pred,targetTransTest));
            end
            valError = valError/3;
            cons = [];
            fileName = num2str(valError) + ".mat";
        end

    end
end

