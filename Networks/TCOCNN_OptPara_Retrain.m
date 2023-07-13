function ObjFcn = TCOCNN_OptPara_Retrain(XTrain,YTrain,XValidation,YValidation,defaultParameter,dataTransTrain,targetTransTrain,dataTransTest,targetTransTest)
    ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
        g = gpuDevice();
        reset(g)
        valError = 0;

        if defaultParameter.Regression == false
            for i = 1:3
                cnn2DNet = TCOCNN(defaultParameter.InputSize,defaultParameter.OutputSize,defaultParameter.Regression,optVars);
                cnn2DNet.train(XTrain,YTrain);
                [~,label] = max(cnn2DNet.apply(XValidation),[],2);
                valError = valError+helpers.ClassificationError.loss(categorical(label),YValidation);

            end
            valError = valError/3;
            cons = [];
            fileName = num2str(valError) + ".mat";
        else
            for i = 1:3

                cnn2DNet = TCOCNN(defaultParameter.InputSize,defaultParameter.OutputSize,defaultParameter.Regression,optVars);
                cnn2DNet.train(XTrain,YTrain);
                cnn2DNet.retrainFull(dataTransTrain,targetTransTrain,optVars)
                pred = cnn2DNet.apply(dataTransTest);
                valError = valError + sum(helpers.RMSE.loss(pred,targetTransTest));
            end
            valError = valError/3;
            cons = [];
            fileName = num2str(valError) + ".mat";
        end

    end
end

