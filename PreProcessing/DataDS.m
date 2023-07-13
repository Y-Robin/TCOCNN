function dataTrans = DataDS(dataTrain,targetTrain, dataTest, targetTest, numTrans, randomFlag)
    % DataDS performces direct standartization between master and slave
    % sensors (cf. https://www.mdpi.com/2073-4433/14/7/1123)
    % This is a commen task for calibration transfer of MOSGasSensors
    %
    % train: all samples (transformed) used for standardization
    % Test: transformed test Samples
    %
    % Data Train and targetTrain is used as the data of the master Sensor
    % numTrans specifies the amount of transfer UGMs
    % Transfer UGMs are selected randomly from the training set
    %If RandomFlag is False always the same samples will be use for
    %transfer
    % dataTest and Target Test is used as the slave sensor that is transformed
    % /standardize
    % Return dataTrans contains the standardized data of the slave sensor

    dataTrans = [];
    dataTrans.train = [];
    dataTrans.val = [];
    dataTrans.test = [];

    %Set RNG Seed 
    if randomFlag
        rng('shuffle')
    else
        rng(0);
    end

    %TransformationVar
    F = cell(1,size(1,size(dataTrain,1)));

    %Break down to single sensors Train
    dataTrainTrainCells = matrix4DtoCell(dataTrain.train);


    %Break down to single sensors Test
    dataTestTrainCells = matrix4DtoCell(dataTest.train);
    dataTestValCells = matrix4DtoCell(dataTest.val);
    dataTestTestCells = matrix4DtoCell(dataTest.test);

    %Set RNG Seed
    if randomFlag
        rng('shuffle')
    else
        rng(0);
    end

    UGMs = unique(targetTrain.train);
    UGMs = UGMs(randperm(length(UGMs)));
    targetUGMs = UGMs(1:numTrans); % Selected Transfer Samples
    % Ifmultiple parent sensors are available > then 1 
    %Sensors must be recorded within the same dataset or the same amount of UGMs mus be present
    multiplyerUGM = size(dataTrain.train,4)/size(dataTest.train,4); 

    masterSensorArray = cell(1,size(dataTrain.train,1));
    slaveSensorArray = cell(1,size(dataTrain.train,1));
    
    % Stack data of Subsensors to Matrix (Master and Slave)
    for i = 1:size(targetUGMs,1)
        for j = 1:size(masterSensorArray,2)
            masterSensorArray{1,j} = [masterSensorArray{1,j};dataTrainTrainCells{1,j}(targetTrain.train==targetUGMs(i),:)];
            for k = 1:multiplyerUGM
                slaveSensorArray{1,j} = [slaveSensorArray{1,j};dataTestTrainCells{1,j}(targetTest.train==targetUGMs(i),:)];
            end
        end
    end
    
    %Calculate for every sub-sensor the C parameter
    for i = 1:size(masterSensorArray,2)
        F{1,i} = masterSensorArray{1,i}'*pinv(slaveSensorArray{1,i}');
    end
    dataTransTrain = cell(1,size(dataTrain.train,1));
    dataTransVal = cell(1,size(dataTrain.train,1));
    dataTransTest = cell(1,size(dataTrain.train,1));
    %Transform Data
    for i = 1:size(dataTransTrain,2)
       dataTransTrain{1,i} =  dataTestTrainCells{1,i}*F{1,i}';
       dataTransVal{1,i} =  dataTestValCells{1,i}*F{1,i}';
       dataTransTest{1,i} =  dataTestTestCells{1,i}*F{1,i}';
    end
    dataTrans.train = CelltoMatrix4D(dataTransTrain);
    dataTrans.val = CelltoMatrix4D(dataTransVal);
    dataTrans.test = CelltoMatrix4D(dataTransTest);
    
end

function dataCell = matrix4DtoCell(data)
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

function testDataMat4D = CelltoMatrix4D(data)
    %Convert Cell array to Matrix to be compatible with the
    %toolbox
    testDataMat4D = zeros(size(data,2),size(data{1,1},2),1,size(data{1,1},1));
    for i = 1:size(data{1,1},1)
        for lvSubSens = 1:size(data,2)
            testDataMat4D(lvSubSens,:,1,i) = data{1,lvSubSens}(i,:); 
        end
    end
end

