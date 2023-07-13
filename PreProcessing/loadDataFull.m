function [data,target] = loadDataFull(loadStruct)

    % Output specific Szenarios based on the choosen Method
    %M ethod 1: This method does a 80-20 split of Unique Gas Mixtures (UGMs)
    %for training and testing respectively. From the 80 % Training 10 % is
    % put into validation

    %Method 2: This method performs a transfer learning data split. 20% of 
    % the UGMs are reserved for testing (random), while 80% are used for training/validation,
    % with a specific percentage of the training/validation data determined by the transf field of loadStruct.
    % With the random Flag False this method will always the same subset
    % for training
    % The subset for testing is always the same

    %Method 3: This method uses the first X% of the data to "drift" or change
    % the model over time. The percentage used for training/validation is 
    % defined by driftTrain and the percentage for testing by driftTest in loadStruct.

    %Method 4: This method splits the data based on measurements, which then needs to be a field in the loaded target.
    % With trainMeas = loadStruct.measurements{1}; and testMeas =
    % loadStruct.measurements{2}; it is specifed which portion of the data
    % is used for training and which for testing
    % loadStruct.measurements{1}/loadStruct.measurements{2}

    %Method 5: This method appears to be specific to field test data, using
    % the fieldtest property of the targets to split the data.

    %Target: has to be a struct with the field of the target gas and range,
    %measurement and fieldtest

    %Data: has to be of the form of every sensor beeing placed in a Matrix
    %(same size) in one file. See Example

    %LoadStruct essential fields
    % fileNameDataAll: Name of datafiles (iteratable)
    % fileNameTargetAll: Name of targetfiles (iteratable)
    % targetGas: TargetGasName/Names
    % RandomFlag. Tertemines reproducability
    % dataSize: Expected datasize
    % normFlag: If true zscore is performed on every subsensor per
    % observation
    % numRetTar: Instead of a specific Target a array can be specified to select multiple targets 
    % saveName: the data is also stored not only a return value



    % Extras:
    % 1. Always based on UGMs (Unique Gas mixtures)
    % 2. Load multiple Files
    % 3. Spezify Input Size
    % 4. Randomize splits (Adds extra randomness, can be disabled for reproduzability)
    % 5. Load with occlusion

    methodNames = ["Split","Transfer","Drift","Measurment","Fieldtests"];
    
    %Extract Struct
    %File related
    fileNameDataAll = loadStruct.fileNameDataAll;
    fileNameTargetAll =  loadStruct.fileNameTargetAll;
    targetGas =  loadStruct.targetGas;
    
    %Used Method
    loadMethod = loadStruct.loadMethod;
    
    %Expected Input Dimensions
    dataSize = loadStruct.dataSize;
    numSubSenors = dataSize(1);
    numSamples = dataSize(2);

    %Random Flag
    randomFlag = loadStruct.randomFlag;

    %NormalizeInput 
    normFlag = loadStruct.normFlag;

    occlusionFlag = loadStruct.OcclusionFlag;
    
    %Empty Storage for Output
    target.train = [];
    target.val = [];
    target.test = [];
    
    data.train = [];
    data.val = [];
    data.test = [];
    
    % To garantee the same samples from multiple sensors from the same
    % measurment 
    if randomFlag
        rng('shuffle')
        rngVal = randi([1,1000]);
    end


    for k = 1:size(fileNameDataAll,1)

        fileNameData = fileNameDataAll(k,:);
        fileNameTarget = fileNameTargetAll(k,:);
    
        
        
        %Disp used Setup
        disp(fileNameData)
        disp(targetGas)
        disp(methodNames(loadMethod))
        disp("RandomFlag")
        disp(randomFlag)

        % Load Data
        sensorStruct = load(fileNameData);
        load(fileNameTarget);
        sensorCell = struct2cell(sensorStruct);

        %Zielgröße If numRetTar is set multiple targets can be returnd
        if ~isfield(loadStruct,"numRetTar")
            targetGasT= targets.(targetGas);
        else
            tempTarget = struct2cell(targets);
            targetGasT= cell2mat(tempTarget(loadStruct.numRetTar)');
        end
        % Bring data in expected form
        for lvSubSens = 1:numSubSenors
            if numSamples ~= size(sensorCell{lvSubSens},2)
                x = 1:size(sensorCell{lvSubSens},2);
                y = 1:size(sensorCell{lvSubSens},2)/numSamples:size(sensorCell{lvSubSens},2);
                y = linspace(1,size(sensorCell{lvSubSens},2),6000);
                dataT = zeros(size(sensorCell{lvSubSens},1),numSamples);
                for j = 1:size(sensorCell{1},1)
                    dataT(j,:)  = interp1(x,sensorCell{lvSubSens}(j,:),y,'linear','extrap');
                end
                sensorCell{lvSubSens} = dataT;
            end
        end
    
        %Normalize data
        if normFlag
            for lvSubSens = 1:numSubSenors
                sensorCell{lvSubSens} = zscore(sensorCell{lvSubSens},0,2);
            end
        end
        
        %Go Through Method
        if loadMethod == 1

            %Ranges resambles Unique Gas Mixtures
            %Find all valid Samples (not nan)
            ranges =  targets.range(~isnan(targets.range));

            %Extract UGMs
            uRanges = unique(ranges);

            %Shuffel UGMs (True: data is always random, False: Still shuffeld but always with the same seed)
            if randomFlag
                rng(rngVal);
            else
                rng(0);
            end
            uRanges =  uRanges(randperm(length(uRanges)));

            %Specify Index For Remaining, Train, Validation, and Test 
            ind = ~isnan(targets.range);
            ind2 = false(size(ind,1),1);
            ind3 = false(size(ind,1),1);
            ind4 = false(size(ind,1),1);
            %80 - 20 Split
            totalLength = ceil(length(uRanges));
            testSet = floor(0.2*size(uRanges,1));
            secTrain = uRanges(testSet+1:totalLength);



        elseif loadMethod == 2

            %Ranges resambles Unique Gas Mixtures
            %Find all valid Samples (not nan)
            ranges =  targets.range(~isnan(targets.range));

            %Extract UGMs
            uRanges = unique(ranges);

            %Shuffel UGMs : Fixed seed to always get same test Data
            rng(0);

            uRanges =  uRanges(randperm(length(uRanges)));

            %Specify Index For Remaining, Train, Validation, and Test 
            ind = ~isnan(targets.range);
            ind2 = false(size(ind,1),1);
            ind3 = false(size(ind,1),1);
            ind4 = false(size(ind,1),1);
            % 20% Test
            totalLength = ceil(size(uRanges,1));
            testSet = floor(0.2*size(uRanges,1));
            if randomFlag
                rng(rngVal);
                secTrain = uRanges(testSet+1:totalLength);
                secTrain = secTrain(randperm(length(secTrain)));
                secTrain = secTrain(1:ceil(totalLength*loadStruct.transf));
            else
                secTrain = uRanges(testSet+1:totalLength);
                secTrain = secTrain(1:loadStruct.transf);
            end

            
        elseif loadMethod == 3
            %Ranges resambles Unique Gas Mixtures
            %Find all valid Samples (not nan)
            ranges =  targets.range(~isnan(targets.range));

            %Extract UGMs
            uRanges = unique(ranges);
            uRanges = flip(uRanges);
            %Shuffel UGMs (True: data is always random, False: Still shuffeld but always with the same seed)

            %Specify Index For Remaining, Train, Validation, and Test 
            ind = ~isnan(targets.range);
            ind2 = false(size(ind,1),1);
            ind3 = false(size(ind,1),1);
            ind4 = false(size(ind,1),1);
            %80 - 20 Split

            totalLength = ceil(size(uRanges,1));
            testSet = floor(loadStruct.driftTest*size(uRanges,1));
            secTrain = uRanges(ceil((1-loadStruct.driftTrain)*totalLength):end);
            if randomFlag
                rng(rngVal);
                secTrain = secTrain(randperm(length(secTrain)));
            end

        elseif loadMethod == 4
            %Ranges resambles Unique Gas Mixtures
            %Find all valid Samples (not nan)
            ranges =  targets.range(~isnan(targets.range))';
            
            % which measurment to use
            trainMeas = loadStruct.measurements{1};
            testMeas = loadStruct.measurements{2};

            %Shuffel UGMs (True: data is always random, False: Still shuffeld but always with the same seed)

            %Specify Index For Remaining, Train, Validation, and Test 
            ind = ~isnan(targets.range);
            ind2 = false(size(ind,1),1);
            ind3 = false(size(ind,1),1);
            ind4 = false(size(ind,1),1);
            %80 - 20 Split
            
            uRanges = [];
            for indM = 1:size(testMeas,2)
                uRanges = [uRanges;ranges(targets.measurement==testMeas(indM))'];
            end
            uRanges = unique(uRanges);
            testSet = size(uRanges,1);
            secTrain = [];
            for indM = 1:size(trainMeas,2)
                secTrain = [secTrain;ranges(targets.measurement==trainMeas(indM))'];
            end
            secTrain = unique(secTrain);

        elseif loadMethod == 5

            %Ranges resambles Unique Gas Mixtures
            %Find all valid Samples (not nan)
            ranges =  targets.fieldtest(~isnan(targets.fieldtest));

            %Extract UGMs
            uRanges = unique(ranges);

            %Shuffel UGMs (True: data is always random, False: Still shuffeld but always with the same seed)
            if randomFlag
                rng(rngVal);
            else
                rng(0);
            end
            %uRanges =  uRanges(randperm(length(uRanges)));

            %Specify Index For Remaining, Train, Validation, and Test 
            ind = ~isnan(targets.fieldtest);
            ind2 = false(size(ind,1),1);
            ind3 = false(size(ind,1),1);
            ind4 = false(size(ind,1),1);
            %80 - 20 Split
            totalLength = ceil(size(uRanges,1));
            testSet = floor(0.99*size(uRanges,1));
            secTrain = uRanges(testSet+1:totalLength);
            targets.range = targets.fieldtest;
        end


        % Create Data Arrays
        % Exclude 10% for validation from training
        for lv1 = 1:size(secTrain,1)
            if mod(lv1,8)>0
                ind2(targets.range == secTrain(lv1)) = ind(targets.range == secTrain(lv1));
                ind(targets.range == secTrain(lv1)) = false;
            else
                ind3(targets.range == secTrain(lv1)) = ind(targets.range == secTrain(lv1));
                ind(targets.range == secTrain(lv1)) = false;
            end
        end
        
        % CreateTestSet
        for lv1 = 1:testSet
            ind4(targets.range == uRanges(lv1)) = ind(targets.range == uRanges(lv1));
            ind(targets.range == uRanges(lv1)) = false;
        end

        %Alwyays Exclude the first 3 Observations
        ind4(targets.range == 1|targets.range == 2|targets.range == 3) = false;
        ind3(targets.range == 1|targets.range == 2|targets.range == 3) = false;
        ind2(targets.range == 1|targets.range == 2|targets.range == 3) = false;
        
        %Create Targets
        trainTarget = targetGasT(ind2,:);
        valTarget =  targetGasT(ind3,:);
        testTarget = targetGasT(ind4,:);
    
        %Create 4D Array Train
        trainDataMat3 = cell(1,numSubSenors);
        for lvSubSens = 1:numSubSenors
            trainDataMat3{1,lvSubSens} = sensorCell{lvSubSens}(ind2,:);
        end
        trainDataMat4D = zeros(numSubSenors,size(trainDataMat3{1,1},2),1,size(trainDataMat3{1,1},1));
        for i = 1:size(trainDataMat3{1,1},1)
            for lvSubSens = 1:numSubSenors
                trainDataMat4D(lvSubSens,:,1,i) = trainDataMat3{1,lvSubSens}(i,:); 
            end
        end
    
        %Create 4D Array Validation
        valDataMat3 = cell(1,numSubSenors);
        for lvSubSens = 1:numSubSenors
            valDataMat3{1,lvSubSens} = sensorCell{lvSubSens}(ind3,:);
        end
        valDataMat4D = zeros(numSubSenors,size(valDataMat3{1,1},2),1,size(valDataMat3{1,1},1));
        for i = 1:size(valDataMat3{1,1},1)
            for lvSubSens = 1:numSubSenors
                valDataMat4D(lvSubSens,:,1,i) = valDataMat3{1,lvSubSens}(i,:); 
            end
        end

        %Create 4D Array Test
        testDataMat3 = cell(1,numSubSenors);
        for lvSubSens = 1:numSubSenors
            testDataMat3{1,lvSubSens} = sensorCell{lvSubSens}(ind4,:);
        end
        testDataMat4D = zeros(numSubSenors,size(testDataMat3{1,1},2),1,size(testDataMat3{1,1},1));
        for i = 1:size(testDataMat3{1,1},1)
            for lvSubSens = 1:numSubSenors
                testDataMat4D(lvSubSens,:,1,i) = testDataMat3{1,lvSubSens}(i,:); 
            end
        end
    
        target.train = [target.train;trainTarget];
        target.val = [target.val;valTarget];
        target.test = [target.test;testTarget];
    
        data.train = cat(4,data.train,trainDataMat4D);
        data.val = cat(4,data.val,valDataMat4D);
        data.test = cat(4,data.test,testDataMat4D);

        if occlusionFlag
            %%ToDo
        end
    end
    if loadStruct.saveFlag
        save(loadStruct.saveName,"data","target")
    end
end

