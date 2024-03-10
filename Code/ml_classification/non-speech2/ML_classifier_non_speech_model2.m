% -----------------> ML_classifier_non_speech_model2.m <------------------- 

% The following code uses the model GMMs trained on all the MFCCs of
% the ccnu_mobile dataset and Maximum Likelihood classification to classify each 
% recording into one of the 21 models. The classification is done using the 
% negative log likelihood, that is each recording is classified into the 
% model whose GMM gives the smallest negative log likelihood value 
% on the recording's MFCCs

% go to source directory
cd("..\..")

phones = {'AIR1', 'AIR2_1', 'AIR2_2', 'iPAD7', 'iPH6_1', 'iPH6_2', 'iPH6_3', 'iPH6_4', 'iPH6S_1', 'iPH6S_2', 'iPH6S_3', ...
          'iPH7P', 'iPHSE', 'iPHX', 'H10', 'H7X', 'H8_1', 'H8_2', 'H8_3', 'H9', 'HV8', 'HUAWN', 'HUAWN2S', 'HUAWN3E', ...
          'HUAWP10', 'HUAWP20', 'HUAWTAG', 'NZ11', 'OR9S', 'S8', 'SPH', 'VX3F', 'VX7', 'VY11T', 'MI2S', 'MI5', 'MI8', ...
          'MI8SE_1', 'MI8SE_2', 'MIX2', 'R3S', 'RNOTE3', 'RNOTE4X', 'ZC880A', 'ZG719C'};

% number of wav files per phone
samples = 3;
% non-speech / audio folder
f = "non-speech";

ground_truth = repelem(1:45, samples);

% load test set
% allocate space for the MFCCs matrices
mfccs = cell(45 * 3, 1);
mfccs_counter = 1;
% for each phone
for i = 1:45
    folder = "data2\MFCCs\" + f + "\" + phones{i} + "\";
    d = dir(folder);
    
    % for each speaker
    for j = 1:3
        % skip the first two folder (. and ..) and the first two train
        % speakers
        filename = folder + d(j + 4).name;
        mfccs{mfccs_counter} = importdata(filename);
        mfccs_counter = mfccs_counter + 1;
    end
end

% allocate space for the phone GMMs for each number of components
gmms = cell(1, 45);

% for each phone
for j = 1:45
    filename = "data2\GMMs\" + f + "\gmm64_" + phones{j} + ".mat";
    gmms{1, j} = importdata(filename);
end

pred = zeros(1, 45 * 3);

% for each testing example
for j = 1:45 * 3
    test = mfccs{j}';

    % find phone GMM with the smallest negative log likelihood
    pred(j) = 1;
    [~, min_nll] = posterior(gmms{1, 1}, test);

    % for each brand
    for k = 2:45
        [~, nll] = posterior(gmms{1, k}, test);
        if(nll < min_nll)
            min_nll = nll;
            pred(j) = k;
        end
    end
end

acc = sum(ground_truth == pred) / length(pred);

disp("Accuracy of 64-component GMM: " + num2str(100 * acc) + "%")

figure(1)
C = confusionmat(ground_truth, pred);
confusionchart(C, phones);
title("Model identification on non speech frames with a 64-component GMM")

% return to script directory
cd("ml_classification\non-speech2")