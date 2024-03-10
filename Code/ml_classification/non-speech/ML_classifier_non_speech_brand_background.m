% -----------------> ML_classifier_non_speech_brand_background.m <------------------- 

% The following code uses the brand GMMs trained on the non speech MFCCs of
% the baseline + background dataset and Maximum Likelihood classification to classify 
% each recording into one of  the 7 brands. The classification is done using
% the negative log likelihood, that is each recording is classified into the 
% brand whose GMM gives the smallest negative log likelihood value on the 
% recording's MFCCs

% go to source directory
cd("..\..")

num_comp = [1, 2, 4, 8, 16, 32, 64];
phones = {"HTC1", "HTC2", "LG1", "LG2", "LG3", "LG4", "N1", "N2", "N3", ...
          "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "SE1", "SE2", ...
          "V1", "iPH1"};
brands = {'HTC', 'LG', 'NOKIA', 'SAMSUNG', 'SONY_ERICSSON', 'VODAFONE', 'APPLE'};

% which subsets of the database to use
databases_names = {"baseline", "background"};
% number of wav files per phone
samples = 12 * length(databases_names);
% non-speech / audio folder
f1 = "non-speech";
% which GMM dataset folder
f2 = "background";

% 1: HTC, 2: LG, 3: NOKIA, 4: SAMSUNG, 5: SONY ERICSON, 6: VODAFONE, 7: APPLE
ground_truth = [ones(1, 2 * samples), 2 * ones(1, 4 * samples), 3 * ones(1, 3 * samples), ...
                4 * ones(1, 8 * samples), 5 * ones(1, 2 * samples), 6 * ones(1, samples), ...
                7 * ones(1, samples)];

% load test set
% allocate space for the MFCCs matrices
mfccs = cell(21 * 12 * length(databases_names), 1);
mfccs_counter = 1;
% label for each testing example that indicates from which subset the
% example comes (1: baseline, 2: background)
subset = zeros(1, 21 * 12 * length(databases_names));
% for each phone
for i = 1:21
    % for each subset of the augmented database
    for m = 1:length(databases_names)
        folder = "data\MFCCs\" + f1 + "\" + databases_names{m} + "\test\" + phones{i} + "\";
        d = dir(folder);
    
        % for each speaker
        for j = 1:12
            filename = folder + d(j + 2).name;
            mfccs{mfccs_counter} = importdata(filename);
            subset(mfccs_counter) = m;
            mfccs_counter = mfccs_counter + 1;
        end
    end
end

% allocate space for the brand GMMs for each number of components
gmms = cell(length(num_comp), 7);

% for each number of components
for i = 1:length(num_comp)
    % for each brand
    for j = 1:7
        filename = "data\GMMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\gmm_" + brands{j} + ".mat";
        gmms{i, j} = importdata(filename);
    end
end


for i = 1:length(num_comp)
    pred = zeros(1, 21 * 12 * length(databases_names));
    
    % for each testing example
    for j = 1:21 * 12 * length(databases_names)
        test = mfccs{j}';

        % find brand GMM with the smallest negative log likelihood
        pred(j) = 1;
        [~, min_nll] = posterior(gmms{i, 1}, test);

        % for each brand
        for k = 2:7
            [~, nll] = posterior(gmms{i, k}, test);
            if(nll < min_nll)
                min_nll = nll;
                pred(j) = k;
            end
        end
    end

    acc = sum(ground_truth == pred) / length(pred);
    acc_baseline = sum(ground_truth(subset == 1) == pred(subset == 1)) / 252;
    acc_background = sum(ground_truth(subset == 2) == pred(subset == 2)) / 252;

    disp("Accuracy of " + num2str(num_comp(i)) + "-component GMM: " + num2str(100 * acc) + "%: " + ...
         "Baseline: " + num2str(100 * acc_baseline) + "% / " + "Background: " + num2str(100 * acc_background) + "%")

    figure(i)
    C = confusionmat(ground_truth, pred);
    confusionchart(C, brands);
    title(num2str("Brand identification on non speech frames with a " + num_comp(i)) + "-component GMM")
end

% return to script directory
cd("ml_classification\non-speech")