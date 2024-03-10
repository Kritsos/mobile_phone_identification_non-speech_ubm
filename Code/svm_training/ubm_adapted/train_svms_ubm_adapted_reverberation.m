% -----------------> train_svms_ubm_adapted_reverberation.m <------------------- 

% The following code uses the UBM adapted speaker GMMs trained on the MFCCs 
% of all the frames of the baseline + reverberation dataset to extract the training 
% GSVs and trains brand and model SVMs using bayesian optimization for hyperparameter 
% tuning. The means of the GMMs are only adapted so we only use mean GSVs

% go to source directory
cd("..\..")

% number of components of the GSVs
num_comp = [1, 2, 4, 8, 16];
phones = {"HTC1", "HTC2", "LG1", "LG2", "LG3", "LG4", "N1", "N2", "N3", ...
          "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "SE1", "SE2", ...
          "V1", "iPH1"};

train_speakers = [5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23];
% which subsets of the database to use
databases_names = {"baseline", "reverberation"};
% number of wav files per phone
samples = 12 * length(databases_names);
% which SVM folder
f = "reverberation";

y_model = repelem(1:21, samples);
% 1: HTC, 2: LG, 3: NOKIA, 4: SAMSUNG, 5: SONY ERICSON, 6: VODAFONE, 7: APPLE
y_brand = [ones(1, 2 * samples), 2 * ones(1, 4 * samples), 3 * ones(1, 3 * samples), ...
                4 * ones(1, 8 * samples), 5 * ones(1, 2 * samples), 6 * ones(1, samples), ...
                7 * ones(1, samples)];


for i = 1:length(num_comp)
    % load train sets
    x_mu = zeros(21 * samples, 23 * num_comp(i));
    x_counter = 1;

    % load UBM
    filename = "data\UBM-TIMIT\UBMs\ubm_" + num2str(num_comp(i)) + ".mat";
    ubm = importdata(filename);

    for j = 1:length(phones)
        for k = 1:length(train_speakers)
            for m = 1:length(databases_names)
                % folder of phone's MFCCs
                filename = "data\MFCCs\audio\" + databases_names{m} + "\train\" + phones{j} + "\" + "mfccs_speaker" + num2str(train_speakers(k))...
                            + ".mat";
                mfccs = importdata(filename);

                % MAP adapt means of UBM to speaker's MFCCS
                mu = map_adapt_means(ubm, mfccs', 16);
        
                % reshape the means matrix from num_comp x 23 to a 23 *
                % num_comp vector containing the mean vectors stacked
                % horizontally
                x_mu(x_counter, :) = reshape(mu', 1, []);
    
                x_counter = x_counter + 1;
            end
        end
    end
    
    % train brand and model SVMs using bayesian optimization for
    % hyperparameter tuning

    svm_brand = fitcecoc(x_mu, y_brand, 'Learners', 'svm', 'OptimizeHyperparameters', 'all', ...
                        'HyperparameterOptimizationOptions',struct('ShowPlots', false, 'UseParallel', true));
    filename = "data\SVMs\ubm_adapted\" + f + "\" + num2str(num_comp(i)) + "\svm_brand.mat";
    save(filename, "svm_brand")

    svm_model = fitcecoc(x_mu, y_model, 'Learners', 'svm', 'OptimizeHyperparameters', 'all', ...
                        'HyperparameterOptimizationOptions',struct('ShowPlots', false, 'UseParallel', true));
    filename = "data\SVMs\ubm_adapted\" + f + "\" + num2str(num_comp(i)) + "\svm_model.mat";
    save(filename, "svm_model")
end

% return to script directory
cd("svm_training\ubm_adapted")