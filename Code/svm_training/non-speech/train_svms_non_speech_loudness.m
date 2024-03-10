% -----------------> train_svms_non_speech_loudness.m <------------------- 

% The following code uses the speaker GMMs trained on the MFCCs of the non 
% speech frames of the baseline + loudness dataset to extract the training GSVs 
% and trains brand and model SVMs using bayesian optimization for hyperparameter 
% tuning. For the training set we use both the GSVs with the means only and 
% the GSVs with the means and covariances

% go to source directory
cd("..\..")

% number of components of the GMMs
num_comp = [1, 2, 4];
phones = {"HTC1", "HTC2", "LG1", "LG2", "LG3", "LG4", "N1", "N2", "N3", ...
          "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "SE1", "SE2", ...
          "V1", "iPH1"};

train_speakers = [5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23];
% which subsets of the database to use
databases_names = {"baseline", "loudness"};
% number of wav files per phone
samples = 12 * length(databases_names);
% non-speech / audio folder
f1 = "non-speech";
% which SVM folder
f2 = "loudness";

y_model = repelem(1:21, samples);
% 1: HTC, 2: LG, 3: NOKIA, 4: SAMSUNG, 5: SONY ERICSON, 6: VODAFONE, 7: APPLE
y_brand = [ones(1, 2 * samples), 2 * ones(1, 4 * samples), 3 * ones(1, 3 * samples), ...
                4 * ones(1, 8 * samples), 5 * ones(1, 2 * samples), 6 * ones(1, samples), ...
                7 * ones(1, samples)];

for i = 1:length(num_comp)
    % load train sets (only means and means and covariance)
    x_mu = zeros(21 * samples, 23 * num_comp(i));
    x_sigma = zeros(21 * samples, 2 * 23 * num_comp(i));
    x_counter = 1;

    for j = 1:length(phones)
        % folder of phone's num_comp(i)-component speaker GMMs
        folder = "data\GMMs\" + f1 + "\speakers\" + num2str(num_comp(i)) + "\" + phones{j} + "\";
    
        for k = 1:length(train_speakers)
            for m = 1:length(databases_names)
                filename = folder + "gmm_speaker_" + num2str(train_speakers(k)) + "_" + databases_names{m} + ".mat";
                gmm = importdata(filename);
        
                % reshape the means matrix from num_comp x 23 to a 23 *
                % num_comp vector containing the mean vectors stacked
                % horizontally
                x_mu(x_counter, :) = reshape(gmm.mu', 1, []);
                % also reshape the covariance matrix from 1x23xnum_comp to
                % 1x23*num_comp and horizontally stack it
                x_sigma(x_counter, :) = [reshape(gmm.mu', 1, []), reshape(gmm.Sigma, 1, [])];
    
                x_counter = x_counter + 1;
            end
        end
    end
    
    % train brand and model SVMs using bayesian optimization for
    % hyperparameter tuning

    svm_brand = fitcecoc(x_mu, y_brand, 'Learners', 'svm', 'OptimizeHyperparameters', 'all', ...
                        'HyperparameterOptimizationOptions',struct('ShowPlots', false, 'UseParallel', true));
    filename = "data\SVMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\svm_brand.mat";
    save(filename, "svm_brand")

    svm_model = fitcecoc(x_mu, y_model, 'Learners', 'svm', 'OptimizeHyperparameters', 'all', ...
                        'HyperparameterOptimizationOptions',struct('ShowPlots', false, 'UseParallel', true));
    filename = "data\SVMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\svm_model.mat";
    save(filename, "svm_model")

    svm_sigma_brand = fitcecoc(x_sigma, y_brand, 'Learners', 'svm', 'OptimizeHyperparameters', 'all', ...
                              'HyperparameterOptimizationOptions',struct('ShowPlots', false, 'UseParallel', true));
    filename = "data\SVMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\svm_sigma_brand.mat";
    save(filename, "svm_sigma_brand")

    svm_sigma_model = fitcecoc(x_sigma, y_model, 'Learners', 'svm', 'OptimizeHyperparameters', 'all', ...
                              'HyperparameterOptimizationOptions',struct('ShowPlots', false, 'UseParallel', true));
    filename = "data\SVMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\svm_sigma_model.mat";
    save(filename, "svm_sigma_model")
end

% return to script directory
cd("svm_training\non-speech")