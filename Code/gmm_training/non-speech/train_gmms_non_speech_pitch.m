% -----------------> train_gmms_non_speech_pitch.m <------------------- 

% The following code uses the extracted training and testing non speech MFCCs 
% of the baseline + pitch dataset and uses them to train GMMs with the number 
% of components being 1, 2, 4, 8, 16, 32 and 64. Brand and model dependent GMMs are
% trained to be used in the Nearest Neighbor classification

% number of components of the GMMs
num_comp = [1, 2, 4, 8, 16, 32, 64];
phones = {"HTC1", "HTC2", "LG1", "LG2", "LG3", "LG4", "N1", "N2", "N3", ...
          "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "SE1", "SE2", ...
          "V1", "iPH1"};

% which subsets of the database to use
databases_names = {"baseline", "pitch"};
% non-speech / audio folder
f1 = "non-speech";
% which GMM dataset folder
f2 = "pitch";

% go to source directory
cd("..\..")

%%%% 1. Construct model dependent training GMMs using the MFCCs of all the speakers of each phone %%%%

% for each phone
for i = 1:length(phones)
    % allocate space for 12 * number of databases 23xnumframes MFCC arrays
    data = cell(12 * length(databases_names), 1);

    % for each subset of the database
    for m = 1:length(databases_names)
        % get folder of training MFCCs of phone in the current subset
        folder = "data\MFCCs\" + f1 + "\" + databases_names{m} + "\train\" + phones{i} + "\";
        d = dir(folder);
    
        % read the MFCCs of each speaker
        for j = 1:12
            filename = folder + d(j + 2).name;
            data{(m - 1) * 12 + j} = importdata(filename);
        end
    end

    % concatenate all MFCCs in a numframesx23 matrix
    mfccs = [data{:}]';

    % train the GMMs with the different number of components that will be 
    % used as the training set
    for j = 1:length(num_comp)
        options = statset("Display", "off", "MaxIter", 500);
        gmm = fitgmdist(mfccs, num_comp(j), "CovarianceType", "diagonal", "RegularizationValue", 1e-6, "Options", options);
        
        filename = "data\GMMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(j)) + "\gmm_" + phones{i} + ".mat";
        save(filename, "gmm");
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%% 2. Construct brand dependent training GMMs using the MFCCs of all the speakers of each brand %%%%

% brand and number of phones of each brand
brands = {"HTC", "LG", "NOKIA", "SAMSUNG", "SONY_ERICSSON"};
num_phones = [2, 4, 3, 8, 2];
% variable to help us start from the first phone of each brand
offset = 0;
for n = 1:length(brands)
    % allocate space for the MFCCs of the 12 * number of databases speakers 
    % of each phone of the brand
    data = cell(12 * num_phones(n) * length(databases_names), 1);
    data_counter = 1;
    for i = offset + 1:offset + num_phones(n)
        % for each subset of the database
        for m = 1:length(databases_names)
            % folder of training MFCCs of phone in the current subset
            folder = "data\MFCCs\" + f1 + "\" + databases_names{m} + "\train\" + phones{i};
            d = dir(folder);
            for j = 1:12
                filename = folder + "\" + d(j + 2).name;
                data{data_counter} = importdata(filename);
                data_counter = data_counter + 1;
            end
        end
    end
    % concatenate all MFCCs in a numframesx23 matrix
    data = [data{:}]';
    
    for j = 1:length(num_comp)
        options = statset("Display", "off", "MaxIter", 500);
        gmm = fitgmdist(data, num_comp(j), "CovarianceType", "diagonal", "RegularizationValue", 1e-6, "Options", options);
    
        filename = "data\GMMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(j)) + "\gmm_" + brands(n) + ".mat";
        save(filename, "gmm")
    end
    offset = offset + num_phones(n);
end

% Apple and Vodafone only have one phone so the phone GMM is also the
% brand GMM
for i = 1:length(num_comp)
    gmm = importdata("data\GMMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\gmm_V1.mat");
    save("data\GMMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\gmm_VODAFONE.mat", "gmm")

    gmm = importdata("data\GMMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\gmm_iPH1.mat");
    save("data\GMMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\gmm_APPLE.mat", "gmm")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to script directory
cd("gmm_training\non-speech")