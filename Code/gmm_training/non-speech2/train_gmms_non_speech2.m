% -----------------> train_gmms_non_speech2.m <------------------- 

% The following code uses the extracted training and testing non speech MFCCs 
% of the ccnu_mobile dataset and uses them to train GMMs with 64 components. 
% Brand and model dependent GMMs are trained to be used in the Nearest Neighbor 
% classification

% number of components of the GMMs
num_comp = 64;
% the order of the phones must match the order of the brands further below
phones = {"AIR1", "AIR2_1", "AIR2_2", "iPAD7", "iPH6_1", "iPH6_2", "iPH6_3", "iPH6_4", "iPH6S_1", "iPH6S_2", "iPH6S_3", ...
          "iPH7P", "iPHSE", "iPHX", "H10", "H7X", "H8_1", "H8_2", "H8_3", "H9", "HV8", "HUAWN", "HUAWN2S", "HUAWN3E", ...
          "HUAWP10", "HUAWP20", "HUAWTAG", "NZ11", "OR9S", "S8", "SPH", "VX3F", "VX7", "VY11T", "MI2S", "MI5", "MI8", ...
          "MI8SE_1", "MI8SE_2", "MIX2", "R3S", "RNOTE3", "RNOTE4X", "ZC880A", "ZG719C"};

% non-speech / audio folder
f = "non-speech";

num_train_speakers = 2;

% go to source directory
cd("..\..")

%%%% 1. Construct model dependent training GMMs using the MFCCs of all the speakers of each phone %%%%

% for each phone
for i = 1:length(phones)
    % allocate space for num_train_speakers 39xnumframes MFCC arrays
    data = cell(num_train_speakers, 1);

    % get folder of training MFCCs of phone
    folder = "data2\MFCCs\" + f + "\" + phones{i} + "\";
    d = dir(folder);
    
    % read the MFCCs of each speaker
    for j = 1:num_train_speakers
        filename = folder + d(j + 2).name;
        data{j} = importdata(filename);
    end

    % concatenate all MFCCs in a numframesx23 matrix
    mfccs = [data{:}]';

    % train the GMM
    options = statset("Display", "off", "MaxIter", 500);
    gmm = fitgmdist(mfccs, num_comp, "CovarianceType", "diagonal", "RegularizationValue", 1e-6, "Options", options);
        
    filename = "data2\GMMs\" + f + "\gmm64_" + phones{i} + ".mat";
    save(filename, "gmm");
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%% 2. Construct brand dependent training GMMs using the MFCCs of all the speakers of each brand %%%%

% brand and number of phones of each brand
brands = {"APPLE", "HONOR", "HUAWEI", "NUBIA", "OPPO", "SAMSUNG", "VIVO", "XIAOMI", "ZTE"};
num_phones = [14, 7, 6, 1, 1, 2, 3, 9, 2];
% variable to help us start from the first phone of each brand
offset = 0;
for n = 1:length(brands)
    % allocate space for the MFCCs of the 2 * number of databases speakers 
    % of each phone of the brand
    data = cell(num_train_speakers, 1);
    data_counter = 1;
    for i = offset + 1:offset + num_phones(n)
        % folder of training MFCCs of phone in the current subset
        folder = "data2\MFCCs\" + f + "\" + phones{i};
        d = dir(folder);
        for j = 1:num_train_speakers
            filename = folder + "\" + d(j + 2).name;
            data{data_counter} = importdata(filename);
            data_counter = data_counter + 1;
        end
    end

    % concatenate all MFCCs in a numframesx23 matrix
    data = [data{:}]';
    
    options = statset("Display", "off", "MaxIter", 500);
    gmm = fitgmdist(data, num_comp, "CovarianceType", "diagonal", "RegularizationValue", 1e-6, "Options", options);
    
    filename = "data2\GMMs\" + f + "\gmm64_" + brands(n) + ".mat";
    save(filename, "gmm")

    offset = offset + num_phones(n);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return to script directory
cd("gmm_training\non-speech2")