% -----------------> train_gmms_audio_speakers2.m <------------------- 

% The following code uses the extracted training and testing audio MFCCs of 
% the ccnu_mobile dataset and uses them to train speaker dependent GMMs 
% (for the train and the test set) which will be used for GSV extraction 
% in the SVM training

% number of components of the GMMs
num_comp = [1, 64];
phones = {"AIR1", "AIR2_1", "AIR2_2", "H10", "H7X", "H8_1", "H8_2", "H8_3", "H9", "HV8", "HUAWN", "HUAWN2S", "HUAWN3E", ...
          "HUAWP10", "HUAWP20", "HUAWTAG", "NZ11", "OR9S", "S8", "SPH", "VX3F", "VX7", "VY11T", "MI2S", "MI5", "MI8", ...
          "MI8SE_1", "MI8SE_2", "MIX2", "R3S", "RNOTE3", "RNOTE4X", "ZC880A", "ZG719C", "iPAD7", "iPH6_1", "iPH6_2", ...
          "iPH6_3", "iPH6_4", "iPH6S_1", "iPH6S_2", "iPH6S_3", "iPH7P", "iPHSE", "iPHX"};

% non-speech / audio folder
f = "audio";

% go to source directory
cd("..\..")

% for each phone
for i = 1:length(phones)
    % get folder of testing audio of phone in current subset
    folder = "data2\MFCCs\" + f + "\" + phones{i} + "\";
        
    % read the MFCCs of each speaker
    for j = 1:5
        filename = folder + "mfccs_speaker" + num2str(j) + ".mat";
        mfccs = importdata(filename);
        % convert to numframesx39 matrix
        mfccs = mfccs';
        
        % train the GMMs with the different number of components
        for k = 1:length(num_comp)
            options = statset("Display", "off", "MaxIter", 500);
            gmm = fitgmdist(mfccs, num_comp(k), "CovarianceType", "diagonal", "RegularizationValue", 1e-6, "Options", options);
                    
            filename = "data2\GMMs\" + f + "\speakers\" + num2str(num_comp(k)) + "\" + phones{i} + "\gmm_speaker_" + num2str(j) ...
                        + ".mat";
            save(filename, "gmm");
        end
    end
end

% return to script directory
cd("gmm_training\audio2")