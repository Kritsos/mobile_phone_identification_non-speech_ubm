% -----------------> extract_mfccs_non_speech2.m <------------------- 

% The following code extracts the non speech MFCCs of the train and test 
% speakers of the ccnu_mobile dataset. For each phone and for each speaker 
% (for the train and the test set) it saves a 39xnumframes  matrix containing 
% the corresponding MFCCs and their deltas and delta-deltas (13 of each)

% the dir() function sorts the files and folders alphabetically with
% capital letters being first so this is the order in which the phones are
% read: Apple Air1, Apple Air2 (1), Apple Air2 (2), Honor 10, Honor 7x, 
% Honor 8 (1), Honor 8 (2), Honor 8 (3), Honor 9, Honor V8, Huawei Nova, 
% Huawei Nova 2S, Huawei Nova 3E, Huawei P10, Huawei P20, Huawei TAG-AL00, 
% Nubia Z11, Oppo R9s, Samsung S8, Samsung SPH-D710, Vivo X3F, Vivo X7, 
% Vivo Y11t, Xiaomi Mi 2S, Xiaomi Mi 5, Xiaomi Mi 8, Xiaomi Mi 8 SE (1), 
% Xiaomi Mi 8 SE (2), Xiaomi Mi Mix 2, Xiaomi Redmi 3S, Xiaomi Redmi Note 3, 
% Xiaomi Redmi Note 4X, ZTE C880A, ZTE G719C, iPad 7, iPhone 6 (1),
% iPhone 6 (2), iPhone 6 (3), iPhone 6 (4), iPhone 6S (1), iPhone 6S (2),
% iPhone 6S (3), iPhone 7 Plus, iPhone SE, iPhone X and so the below IDs
% correspond to these phones in the above order
phones = {"AIR1", "AIR2_1", "AIR2_2", "H10", "H7X", "H8_1", "H8_2", "H8_3", "H9", "HV8", "HUAWN", "HUAWN2S", "HUAWN3E", ...
          "HUAWP10", "HUAWP20", "HUAWTAG", "NZ11", "OR9S", "S8", "SPH", "VX3F", "VX7", "VY11T", "MI2S", "MI5", "MI8", ...
          "MI8SE_1", "MI8SE_2", "MIX2", "R3S", "RNOTE3", "RNOTE4X", "ZC880A", "ZG719C", "iPAD7", "iPH6_1", "iPH6_2", ...
          "iPH6_3", "iPH6_4", "iPH6S_1", "iPH6S_2", "iPH6S_3", "iPH7P", "iPHSE", "iPHX"};

% go to source directory
cd("..")

d1 = dir("CCNU_MOBILE");
    
for i = 3:numel(d1)
    % path of phone's audio files
    path = "CCNU_MOBILE\" + d1(i).name + "\";
    cd(path)

    d2 = dir(pwd);
            
    % for each speaker
    for j = 1:5
        % read wav file
        [x, fs] = audioread(d2(j + 2).name);      

        % perform speech enhancement via spectral subtraction
        sp = v_specsub(x, fs);
    
        % remove the estimated speech signal from the audio
        x = x - sp;
                 
        % extract MFCCs on 16ms frames with 8ms hop
        y = v_melcepst(x, fs, '0dD');
        y = y';
    
        % cepstral mean and variance normalization
        % y = cmvn(y', true);
            
        % go back to initial directory to save the MFCCs
        cd("..\..\") 

        filename = "data2\MFCCs\non-speech\" + phones{i - 2} + "\mfccs_speaker" + num2str(j) + ".mat";
        save(filename, "y");
        % go into the folder with the phone's audio files again
        cd(path)
    end
    % go back to initial directory
    cd("..\..\")
end

% return to script directory
cd("mfcc_extraction")