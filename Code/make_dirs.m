% -----------------> make_dirs.m <------------------- 

% The following code creates all the necessary directories in which the
% various extracted features and trained models will be stored

phones = {"HTC1", "HTC2", "LG1", "LG2", "LG3", "LG4", "N1", "N2", "N3", ...
          "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "SE1", "SE2", ...
          "V1", "iPH1"};

f1 = {"audio", "non-speech"};
f2 = {"baseline", "gaussian", "background", "reverberation", "crop", "loudness", "pitch", "speed", "vtlp"};
f3 = {"train", "test"};

mkdir("data")

% folders for MFCCs
mkdir("data\MFCCs")

for i = 1:length(f1)
    for j = 1:length(f2)
        for k = 1:length(f3)
            for l = 1:length(phones)
                mkdir("data\MFCCs\" + f1{i} + "\" + f2{j} + "\" + f3{k} + "\" + phones{l})
            end
        end
    end
end

% folders for GMMs
mkdir("data\GMMs")
num_comp = [1, 2, 4, 8, 16, 32, 64];
f4 = {"baseline", "augmented", "gaussian", "background", "reverberation", "crop", "loudness", "pitch", "speed", "vtlp", ...
      "augmented2", "all"};

for i = 1:length(f1)
    mkdir("data\GMMs\" + f1{i})
    
    for j = 1:length(f4)
        mkdir("data\GMMs\" + f1{i} + "\" + f4{j})

        for k = 1:length(num_comp)
            mkdir("data\GMMs\" + f1{i} + "\" + f4{j} + "\" + num_comp(k))
        end
    end
end

for i = 1:length(f1)
    mkdir("data\GMMs\" + f1{i} + "\speakers")

    for j = 1:length(num_comp)
        mkdir("data\GMMs\" + f1{i} + "\speakers\" + "\" + num2str(num_comp(j)))

        for k = 1:length(phones)
            mkdir("data\GMMs\" + f1{i} + "\speakers\" + "\" + num2str(num_comp(j)) + "\" + phones{k})
        end
    end
end

% folders for SVMs
num_comp_svm = [1, 2, 4];

mkdir("data\SVMs")
for i = 1:length(f1)
    mkdir("data\SVMs\" + f1{i})
    
    for j = 1:length(f4)
        mkdir("data\SVMs\" + f1{i} + "\" + f4{j})

        for k = 1:length(num_comp_svm)
            mkdir("data\SVMs\" + f1{i} + "\" + f4{j} + "\" + num2str(num_comp_svm(k)))
        end
    end
end

% folder for UBM and TIMIT related data
mkdir("data\UBM-TIMIT")
mkdir("data\UBM-TIMIT\MFCCs")
mkdir("data\UBM-TIMIT\UBMs")

mkdir("data\SVMs\ubm_adapted")
num_comp_svm2 = [1, 2, 4, 8, 16];

    
for i = 1:length(f4)
    mkdir("data\SVMs\ubm_adapted\" + f4{i})

    for j = 1:length(num_comp_svm2)
        mkdir("data\SVMs\ubm_adapted\" + f4{i} + "\" + num2str(num_comp_svm2(j)))
    end
end

% data2 folder for data of the ccnu_mobile dataset
mkdir("data2")

% MFCCs
mkdir("data2\MFCCs")

phones2 = {"AIR1", "AIR2_1", "AIR2_2", "H10", "H7X", "H8_1", "H8_2", "H8_3", "H9", "HV8", "HUAWN", "HUAWN2S", "HUAWN3E", ...
           "HUAWP10", "HUAWP20", "HUAWTAG", "NZ11", "OR9S", "S8", "SPH", "VX3F", "VX7", "VY11T", "MI2S", "MI5", "MI8", ...
           "MI8SE_1", "MI8SE_2", "MIX2", "R3S", "RNOTE3", "RNOTE4X", "ZC880A", "ZG719C", "iPAD7", "iPH6_1", "iPH6_2", ...
           "iPH6_3", "iPH6_4", "iPH6S_1", "iPH6S_2", "iPH6S_3", "iPH7P", "iPHSE", "iPHX"};

for i = 1:length(f1)
    for k = 1:length(phones2)
        mkdir("data2\MFCCs\" + f1{i} + "\" + phones2{k})
    end
end

% GMMs
num_comp2 = [1, 4, 64];
mkdir("data2\GMMs")

for i = 1:length(f1)
    mkdir("data2\GMMs\" + f1{i})
    mkdir("data2\GMMs\" + f1{i} + "\speakers")
    for j = 1:length(num_comp2)
        for k = 1:length(phones2)
            mkdir("data2\GMMs\" + f1{i} + "\speakers\" + num2str(num_comp2(j)) + "\" + phones2{k})
        end
    end
end

% SVMs
mkdir("data2\SVMs")

for i = 1:length(f1)
    mkdir("data2\SVMs\" + f1{i})
end

% folder for UBM and TIMIT related data
mkdir("data2\UBM-TIMIT")
mkdir("data2\UBM-TIMIT\MFCCs")
mkdir("data2\UBM-TIMIT\UBMs")

mkdir("data2\SVMs\ubm_adapted")