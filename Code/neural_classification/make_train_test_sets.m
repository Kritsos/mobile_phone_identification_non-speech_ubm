% go to source directory
cd("..")

% training wav files per phone
train_samples = 2;
% testing wav files per phone
test_samples = 3;
% number of training examples
n_train = train_samples * 45;
% number of testing examples
n_test = test_samples * 45;
% number of MFCC vectors to keep
num_mfcc = 650;
% dimensionality of MFCC vectors
dim_mfcc = 39;
% number of components of GSVs
num_comp = 64;

% the order of the phones must match the order of the brands in y_brand
phones = {"AIR1", "AIR2_1", "AIR2_2", "iPAD7", "iPH6_1", "iPH6_2", "iPH6_3", "iPH6_4", "iPH6S_1", "iPH6S_2", "iPH6S_3", ...
          "iPH7P", "iPHSE", "iPHX", "H10", "H7X", "H8_1", "H8_2", "H8_3", "H9", "HV8", "HUAWN", "HUAWN2S", "HUAWN3E", ...
          "HUAWP10", "HUAWP20", "HUAWTAG", "NZ11", "OR9S", "S8", "SPH", "VX3F", "VX7", "VY11T", "MI2S", "MI5", "MI8", ...
          "MI8SE_1", "MI8SE_2", "MIX2", "R3S", "RNOTE3", "RNOTE4X", "ZC880A", "ZG719C"};


%%%%%%%%%%%%%%%%%%%%%%%%%% audio frames %%%%%%%%%%%%%%%%%%%%%%%%%%
x_train_mfcc = zeros(n_train, num_mfcc, dim_mfcc);
x_test_mfcc = zeros(n_test, num_mfcc, dim_mfcc);
x_train_gsv = zeros(n_train, dim_mfcc, num_comp);
x_test_gsv = zeros(n_test, dim_mfcc, num_comp);

x_train_counter1 = 1;
x_test_counter1 = 1;
x_train_counter2 = 1;
x_test_counter2 = 1;

for i = 1:length(phones)
    % load MFCCs
    % get folder of training MFCCs of phone
    folder = "data2\MFCCs\audio\" + phones{i} + "\";
    d = dir(folder);

    % read the MFCCs of each train speaker
    for j = 1:5
        filename = folder + d(j + 2).name;
        mfccs = importdata(filename);
        mfccs = mfccs';
        if(j < 3)
            x_train_mfcc(x_train_counter1, :, :) = mfccs(1:num_mfcc, :);
            x_train_counter1 = x_train_counter1 + 1;
        else
            x_test_mfcc(x_test_counter1, :, :) = mfccs(1:num_mfcc, :);
            x_test_counter1 = x_test_counter1 + 1;
        end
    end

    % load GSVs
    % get folder of train speaker GMMs of phone
    folder = "data2\GMMs\audio\speakers\64\" + phones{i} + "\";
    d = dir(folder);

    % read the GMM of each train speaker
    for j = 1:5
        filename = folder + d(j + 2).name;
        gmm = importdata(filename);
        if(j < 3)
            x_train_gsv(x_train_counter2, :, :) = gmm.mu';
            x_train_counter2 = x_train_counter2 + 1;
        else
            x_test_gsv(x_test_counter2, :, :) = gmm.mu';
            x_test_counter2 = x_test_counter2 + 1;
        end
    end
end

filename = "neural_classification\x_train_mfcc_audio.mat";
save(filename, "x_train_mfcc")

filename = "neural_classification\x_test_mfcc_audio.mat";
save(filename, "x_test_mfcc")

filename = "neural_classification\x_train_gsv_audio.mat";
save(filename, "x_train_gsv")

filename = "neural_classification\x_test_gsv_audio.mat";
save(filename, "x_test_gsv")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% non speech frames %%%%%%%%%%%%%%%%%%%%%%%%%%
x_train_mfcc = zeros(n_train, num_mfcc, dim_mfcc);
x_test_mfcc = zeros(n_test, num_mfcc, dim_mfcc);
x_train_gsv = zeros(n_train, dim_mfcc, num_comp);
x_test_gsv = zeros(n_test, dim_mfcc, num_comp);

x_train_counter1 = 1;
x_test_counter1 = 1;
x_train_counter2 = 1;
x_test_counter2 = 1;

for i = 1:length(phones)
    % load MFCCs
    % get folder of training MFCCs of phone
    folder = "data2\MFCCs\non-speech\" + phones{i} + "\";
    d = dir(folder);

    % read the MFCCs of each train speaker
    for j = 1:5
        filename = folder + d(j + 2).name;
        mfccs = importdata(filename);
        mfccs = mfccs';
        if(j < 3)
            x_train_mfcc(x_train_counter1, :, :) = mfccs(1:num_mfcc, :);
            x_train_counter1 = x_train_counter1 + 1;
        else
            x_test_mfcc(x_test_counter1, :, :) = mfccs(1:num_mfcc, :);
            x_test_counter1 = x_test_counter1 + 1;
        end
    end

    % load GSVs
    % get folder of train speaker GMMs of phone
    folder = "data2\GMMs\non-speech\speakers\64\" + phones{i} + "\";
    d = dir(folder);

    % read the GMM of each train speaker
    for j = 1:5
        filename = folder + d(j + 2).name;
        gmm = importdata(filename);
        if(j < 3)
            x_train_gsv(x_train_counter2, :, :) = gmm.mu';
            x_train_counter2 = x_train_counter2 + 1;
        else
            x_test_gsv(x_test_counter2, :, :) = gmm.mu';
            x_test_counter2 = x_test_counter2 + 1;
        end
    end
end

filename = "neural_classification\x_train_mfcc_non_speech.mat";
save(filename, "x_train_mfcc")

filename = "neural_classification\x_test_mfcc_non_speech.mat";
save(filename, "x_test_mfcc")

filename = "neural_classification\x_train_gsv_non_speech.mat";
save(filename, "x_train_gsv")

filename = "neural_classification\x_test_gsv_non_speech.mat";
save(filename, "x_test_gsv")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% ubm adapted %%%%%%%%%%%%%%%%%%%%%%%%%%
x_train_gsv = zeros(n_train, dim_mfcc, num_comp);
x_test_gsv = zeros(n_test, dim_mfcc, num_comp);

x_train_counter1 = 1;
x_test_counter1 = 1;

filename = "data2\UBM-TIMIT\UBMs\ubm_64.mat";
ubm = importdata(filename);

for i = 1:length(phones)
    % load MFCCs
    % get folder of training MFCCs of phone
    folder = "data2\MFCCs\audio\" + phones{i} + "\";
    d = dir(folder);

    % read the MFCCs of each train speaker
    for j = 1:5
        filename = folder + d(j + 2).name;
        mfccs = importdata(filename);
        mu = map_adapt_means(ubm, mfccs', 16);
        if(j < 3)
            x_train_gsv(x_train_counter1, :, :) = mu';
            x_train_counter1 = x_train_counter1 + 1;
        else
            x_test_gsv(x_test_counter1, :, :) = mu';
            x_test_counter1 = x_test_counter1 + 1;
        end
    end
end

filename = "neural_classification\x_train_gsv_ubm_adapted.mat";
save(filename, "x_train_gsv")

filename = "neural_classification\x_test_gsv_ubm_adapted.mat";
save(filename, "x_test_gsv")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


y_train_model = repelem(1:45, train_samples);
% 1: APPLE, 2: HONOR, 3: HUAWEI, 4: NUBIA, 5: OPPO, 6: SAMSUNG, 7: VIVO, 8: XIAOMI, 9: ZTE
y_train_brand = [ones(1, 14 * train_samples), 2 * ones(1, 7 * train_samples), 3 * ones(1, 6 * train_samples), ...
                4 * ones(1, train_samples), 5 * ones(1, train_samples), 6 * ones(1, 2 * train_samples), ...
                7 * ones(1, 3 * train_samples), 8 * ones(1, 9 * train_samples), 9 * ones(1, 2 * train_samples)];
% one hot encode y
y_train_model = (y_train_model' == 1:45);
y_train_brand = (y_train_brand' == 1:9);

y_test_model = repelem(1:45, test_samples);
% 1: APPLE, 2: HONOR, 3: HUAWEI, 4: NUBIA, 5: OPPO, 6: SAMSUNG, 7: VIVO, 8: XIAOMI, 9: ZTE
y_test_brand = [ones(1, 14 * test_samples), 2 * ones(1, 7 * test_samples), 3 * ones(1, 6 * test_samples), ...
                4 * ones(1, test_samples), 5 * ones(1, test_samples), 6 * ones(1, 2 * test_samples), ...
                7 * ones(1, 3 * test_samples), 8 * ones(1, 9 * test_samples), 9 * ones(1, 2 * test_samples)];
% one hot encode y
y_test_model = (y_test_model' == 1:45);
y_test_brand = (y_test_brand' == 1:9);

filename = "neural_classification\y_train_model.mat";
save(filename, "y_train_model")

filename = "neural_classification\y_test_model.mat";
save(filename, "y_test_model")

filename = "neural_classification\y_train_brand.mat";
save(filename, "y_train_brand")

filename = "neural_classification\y_test_brand.mat";
save(filename, "y_test_brand")

% return to script directory
cd("neural_classification\")