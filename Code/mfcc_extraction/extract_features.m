function [ y ] = extract_features(x, fs)
% extracts features from an audio signal (only one channel e.g. left one)

    % Left channel
    input = x(:, 1);
    y = melfcc(input, fs, 'maxfreq', fs/2, 'numcep', 23,...
                'nbands', 42, 'fbtype', 'fcmel', 'dcttype', 1, 'usecmp', 1,...
                'wintime', 0.02, 'hoptime', 0.01, 'preemph', 0, 'dither', 1);
end

