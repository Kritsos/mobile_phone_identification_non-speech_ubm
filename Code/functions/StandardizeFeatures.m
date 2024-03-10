function  standardizedFeatureMatrix  = StandardizeFeatures(featureMatrix)

% Initialize
%startup;


    [r c] = size(featureMatrix);
    % Standardize values
    maxFeatures = max(featureMatrix,[],2);
    minFeatures = min(featureMatrix,[],2);

    for i=1:r
        for j=1:c
            D = maxFeatures(i)-minFeatures(i);
            if D == 0; D = 0.000001; end
            standardizedFeatureMatrix(i,j) = (featureMatrix(i,j) - minFeatures(i))/(D);
            if (isnan(standardizedFeatureMatrix(i,j))) standardizedFeatureMatrix(i,j) = 0.0; end;
        end
    end





