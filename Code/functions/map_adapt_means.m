% This function MAP adapts the means of the given UBM-GMM to the data it is
% given
% Input:  ubm: gmdistribution object of the UBM-GMM
%         x:   data (num_examples x num_features) to which the UBM will be adapted to
%         r:   relevance factor for the adaptation coefficient
% Output: A num_components x num_features matrix containing the adapted
%         mean vectors
function new_mu = map_adapt_means(ubm, x, r)
    % probability of each vector x_t coming from each component of the ubm
    pr = posterior(ubm, x); % num_x x num_components

    % if zero probabilities are found add a small value to ensure that NaNs
    % dont appear later
    if(any(pr(:) == 0))
        pr = pr + eps;
    end

    % sum of probabilities of each component (count statistic)
    n = sum(pr, 1); % 1 x num_components

    % initialize matrix of size num_components x num_dimensions_x
    % (first moments statistic) and new means matrix
    e = zeros(size(ubm.mu, 1), size(ubm.mu, 2));
    new_mu = zeros(size(ubm.mu, 1), size(ubm.mu, 2));

    % for each component
    for i = 1:size(ubm.mu, 1)
        % first moment
        e(i, :) = (1/n(i)) * sum(pr(:, i) .* x, 1);
        
        % adaptation coefficient
        a = n(i) / (n(i) + r);
        
        % adapted mean vector for component i
        new_mu(i, :) = a * e(i, :) + (1 - a) * ubm.mu(i, :);
    end
end