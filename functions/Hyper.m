function H = Hyper(data, kNN)
    m = size(data, 1);    % Number of data points
    H = zeros(m, m);      % Initialize the incidence matrix
    
    % For each data point, find its kNN neighbors using Hamming distance
    for j = 1:m
        % Find kNN neighbors of data point j
        neighbors = knnsearch(data, data(j,:), 'dist', 'hamming', 'K', kNN);
        
        % Update the incidence matrix H
        for ii = 1:kNN
            H(neighbors(ii), j) = 1;
        end
    end
end
