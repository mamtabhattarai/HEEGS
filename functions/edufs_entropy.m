function [ret] = edufs_entropy(data, class_size, para, entropy_type)
    %lambda3 is mu lambda1 is alpha lambda3 is beta lambda4 is lambda
    lambda1 = para.lambda1;
    lambda2 = para.lambda2;
    %lambda3 = para.lambda3;
    lambda4 = para.lambda4;
    % n: sample, d: feature
    [n, d] = size(data);
    mu = 0.01;
    X = data;
    ITER1 = 100;

    %alpha = 2;%input('Enter the alpha value: ');
    [d, n] = size(X);
    X = X - mean(X, 2) * ones(1, n);
    m = class_size;

    % mu = 10^-3;
    % in paper, mu was set in the range of 10^-6 to 10^-3
    %mu = lambda3;
    rho = 1.1;
    mu_max = 10^10;
    [n, d] = size(data);
    U = rand(n, m);
    V = rand(d, m);
    q = 0;
    Y1 = 0;
    Y2 = 0;
    E = zeros(n, d);
    K = 0;
    ITER = 10;
    options = [];
    KNN = 5;
    SI = Hyper(X, KNN);
    Ls = HypLap(SI, 'Saito');
    %Ls=HypLap(SI,'Zhou');
    %Ls=HypLap(SI,'Rod');
    %options.Metric = 'Euclidean';
    %options.NeighborMode = 'KNN';
    %options.k = 5;
    %options.WeightMode = 'Binary';
    %options.WeightMode = 'HeatKernel';
    %W = constructW(X,options);
    %D = diag(sum(W,1));
    %Q=D-W;
    Q = Ls;

    % Choose entropy measure
    switch entropy_type
        case 'shannon'
            QQ = calculate_shannon_entropy(data);
        case 'renyi'
            alpha = 2; % You can change the order of Rényi entropy here
            QQ = calculate_renyi_entropy(data, alpha);
        case 'tsallis'
            q = 2; % You can change the q value of Tsallis entropy here
            QQ = calculate_tsallis_entropy(data, q);
        case 'collision'
            QQ = calculate_collision_entropy(data);
        case'L1'
            QQ=L1_entropy(data);
        case'L2'
            QQ=L2_entropy(data);
        otherwise
            error('Unknown entropy type');
    end

    
    
    for i = 1:ITER
        q = X - U * V' + (1 / mu) * Y2;

        % Update E
        for j = 1:n
            qi_norm = sqrt(sum(q(j, :).^2));
            if qi_norm > (1 / mu)
                E(j, :) = (1 - (1 / (mu * qi_norm))) * q(j, :);
            else
                E(j, :) = 0;
            end
        end

        K = (X - E + (1 / mu) * Y2)' * U;
        DD=diag(1/norm(V));
%         DD = diag(1 ./ vecnorm(V));
       V= (eye(d)+ lambda1*DD+lambda4*QQ)*K;

        
        T = U - (1 / mu) * Y1 - (lambda2 / mu) * Q * U;
        Z = T;
        Z(Z < 0) = 0;
        N = (1 / mu) * Y1 + Z - lambda2 * Q * Z + (X - E + (1 / mu) * Y2) * V;
        [t1, ~, t2] = svd(N, 'econ');
        U = t1 * t2'; %t1=P, t2=Q
        Y1 = Y1 + mu * (Z - U);
        Y2 = Y2 + mu * (X - U * V' - E);
        mu = min([rho * mu, mu_max]);
    end

    [~, ret] = sort(sum(V.^2, 2), 'descend');
end
function entropy = calculate_shannon_entropy(data)
    % Calculate Shannon entropy for each feature in the data
    [~, n] = size(data);
    entropy = zeros(1, n);
    for i = 1:n
        feature_values = data(:, i);
        prob = histcounts(feature_values, 'Normalization', 'probability');
        prob = prob(prob > 0); % Remove zero probabilities
        entropy(i) = -sum(prob .* log2(prob));
    end
end

function entropy = calculate_renyi_entropy(data, alpha)
    % Calculate Rényi entropy for each feature in the data
    [~, n] = size(data);
    entropy = zeros(1, n);
    for i = 1:n
        feature_values = data(:, i);
        prob = histcounts(feature_values, 'Normalization', 'probability');
        prob = prob(prob > 0); % Remove zero probabilities
        entropy(i) = (1 / (1 - alpha)) * log2(sum(prob .^ alpha));
    end
end

function entropy = calculate_tsallis_entropy(data, q)
    % Calculate Tsallis entropy for each feature in the data
    [~, n] = size(data);
    entropy = zeros(1, n);
    for i = 1:n
        feature_values = data(:, i);
        prob = histcounts(feature_values, 'Normalization', 'probability');
        prob = prob(prob > 0); % Remove zero probabilities
        entropy(i) = (1 / (q - 1)) * (1 - sum(prob .^ q));
    end
end

function entropy = calculate_collision_entropy(data)
    % Calculate Collision entropy for each feature in the data
    [~, n] = size(data);
    entropy = zeros(1, n);
    for i = 1:n
        feature_values = data(:, i);
        prob = histcounts(feature_values, 'Normalization', 'probability');
        prob = prob(prob > 0); % Remove zero probabilities
        entropy(i) = -log2(sum(prob .^ 2));
    end
end
function Q = L1_entropy(data)
    % Calculate L1 entropy for each feature in the data
    [~, n] = size(data);
    Q = zeros(1, n);
    for i = 1:n
        feature_values = data(:, i);
        abs_diff = abs(feature_values - median(feature_values));
        Q(i) = 2 - sum(abs_diff) / n;
    end
end
function Q = L2_entropy(data)
    % Calculate L2 entropy for each feature in the data
    [~, n] = size(data);
    Q = zeros(1, n);
    for i = 1:n
        feature_values = data(:, i);
        Q(i) = 1 - (feature_values' * feature_values);
    end
end

