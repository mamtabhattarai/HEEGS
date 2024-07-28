addpath(genpath('.\'));
clear; clc;

% Define datasets to load
datasets = {'Prostate-GE.mat'}; 

% Initialize result storage
results = {};

% Define parameter ranges
lambda_range = logspace(-3, 3, 7);
entropy_types = {'shannon', 'renyi', 'tsallis', 'collision'};

% Loop through each dataset
for d = 1:length(datasets)
    dataset = datasets{d};
    load(dataset);
    
    K1 = length(unique(Y));
    K = 5;
    X = zscore(X);
    
    % Randomize the data
    n = size(X, 1);
    perm = randperm(n);
    X = X(perm, :);
    Y = Y(perm);

    para.maxiter = 100;
    fold = 10;
    para.iter = 20;
    
    rng(15, "twister");

    % Initialize best results
    best_result = struct('nmi', 0, 'nmi_std', 0, 'nmi_K', 0, 'acc', 0, 'acc_std', 0, 'acc_K', 0, 'class_acc', 0, 'class_acc_std', 0, 'class_acc_K', 0, 'lambda1', 0, 'lambda2', 0, 'lambda4', 0, 'entropy_type', '');

    % Parameter tuning
    for lambda1 = lambda_range
        for lambda2 = lambda_range
            for lambda4 = lambda_range
                para.lambda1 = lambda1;
                para.lambda2 = lambda2;
                para.lambda4 = lambda4;

                for entropy_type = entropy_types
                    Ind = edufs_entropy(X, K, para, entropy_type{1});

                    % Select top K features
                    K_features = [50, 100, 150, 200, 250, 300];

                    for k = K_features
                        Ind_k = Ind(1:k, 1);
                        Ind_k = sort(Ind_k);
                        X_k = X(:, Ind_k);
                        data = cat(2, X_k, Y);
                        method = 'KNN';
                        varargin = 0;

                        [class_acc_mean_k, class_acc_std_k] = crossvalidate(data, fold, method, varargin);

                        nmi_values = zeros(1, para.iter);
                        acc_values = zeros(1, para.iter);

                        for ii = 1:para.iter
                            Ynew = kmeans(X_k, K1, 'Emptyaction', 'drop');
                            nmi_values(ii) = NMI_calculate(Y, Ynew);
                            acc_values(ii) = clusterAccuracy1(Y, Ynew);
                        end

                        avg_nmi = mean(nmi_values);
                        std_nmi = std(nmi_values);
                        avg_acc = mean(acc_values);
                        std_acc = std(acc_values);

                        if avg_nmi > best_result.nmi
                            best_result.nmi = avg_nmi;
                            best_result.nmi_std = std_nmi;
                            best_result.nmi_K = k;
                            best_result.lambda1 = lambda1;
                            best_result.lambda2 = lambda2;
                            best_result.lambda4 = lambda4;
                            best_result.entropy_type = entropy_type{1};
                        end

                        if avg_acc > best_result.acc
                            best_result.acc = avg_acc;
                            best_result.acc_std = std_acc;
                            best_result.acc_K = k;
                        end

                        if class_acc_mean_k > best_result.class_acc
                            best_result.class_acc = class_acc_mean_k;
                            best_result.class_acc_std = class_acc_std_k;
                            best_result.class_acc_K = k;
                        end
                    end
                end
            end
        end
    end

    % Store the results for the current dataset
    results{end+1, 1} = dataset;
    results{end, 2} = best_result.nmi;
    results{end, 3} = best_result.nmi_std;
    results{end, 4} = best_result.nmi_K;
    results{end, 5} = best_result.acc;
    results{end, 6} = best_result.acc_std;
    results{end, 7} = best_result.acc_K;
    results{end, 8} = best_result.class_acc;
    results{end, 9} = best_result.class_acc_std;
    results{end, 10} = best_result.class_acc_K;
    results{end, 11} = best_result.lambda1;
    results{end, 12} = best_result.lambda2;
    results{end, 13} = best_result.lambda4;
    results{end, 14} = best_result.entropy_type;
end

% Convert results to a table and write to an Excel file
results_table = cell2table(results, 'VariableNames', {'Dataset', 'Best_NMI', 'NMI_STD', 'Best_NMI_K', 'Best_Clustering_Accuracy', 'Clustering_STD', 'Best_Clustering_K', 'Best_Classification_Accuracy', 'Classification_STD', 'Best_Classification_K', 'Best_Lambda1', 'Best_Lambda2', 'Best_Lambda3', 'Entropy_Type'});
writetable(results_table, 'results.xlsx');

fprintf('Results saved to results.xlsx\n');