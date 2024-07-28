function nmi = NMI_calculate(labels_true, labels_pred)
    % Calculate the confusion matrix
    num_samples = length(labels_true);
    num_classes = max(labels_true);

    confusion_matrix = zeros(num_classes, num_classes);
    for i = 1:num_samples
        confusion_matrix(labels_true(i), labels_pred(i)) = confusion_matrix(labels_true(i), labels_pred(i)) + 1;
    end

    % Calculate marginal probabilities
    sum_true = sum(confusion_matrix, 2);
    sum_pred = sum(confusion_matrix, 1);
    P_true = sum_true / num_samples;
    P_pred = sum_pred / num_samples;

    % Calculate conditional probabilities
    P_joint = confusion_matrix / num_samples;

    % Calculate entropy of true labels and predicted labels
    H_true = -sum(P_true .* log(P_true + eps));
    H_pred = -sum(P_pred .* log(P_pred + eps));

    % Calculate mutual information
    MI = sum(sum(P_joint .* log((P_joint + eps) ./ (P_true * P_pred))));

    % Calculate normalized mutual information
    nmi = 2 * MI / (H_true + H_pred);
end


