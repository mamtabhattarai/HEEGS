function accuracy = clusterAccuracy1(true_labels, predicted_clusters)
    unique_clusters = unique(predicted_clusters);
    total_correct = 0;
    for i = 1:numel(unique_clusters)
        cluster_label = unique_clusters(i);
        cluster_indices = predicted_clusters == cluster_label;  % Logical indexing
        cluster_labels = true_labels(cluster_indices);
        [~, majority_class] = max(histcounts(cluster_labels)); % Find majority class
        total_correct = total_correct + sum(cluster_labels == majority_class);
    end
    accuracy = total_correct / numel(true_labels);
end

% Example usage

