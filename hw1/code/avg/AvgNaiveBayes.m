function [ model ] = AvgNaiveBayes(XTrain, yTrain)
% This function computes all the neccessary probabilities for Average Naive
% Bayes classification.
%
% Each feature of X is assumed to conform a Gaussian distribution N(\mu_k,
% \sigma_k). 

    % Compute the unconditioned log probability of X.
    % \sum_{i=1}^{N} log(P(x_i))
    % Each column of sumPx corresponds to a feature.
    model.mu = mean(XTrain);
    model.sigma = std(XTrain);
    tmp_mu = repmat(model.mu, size(XTrain, 1), 1);
    tmp_sigma = repmat(model.sigma, size(XTrain, 1), 1);
    Px = - (XTrain - tmp_mu).^2 ./ (2 * tmp_sigma.^2) - log(sqrt(2*pi) * tmp_sigma);
    model.sumPx = sum(Px);

    % Estimate prior distribution of classes Py, and parameters of Gaussian
    % distributions conditioned on label y.
    classes = unique(yTrain);
    for i = 1:length(classes)
        class = classes(i);
        index = yTrain == class;
        model.Py(class) = sum(index) / size(yTrain,1);
        xTmp = XTrain(index, :);
        model.cond_mu(class,:) = mean(xTmp);
        model.cond_sigma(class,:) = std(xTmp);
    end
    
    % Compute the conditional log probability of X give label y.
    % \sum_{i=1}^{N} log(P(x_i | y_i))
    % Each column of sum_cond_Px corresponds to a feature.
    tmp_mu = model.cond_mu(yTrain,:);
    tmp_sigma = model.cond_sigma(yTrain,:);
    Px = normpdf(XTrain, tmp_mu, tmp_sigma);
    model.sum_cond_Px = sum(log(Px));
end

