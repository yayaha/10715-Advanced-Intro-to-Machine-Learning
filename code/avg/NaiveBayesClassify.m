function [predictions] = NaiveBayesClassify(model, XTest)
% Compute your probabilities from the model here
% predictions = ones(size(XTest,1),1);
    Y = zeros(size(XTest, 1), length(model.Py));
    for i = 1:length(model.Py)
        mu_i = model.muX(i,:);
        var_i = model.varX(i,:);
        mu = repmat(mu_i, size(XTest, 1), 1);
        sigma = repmat(var_i, size(XTest, 1), 1);
        tmpX = normpdf(XTest, mu, sigma);
        tmpX = log(tmpX);
        tmpX = sum(tmpX, 2);
        tmpX = log(model.Py(i)) + tmpX;
        Y(:,i) = tmpX;
    end
    [~, predictions] = max(Y, [], 2);
    
end