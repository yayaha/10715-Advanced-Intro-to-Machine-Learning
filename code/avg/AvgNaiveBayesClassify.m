function [ predictions ] = AvgNaiveBayesClassify(model, XTest, Beta)
% Computes the posterior likelihood of X belongs to each class y.
% The log likelihood function is \sum_{i=1}^{N+1} \log(P(y^{(i)})) +
% \sum_{k=1}^{K} \log (\prod_{i=1}^{N+1} P(x^{(i)}_k + 1 / \beta
% \prod_{i=1}^{N+1} P(x_k^{(i)} | y^{(i)}).
% See solution pdf for more details.

    Y = zeros(size(XTest, 1), length(model.Py));
    un_cond_mu = repmat(model.mu, size(XTest, 1), 1);
    un_cond_sigma = repmat(model.sigma, size(XTest, 1), 1);
    un_cond_Px = normpdf(XTest, un_cond_mu, un_cond_sigma);
    tmp_un_cond_Px = log(un_cond_Px);
    
    for i = 1:length(model.Py)
        cond_mu = repmat(model.cond_mu(i,:), size(XTest, 1), 1);
        cond_sigma = repmat(model.cond_sigma(i,:), size(XTest, 1), 1);
        cond_Px = normpdf(XTest, cond_mu, cond_sigma);
        tmp_cond_Px = log(cond_Px);
        
        % Log trick to avoid floating point underflow
        % \log(e^a + e^b) = m + \log(e^{a-m} + b^{b-M}).
        a = tmp_un_cond_Px + repmat(model.sumPx, size(tmp_un_cond_Px, 1), 1);
        b = tmp_cond_Px + repmat(model.sum_cond_Px, size(tmp_cond_Px, 1), 1) - log(Beta); 
        m = max(a,b);
        l = m + log(exp(a-m) + exp(b-m));
        tmp_result = sum(l, 2) + log(model.Py(i));
        
        Y(:,i) = tmp_result;
    end
    [~, predictions] = max(Y, [], 2);
    
end