function [ model ] = NaiveBayes(XTrain, yTrain)

    classes = unique(yTrain);
    for i = 1:length(classes)
        class = classes(i);
        index = yTrain == class;
        model.Py(class) = sum(index) / size(yTrain,1);
        xTmp = XTrain(index, :);
        model.muX(class,:) = mean(xTmp);
        model.varX(class,:) = sqrt(var(xTmp));
    end
    
end

