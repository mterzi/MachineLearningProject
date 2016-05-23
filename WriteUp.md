##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity 
relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements 
about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. 
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 
They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

###Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

###Data Preparation

Load the training and testing data sets, changing the values "#DIV/0!" and "" to NA.
```r
> train<-read.csv("~/Desktop/PracticalMachineLearning/pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
> test<-read.csv("~/Desktop/PracticalMachineLearning/pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))
```
Check the dimensions of the training and testing data sets
```r
> dim(train)
[1] 19622   160

> dim(test)
[1]  20 160
```
By looking at the column names, we see that some of them, namely the first seven 
("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window") are 
irrelevant for building a model, so we delete them.
```r
> train<-train[,-c(1:7)]
> test<-test[,-c(1:7)]
```
Now, first delete the columns with all missing values.
```r
> train<-train[, colSums(is.na(train)) != nrow(train)]
> test<-test[, colSums(is.na(test)) != nrow(test)]
> dim(train)
[1] 19622   147
> dim(test)
[1] 20 53
```
If we delete all the variables with missing values, then the dimensions of the training and testing sets will match:
```r
> dim(train[,colSums(is.na(train)) == 0])
[1] 19622    53
> dim(test[,colSums(is.na(test)) == 0])
[1] 20 53
```
Notice that one of the variable names doesn't macth:
```r
> colnames(test[,colSums(is.na(test)) == 0])==colnames(train[,colSums(is.na(train)) == 0])

 [1]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
[14]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
[27]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
[40]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
[53] FALSE
```
By checking the variable names we see that it is variable "classe" (response) for the training set, and "problem_id" for the testing set. 
So, we delete the "problem_id" variable and fix our training and testing data sets.
```r
> train<-train[,colSums(is.na(train)) == 0]
> test<-test[,colSums(is.na(test)) == 0]
> test<-test[,-c(53)]
```
Test set doesn't have the "classe" variable, since that is the one we are going to predict. So, our model will have 52 predictors.

###Cross Validation
The test set is large enough (19622 observations) to be devided into sub-training set (on which we will build the model) and 
sub-testing set 
(on which we will test the model and estimate the out-of-sample error)
```r
>> inTrain<-createDataPartition(y=train$classe,p=0.75,list=FALSE)
> subtrain<-train[inTrain,]
> subtest<-train[-inTrain,]
> dim(subtrain)
[1] 14718    53
> dim(subtest)
[1] 4904   53
> dim(test)
[1] 20 52
```
##Building the models

#####First prediction model: Using Decision Tree
```r
> library(rpart)
> library(rpart.plot)
> model1 <- rpart(classe ~ ., data=subtrain, method="class")
> prediction1 <- predict(model1, subtest, type = "class")
>rpart.plot(model1, extra=102, under=TRUE, faclen=0)
```
Test results on the subtest data set
```r
> confusionMatrix(prediction1, subtest$classe)
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1242  171   24   63   29
         B   49  569   82   59   72
         C   28   86  677  114   99
         D   56   80   52  524   66
         E   20   43   20   44  635

Overall Statistics
                                          
               Accuracy : 0.7437          
                 95% CI : (0.7312, 0.7559)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.6748          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.8903   0.5996   0.7918   0.6517   0.7048
Specificity            0.9182   0.9338   0.9192   0.9380   0.9683
Pos Pred Value         0.8123   0.6847   0.6743   0.6735   0.8333
Neg Pred Value         0.9547   0.9067   0.9544   0.9321   0.9358
Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
Detection Rate         0.2533   0.1160   0.1381   0.1069   0.1295
Detection Prevalence   0.3118   0.1695   0.2047   0.1586   0.1554
Balanced Accuracy      0.9043   0.7667   0.8555   0.7949   0.8365
>
```
#####Second prediction model: Using Random Forest
```r
> library(randomForest)
randomForest 4.6-10
Type rfNews() to see new features/changes/bug fixes.

> model2 <- randomForest(classe ~. , data=subtrain, method="class")
```
Test results on the subtest data set
```r
> prediction2 <- predict(model2, subtest, type = "class")
> confusionMatrix(prediction2, subtest$classe)
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1395    4    0    0    0
         B    0  943    2    0    0
         C    0    2  852    4    0
         D    0    0    1  800    6
         E    0    0    0    0  895

Overall Statistics
                                         
               Accuracy : 0.9961         
                 95% CI : (0.994, 0.9977)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.9951         
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9937   0.9965   0.9950   0.9933
Specificity            0.9989   0.9995   0.9985   0.9983   1.0000
Pos Pred Value         0.9971   0.9979   0.9930   0.9913   1.0000
Neg Pred Value         1.0000   0.9985   0.9993   0.9990   0.9985
Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
Detection Rate         0.2845   0.1923   0.1737   0.1631   0.1825
Detection Prevalence   0.2853   0.1927   0.1750   0.1646   0.1825
Balanced Accuracy      0.9994   0.9966   0.9975   0.9967   0.9967
>
```
###Conclusion
We choose the Random Forest model since its accuracy is very high, 0.9961 (vs. 0.7437). 
The expected out-of-sample error is 1 - accuracy = 0.0039, or 0.39%, so, we can expect 
that very few, or none, of the 20 test samples will be missclassified.
