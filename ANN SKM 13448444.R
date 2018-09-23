#=======================================================================================================================
#=======================================================================================================================
#  
#                                       Title: ANN - Mexican Hat Problem
#                                        Author: Shantesh Mani
#                                        
#                                        
#                                        Created: 10 June 2018
#                                        Last Edit: 20 June 2018
#                             
#                                          Version: 0.01.09
#                             
#                                          Copyright: 2018
# Notes:
  
# Script name - ANN SKM 13448444.R
  
# ANN Models to learn and classify the mexican hat problem.
#---------------------------------------------------------------------------------------------------------------------- 

# WARNING: Script lines 29-33 will install required packages onto pc if not installed already.  
                            
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------  


rm(list=ls(all=TRUE))


# Ensuring the correct packages are installed before executing the script

if("neuralnet" %in% rownames(installed.packages())==FALSE) {install.packages("neuralnet")}
if("caret" %in% rownames(installed.packages())==FALSE) {install.packages("caret")}
if("parallel" %in% rownames(installed.packages())==FALSE) {install.packages("parallel")}
if("doParallel" %in% rownames(installed.packages())==FALSE) {install.packages("doParalell")}
if("doMC" %in% rownames(installed.packages())==FALSE) {install.packages("doMC")}

#Run models on all cores
library(parallel)
detectCores()

if (Sys.info() ['sysname'] == "Windows")
{ 
  library(doParallel)
  registerDoParallel(cores=detectCores())  
} else {
  library(doMC)  
  registerDoMC(cores=detectCores())
}



#Before building an ANN model, we must understand the affects of Lambda, being errors or noise in the dataset and how that affects the distribution.


plot.y <- function(number,min,max,mean,lambda,main) {
  x <- runif(number,min=min,max=max)
  error <- rnorm(number, mean = mean, sd = (lambda/(1+Mod(x))))
  y <- sin(x)/x
  new.y <- (y+error)
  plot(x,new.y,main=main)
}

par(mfrow=c(2,2))
set.seed(100)
plot.y(2000,-4*pi,4*pi,0,1, "Sinc function with lambda 1")
plot.y(2000,-4*pi,4*pi,0,0.75, "Sinc function with lambda 0.75")
plot.y(2000,-4*pi,4*pi,0,0.5, "Sinc function with lambda 0.5")
plot.y(2000,-4*pi,4*pi,0,0.25, "Sinc function with lambda 0.25")
plot.y(2000,-4*pi,4*pi,0,0.1, "Sinc function with lambda 0.1")
plot.y(2000,-4*pi,4*pi,0,0.01, "Sinc function with lambda 0.01")
plot.y(2000,-4*pi,4*pi,0,0.001, "Sinc function with lambda 0.001")
plot.y(2000,-4*pi,4*pi,0,0, "Sinc function with lambda 0")

# Each graph represents how much noise will be in the dataset and how well the ANN model can learn with such errors.


# loading the package needed in the assignment
library(neuralnet)
library(caret)


n <- 2000    # number of data points

# set up the boundary for random variable x
set.seed(1)
x <- runif(n, min=-4*pi, max=4*pi)

# create the mathematical function for creating random data

sinc <- function(x){
  y <- sin(x)/x
  y [x==0] <- 1
  y
}

lambda <- function(a){
  a/(1+abs(x))
  a
}


set.seed(1)
b <- lambda(0) # create the perfect data first
e <- rnorm(n,mean=0,sd=b) # the measurement error terms


# create the first dataset
set.seed(1)
output <- sinc(x)+e
input <- x
d1 <- data.frame(input,output)

# spilt the dataset into train, validate, test data
set.seed(1)
size1 <- floor(0.5*nrow(d1))    
train_ind <- sample(seq(nrow(d1)),size=size1)

trainset <- d1[train_ind,]
restofd1 <- d1[-train_ind,]

set.seed(1)
size2 <- floor(0.5*nrow(restofd1))
test_ind <- sample(seq(nrow(restofd1)),size=size2)

testset <- restofd1[test_ind,]
valiset <- restofd1[-test_ind,]

# plot the input and output
plot(input,output, main="Graph of Dataset")



# create the ANN model 
# test the model with different number of hidden neurons in each layer, threshold=0.01 and tangent hyperbolicus  
nntest <- lapply(seq(from=5,to=50,by=5),function(hidden){
  net <- neuralnet(output~input, data=trainset, hidden=hidden, threshold = 0.01, act.fct="tanh", stepmax = 10000000,rep=1)
  vald <- compute(net,valiset[1])
  com <- cbind(valiset, as.data.frame(vald$net.result))
  colnames(com) <- c("Input","Expected Output","Neural Net Output")
  sqrt(mean((com$"Neural Net Output"-com$"Expected Output")^2))
}
)

testresult <- do.call(rbind, nntest)
nnresult <-cbind(as.data.frame(seq(from=5,to=50,by=5)),testresult)
colnames(nnresult) <- c("number of hidden neurons","RMSE")
plot(nnresult,main="Threshold = 0.01 & tanh") #20 neurons being ideal with this model.


# test the model with different hidden neurons, threshold=0.1 and tangent hyperbolicus  
nntest2 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  net2 <- neuralnet(output~input, data=trainset, hidden=hidden, threshold = 0.1, act.fct="tanh", stepmax = 10000000,rep=1)
  val2 <- compute(net2,valiset[1])
  com2 <- cbind(valiset, as.data.frame(val2$net.result))
  colnames(com2) <- c("Input","Expected Output","Neural Net Output")
  sqrt(mean((com2$"Neural Net Output"-com2$"Expected Output")^2))
}
)

testresult2 <- do.call(rbind, nntest2)
nnresult2 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),testresult2)
colnames(nnresult2) <- c("number of hidden neurons","RMSE")
plot(nnresult2,main="Threshold = 0.1 & tanh") #There is not much difference between threshold of 0.01 and 0.1. 20 neurons being ideal in both models



# test the model with different hidden neurons, threshold=1 and tangent hyperbolicus  
nntest3 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  net3 <- neuralnet(output~input, data=trainset, hidden=hidden, threshold = 1, act.fct="tanh", stepmax = 10000000,rep=1)
  val3 <- compute(net3,valiset[1])
  com3 <- cbind(valiset, as.data.frame(val3$net.result))
  colnames(com3) <- c("Input","Expected Output","Neural Net Output")
  sqrt(mean((com3$"Neural Net Output"-com3$"Expected Output")^2))
}
)

testresult3 <- do.call(rbind, nntest3)
nnresult3 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),testresult3)
colnames(nnresult3) <- c("number of hidden neurons","RMSE")
plot(nnresult3,main="Threshold = 1 & tanh") #This threshold figure proves that the ANN is struggling to learn with errors.



# test the model with different hidden neurons, threshold=0.001 and tangent hyperbolicus  
nntest4 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  net4 <- neuralnet(output~input, data=trainset, hidden=hidden, threshold = 0.001, act.fct="tanh", stepmax = 100000000,rep=1)
  val4 <- compute(net4,valiset[1])
  com4 <- cbind(valiset, as.data.frame(val4$net.result))
  colnames(com4) <- c("Input","Expected Output","Neural Net Output")
  sqrt(mean((com4$"Neural Net Output"-com4$"Expected Output")^2))
}
)

testresult4 <- do.call(rbind, nntest4)
nnresult4 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),testresult4)
colnames(nnresult4) <- c("number of hidden neurons","RMSE")
plot(nnresult4,main="Threshold = 0.001 & tanh") #This model goes to the extreme with the threshold being set at 0.001. the ideal neurons is 20. With a gradient decent model, the experiment proves that 20 neurons is ideal with limiting resources, the ANN model with a gradient decent threshold of 0.1 or 0.01 would suffice as they provide the same result before overfitting occurs.



# test the model with different hidden neurons, threshold=0.01 and logistic function  
nntestlog <- lapply(seq(from=5,to=50,by=5),function(hidden){
  netlog <- neuralnet(output~input, data=trainset, hidden=hidden, threshold = 0.01, act.fct="logistic", stepmax = 10000000,rep=1)
  valdlog <- compute(netlog,valiset[1])
  comlog <- cbind(valiset, as.data.frame(valdlog$net.result))
  colnames(comlog) <- c("Input","Expected Output","Neural Net Output")
  sqrt(mean((comlog$"Neural Net Output"-comlog$"Expected Output")^2))
}
)

testresultlog <- do.call(rbind, nntestlog)
nnresultlog <-cbind(as.data.frame(seq(from=5,to=50,by=5)),testresultlog)
colnames(nnresultlog) <- c("number of hidden neurons","RMSE")
plot(nnresultlog,main="Threshold = 0.01 & logistic") #This model does not perform as well as the tanh activation function. Ideal neurons for this model is 35.



# test the model with different hidden neurons, threshold=0.1 and logistic function  
nntestlog2 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  netlog2 <- neuralnet(output~input, data=trainset, hidden=hidden, threshold = 0.1, act.fct="logistic", stepmax = 10000000,rep=1)
  valdlog2 <- compute(netlog2,valiset[1])
  comlog2 <- cbind(valiset, as.data.frame(valdlog2$net.result))
  colnames(comlog2) <- c("Input","Expected Output","Neural Net Output")
  sqrt(mean((comlog2$"Neural Net Output"-comlog2$"Expected Output")^2))
}
)

testresultlog2 <- do.call(rbind, nntestlog2)
nnresultlog2 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),testresultlog2)
colnames(nnresultlog2) <- c("number of hidden neurons","RMSE")
plot(nnresultlog2,main="Threshold = 0.1 & logistic") # This model outperformed the previous model with a higher threshold. That could be due to the random starting weights in an ANN model. Ideal neurons for this model is 20.


# test the model with different hidden neurons, threshold=1 and logistic function  
nntestlog3 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  netlog3 <- neuralnet(output~input, data=trainset, hidden=hidden, threshold = 1, act.fct="logistic", stepmax = 10000000,rep=1)
  valdlog3 <- compute(netlog3,valiset[1])
  comlog3 <- cbind(valiset, as.data.frame(valdlog3$net.result))
  colnames(comlog3) <- c("Input","Expected Output","Neural Net Output")
  sqrt(mean((comlog3$"Neural Net Output"-comlog3$"Expected Output")^2))
}
)

testresultlog3 <- do.call(rbind, nntestlog3)
nnresultlog3 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),testresultlog3)
colnames(nnresultlog3) <- c("number of hidden neurons","RMSE")
plot(nnresultlog3,main="Threshold = 1 & logistic") #This model does not perform well. There is insufficient learning ability with such a high threshold. 



# test the model with different hidden neurons, threshold=0.001 and logistic function  
nntestlog4 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  netlog4 <- neuralnet(output~input, data=trainset, hidden=hidden, threshold = 0.001, act.fct="logistic", stepmax = 10000000,rep=1)
  valdlog4 <- compute(netlog4,valiset[1])
  comlog4 <- cbind(valiset, as.data.frame(valdlog4$net.result))
  colnames(comlog4) <- c("Input","Expected Output","Neural Net Output")
  sqrt(mean((comlog4$"Neural Net Output"-comlog4$"Expected Output")^2))
}
)

testresultlog4 <- do.call(rbind, nntestlog4)
nnresultlog4 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),testresultlog4)
colnames(nnresultlog4) <- c("number of hidden neurons","RMSE")
plot(nnresultlog4,main="Threshold = 0.001 & logistic")

# test the accuracy by rmse, when hidden=1, threhold=1, with different functions
netd1=neuralnet(output~input, data=trainset, hidden=1, threshold = 1, act.fct="tanh", stepmax = 100000,rep=1)
vald1 <- compute(netd1,valiset[1])
comd1 <- cbind(valiset, as.data.frame(vald1$net.result))
colnames(comd1) <- c("Input","Expected Output","Neural Net Output")
sqrt(mean((comd1$"Neural Net Output"-comd1$"Expected Output")^2)) 

netf1=neuralnet(output~input, data=trainset, hidden=1, threshold = 1,act.fct="logistic", stepmax = 100000,rep=1)
valf1 <- compute(netf1,valiset[1])
comf1 <- cbind(valiset, as.data.frame(valf1$net.result))
colnames(comf1) <- c("Input","Expected Output","Neural Net Output")
sqrt(mean((comf1$"Neural Net Output"-comf1$"Expected Output")^2))
## The RMSE in both models are very high, around 0.2901 to 0.2913
## for logistic function, the more number of hidden neurons in the layer has the smaller value of RMSE
## although when threshold = 0.001, the model performs quite good, but the computational resources are excessive for this model
## Using tanh as the activation function performs significantly better than the logistic activation function. 


## Use the average errors to find the best number of hidden neurons, when threshold = 0.01 & tanh activation function
maetest <- lapply(seq(from=5,to=50,by=5),function(hidden){
  mnet <- neuralnet(output~input, data=trainset, hidden=hidden, threshold = 0.01, act.fct="tanh", stepmax = 10000000,rep=1)
  mvald <- compute(mnet,valiset[1])
  mcom <- cbind(valiset, as.data.frame(mvald$net.result))
  colnames(mcom) <- c("Input","Expected Output","Neural Net Output")
  mean(abs(mcom$"Neural Net Output"-mcom$"Expected Output")/abs(mcom$"Expected Output"))
}
)

mtestresult <- do.call(rbind, maetest)
mnnresult <-cbind(as.data.frame(seq(from=5,to=50,by=5)),mtestresult)
colnames(mnnresult) <- c("number of hidden neurons","Average Errors")
plot(mnnresult,main="Threshold = 0.01 & tanh")



## Ideal model is hidden = 20, threshold = 0.01 in tangent hyperbolicus, as the number of neurons increase past this amount, overfitting occurs.  
modela <- neuralnet(output~input, data=trainset, hidden=20, threshold = 0.01, act.fct="tanh", stepmax = 10000000,rep=1)
modela
plot(modela, rep="best", intercept=FALSE) 


# Use the testing dataset to test this ideal model with different values of lambda
# Calculate RMSE in different values of lambda to check how they perform
set.seed(1)
testlambda <- lapply(seq(from=0.01,to=25.01,by=0.1),function(b){
  sd <- b/(1+abs(x))
  e <- rnorm(1,0,sd)
  xwithe <- testset+e
  model <- neuralnet(output~input, data=trainset, hidden=20, threshold = 0.01, act.fct="tanh", stepmax =   10000000,rep=1)
  result <- compute(model,xwithe[1])
  commodel <- cbind(xwithe, as.data.frame(result$net.result))
  colnames(commodel) <- c("Input","Expected Output","Neural Net Output")
  sqrt(mean((commodel$"Neural Net Output"-commodel$"Expected Output")^2))
}  
)

# visualising the result
lambdaresult <- do.call(rbind, testlambda)
perform <- cbind(as.data.frame(seq(from=0.01, to=25.01, by=0.1)),lambdaresult)
colnames(perform) <- c("Value of lambda","RMSE of model")
plot(perform,main="The accuracy of the model with errors")

#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################

# Part B, The binary values are to be sign(y).
# create the second dataset with binary classification
set.seed(1)
secoutput <- sinc(x)+e
secinput <- x
secd1 <- data.frame(secinput,secoutput)

secd1$positive = c(secd1$secoutput>0)
secd1$negative = c(secd1$secoutput<0)
secd1$secoutput = NULL

for (i in 1:200){if(secd1$positive[i]=="TRUE"){secd1$positive[i]=1}}
for (i in 1:200){if(secd1$negative[i]=="TRUE"){secd1$negative[i]=1}}


# spilt the dataset into train, validate, test data
set.seed(1)
secsize1 <- floor(0.5*nrow(secd1))    
sectrain_ind <- sample(seq(nrow(secd1)),size=secsize1)

sectrainset <- secd1[sectrain_ind,]
restofsecd1 <- secd1[-sectrain_ind,]

secsize2 <- floor(0.5*nrow(restofsecd1))
sectest_ind <- sample(seq(nrow(restofsecd1)),size=secsize2)

sectestset <- restofsecd1[sectest_ind,]
secvaliset <- restofsecd1[-sectest_ind,]

# Create the ANN model and test the model to find the best one
# test with more hidden neurons, when threshold=0.01 and tangent hyperbolicus
secnntest <- lapply(seq(from=5,to=50,by=5),function(hidden){
  secnet <- neuralnet(positive+negative~secinput, data=sectrainset, hidden=hidden, threshold = 0.01, act.fct="tanh", stepmax = 10000000,rep=1)
  secvald <- compute(secnet,secvaliset[1])
  seccom=NULL
  for (i in 1:250){seccom[i]=which.max(secvald$net.result[i,])}
  for (i in 1:250){if(seccom[i]==1){seccom[i]="positive"}}
  for (i in 1:250){if(seccom[i]==2){seccom[i]="negative"}}
  secvs <- secvaliset
  secvs$output=c(secvs$positive>secvs$negative)
  for (i in 1:250){if(secvs$output[i]=="TRUE"){secvs$output[i]="positive"}}
  for (i in 1:250){if(secvs$output[i]=="FALSE"){secvs$output[i]="negative"}}
  result <- table(seccom, secvs$output)
  probability=(result[1,1]+result[2,2])/250
}
)

sectestresult <- do.call(rbind, secnntest)
secnnresult <-cbind(as.data.frame(seq(from=5,to=50,by=5)),sectestresult)
colnames(secnnresult) <- c("Number of hidden neurons","Accuracy of predicting sign(y)")
plot(secnnresult,main="Threshold = 0.01 & tanh")
## threshold = 0.01 and tanh perform well in any number of hidden neurons, smallest accuracy of 90% with 3 neurons.


# test with more hidden neurons, when threshold=1 and tangent hyperbolicus
secnntest2 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  secnet2 <- neuralnet(positive+negative~secinput, data=sectrainset, hidden=hidden, threshold = 1, act.fct="tanh", stepmax = 10000000,rep=1)
  secvald2 <- compute(secnet2,secvaliset[1])
  seccom2=NULL
  for (i in 1:250){seccom2[i]=which.max(secvald2$net.result[i,])}
  for (i in 1:250){if(seccom2[i]==1){seccom2[i]="positive"}}
  for (i in 1:250){if(seccom2[i]==2){seccom2[i]="negative"}}
  secvs2 <- secvaliset
  secvs2$output=c(secvs2$positive>secvs2$negative)
  for (i in 1:250){if(secvs2$output[i]=="TRUE"){secvs2$output[i]="positive"}}
  for (i in 1:250){if(secvs2$output[i]=="FALSE"){secvs2$output[i]="negative"}}
  result2 <- table(seccom2, secvs2$output)
  probability=(result2[1,1]+result2[2,2])/250
}
)

sectestresult2 <- do.call(rbind, secnntest2)
secnnresult2 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),sectestresult2)
colnames(secnnresult2) <- c("Number of hidden neurons","Accuracy of predicting sign(y)")
plot(secnnresult2,main="Threshold = 1 & tanh")
## threshold = 1 and tanh perform well in any number of hidden neurons


# test with more hidden neurons, when threshold=0.1 and tangent hyperbolicus

secnntest3 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  secnet3 <- neuralnet(positive+negative~secinput, data=sectrainset, hidden=hidden, threshold = 0.1, act.fct="tanh", stepmax = 10000000,rep=1)
  secvald3 <- compute(secnet3,secvaliset[1])
  seccom3=NULL
  for (i in 1:250){seccom3[i]=which.max(secvald3$net.result[i,])}
  for (i in 1:250){if(seccom3[i]==1){seccom3[i]="positive"}}
  for (i in 1:250){if(seccom3[i]==2){seccom3[i]="negative"}}
  secvs3 <- secvaliset
  secvs3$output=c(secvs3$positive>secvs3$negative)
  for (i in 1:250){if(secvs3$output[i]=="TRUE"){secvs3$output[i]="positive"}}
  for (i in 1:250){if(secvs3$output[i]=="FALSE"){secvs3$output[i]="negative"}}
  result3 <- table(seccom3, secvs3$output)
  probability=(result3[1,1]+result3[2,2])/250
}
)

sectestresult3 <- do.call(rbind, secnntest3)
secnnresult3 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),sectestresult3)
colnames(secnnresult3) <- c("Number of hidden neurons","Accuracy of predicting sign(y)")
plot(secnnresult3,main="Threshold = 0.1 & tanh")

## threshold = 0.1 and tanh perform well in any number of hidden neurons 


# test with more hidden neurons, when threshold=0.001 and tangent hyperbolicus
secnntest4 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  secnet4 <- neuralnet(positive+negative~secinput, data=sectrainset, hidden=hidden, threshold = 0.001, act.fct="tanh", stepmax = 1000000000,rep=1)
  secvald4 <- compute(secnet4,secvaliset[1])
  seccom4=NULL
  for (i in 1:250){seccom4[i]=which.max(secvald4$net.result[i,])}
  for (i in 1:250){if(seccom4[i]==1){seccom4[i]="positive"}}
  for (i in 1:250){if(seccom4[i]==2){seccom4[i]="negative"}}
  secvs4 <- secvaliset
  secvs4$output=c(secvs4$positive>secvs4$negative)
  for (i in 1:250){if(secvs4$output[i]=="TRUE"){secvs4$output[i]="positive"}}
  for (i in 1:250){if(secvs4$output[i]=="FALSE"){secvs4$output[i]="negative"}}
  result4 <- table(seccom4, secvs4$output)
  probability=(result4[1,1]+result4[2,2])/250
}
)

sectestresult4 <- do.call(rbind, secnntest4)
secnnresult4 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),sectestresult4)
colnames(secnnresult4) <- c("Number of hidden neurons","Accuracy of predicting sign(y)")
plot(secnnresult4,main="Threshold = 0.001 & tanh")

## threshold=0.001 and tanh perform well in any number of hidden neurons
## this model was excluded from the final paper as the resources were limited and this model having such a low threshold, the time it took for the algortihm to converge was over 8 hours alone.
# The results at 0.1 and 0.01 was sufficient to make accurate classifications

# test with more hidden neurons, when threshold=0.01 and logistic function
secnntestf <- lapply(seq(from=5,to=50,by=5),function(hidden){
  secnetf <- neuralnet(positive+negative~secinput, data=sectrainset, hidden=hidden, threshold = 0.01, act.fct="logistic", stepmax = 10000000,rep=1)
  secvaldf <- compute(secnetf,secvaliset[1])
  seccomf=NULL
  for (i in 1:250){seccomf[i]=which.max(secvaldf$net.result[i,])}
  for (i in 1:250){if(seccomf[i]==1){seccomf[i]="positive"}}
  for (i in 1:250){if(seccomf[i]==2){seccomf[i]="negative"}}
  secvsf <- secvaliset
  secvsf$output=c(secvsf$positive>secvsf$negative)
  for (i in 1:250){if(secvsf$output[i]=="TRUE"){secvsf$output[i]="positive"}}
  for (i in 1:250){if(secvsf$output[i]=="FALSE"){secvsf$output[i]="negative"}}
  resultf <- table(seccomf, secvsf$output)
  probability=(resultf[1,1]+resultf[2,2])/250
}
)

sectestresultf <- do.call(rbind, secnntestf)
secnnresultf <-cbind(as.data.frame(seq(from=5,to=50,by=5)),sectestresultf)
colnames(secnnresultf) <- c("number of hidden neurons","Accuracy of predicting sign(y)")
plot(secnnresultf,main="Threshold = 0.01 & logistic")

## threshold=0.01 and log perform well in any number of hidden neurons 

# test with more hidden neurons, when threshold=1 and logistic function
secnntestf2 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  secnetf2 <- neuralnet(positive+negative~secinput, data=sectrainset, hidden=hidden, threshold = 1, act.fct="logistic", stepmax = 10000000,rep=1)
  secvaldf2 <- compute(secnetf2,secvaliset[1])
  seccomf2=NULL
  for (i in 1:250){seccomf2[i]=which.max(secvaldf2$net.result[i,])}
  for (i in 1:250){if(seccomf2[i]==1){seccomf2[i]="positive"}}
  for (i in 1:250){if(seccomf2[i]==2){seccomf2[i]="negative"}}
  secvsf2 <- secvaliset
  secvsf2$output=c(secvsf2$positive>secvsf2$negative)
  for (i in 1:250){if(secvsf2$output[i]=="TRUE"){secvsf2$output[i]="positive"}}
  for (i in 1:250){if(secvsf2$output[i]=="FALSE"){secvsf2$output[i]="negative"}}
  resultf2 <- table(seccomf2, secvsf2$output)
  probability=(resultf2[1,1]+resultf2[2,2])/250
}
)

sectestresultf2 <- do.call(rbind, secnntestf2)
secnnresultf2 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),sectestresultf2)
colnames(secnnresultf2) <- c("Number of hidden neurons","Accuracy of predicting sign(y)")
plot(secnnresultf2,main="Threshold = 1 & logistic")
## threshold=1 and log perform well in any number of hidden neurons   

# test with more hidden neurons, when threshold=0.1 and logistic function
secnntestf3 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  secnetf3 <- neuralnet(positive+negative~secinput, data=sectrainset, hidden=hidden, threshold = 0.1, act.fct="logistic", stepmax = 10000000,rep=1)
  secvaldf3 <- compute(secnetf3,secvaliset[1])
  seccomf3=NULL
  for (i in 1:250){seccomf3[i]=which.max(secvaldf3$net.result[i,])}
  for (i in 1:250){if(seccomf3[i]==1){seccomf3[i]="positive"}}
  for (i in 1:250){if(seccomf3[i]==2){seccomf3[i]="negative"}}
  secvsf3 <- secvaliset
  secvsf3$output=c(secvsf3$positive>secvsf3$negative)
  for (i in 1:250){if(secvsf3$output[i]=="TRUE"){secvsf3$output[i]="positive"}}
  for (i in 1:250){if(secvsf3$output[i]=="FALSE"){secvsf3$output[i]="negative"}}
  resultf3 <- table(seccomf3, secvsf3$output)
  probability=(resultf3[1,1]+resultf3[2,2])/250
}
)

sectestresultf3 <- do.call(rbind, secnntestf3)
secnnresultf3 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),sectestresultf3)
colnames(secnnresultf3) <- c("Number of hidden neurons","Accuracy of predicting sign(y)")
plot(secnnresultf3,main="Threshold=0.1 & logistic")
## threshold=0.1 and log perform well in any number of hidden neurons

# test with more hidden neurons, when threshold=0.001 and logistic function
secnntestf4 <- lapply(seq(from=5,to=50,by=5),function(hidden){
  secnetf4 <- neuralnet(positive+negative~secinput, data=sectrainset, hidden=hidden, threshold = 0.001, act.fct="logistic", stepmax = 1000000000,rep=1)
  secvaldf4 <- compute(secnetf4,secvaliset[1])
  seccomf4=NULL
  for (i in 1:250){seccomf4[i]=which.max(secvaldf4$net.result[i,])}
  for (i in 1:250){if(seccomf4[i]==1){seccomf4[i]="positive"}}
  for (i in 1:250){if(seccomf4[i]==2){seccomf4[i]="negative"}}
  secvsf4 <- secvaliset
  secvsf4$output=c(secvsf4$positive>secvsf4$negative)
  for (i in 1:250){if(secvsf4$output[i]=="TRUE"){secvsf4$output[i]="positive"}}
  for (i in 1:250){if(secvsf4$output[i]=="FALSE"){secvsf4$output[i]="negative"}}
  resultf4 <- table(seccomf4, secvsf4$output)
  probability=(resultf4[1,1]+resultf4[2,2])/250
}
)

sectestresultf4 <- do.call(rbind, secnntestf4)
secnnresultf4 <-cbind(as.data.frame(seq(from=5,to=50,by=5)),sectestresultf4)
colnames(secnnresultf4) <- c("Number of hidden neurons","Accuracy of predicting sign(y)")
plot(secnnresultf4,main="Threshold = 0.001 & logistic")
## threshold=0.001 and log perform well in any number of hidden neurons 
## Due to computational resources this model was excluded from the experiemtn as the previous models performance sufficed.



## As there is no difference between different values of hidden neurons, so we use the ideal model from part A

modelb <- neuralnet(positive+negative~secinput, data=sectrainset, hidden=10, threshold = 0.01, act.fct="tanh", stepmax = 10000000,rep=1)

modelb

plot(modelb, rep = "best", intercept=FALSE) 


# Use the testing dataset to test this ideal model with different values of lambda
# Calculate RMSE in different values of lambda to check how they perform
set.seed(1)
sectestlambda <- lapply(seq(from=0.01,to=25.01,by=0.1),function(b){
  sd <- b/(1+abs(x))
  e <- rnorm(1,0,sd)
  xwithe <- sectestset+e
  model <- neuralnet(positive+negative~secinput, data=sectrainset, hidden=20, threshold = 0.01, act.fct="tanh", stepmax = 10000000,rep=1)
  result <- compute(model,xwithe[1])
  seccommodel=NULL
  for (i in 1:250){seccommodel[i]=which.max(result$net.result[i,])}
  for (i in 1:250){if(seccommodel[i]==1){seccommodel[i]="positive"}}
  for (i in 1:250){if(seccommodel[i]==2){seccommodel[i]="negative"}}
  secr <- secvaliset
  secr$output=c(secr$positive>secr$negative)
  for (i in 1:250){if(secr$output[i]=="TRUE"){secr$output[i]="positive"}}
  for (i in 1:250){if(secr$output[i]=="FALSE"){secr$output[i]="negative"}}
  secresult <- table(seccommodel, secr$output)
  probability=(secresult[1,1]+secresult[2,2])/250
}  
)

# visualising the result
seclambdaresult <- do.call(rbind, sectestlambda)
secperform <- cbind(as.data.frame(seq(from=0.01, to=25.01, by=0.1)),seclambdaresult)
colnames(secperform) <- c("Value of lambda","RMSE of model")
plot(secperform,main="The accuracy of the model with errors")



