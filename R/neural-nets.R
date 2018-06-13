# Jason Rich
# Due Date: 3-17-18


set.seed(100)
setwd('./')
getwd()







################################################################################
 #install packages (if missing)
################################################################################
pkgs <- installed.packages()[,1]
pkgs.need <- c('caTools','tidyverse','MASS', 'Matrix', 'matrixcalc','RSNNS', 'nnet')
pkgs.missing <- pkgs.need[!pkgs.need %in% pkgs]
if (length(pkgs.missing) > 0){
    install.packages(pkgs.missing, dep = TRUE)
}

library(caTools) # Toolkit for data preprocessing in R
library(tidyverse) # Loads tidyverse
library(MASS) # Modern Applied Statistics with S
library(Matrix)
library(matrixcalc)
library(RSNNS)
library(nnet)
################################################################################
# data pre-processing
################################################################################


# check for data directory, then check for data files. If the data files exit
# remove the files in the data directory.
if(!file.exists('./data/x.*') || !file.exists('./data/y.*')){
        system('rm ./data/x.* ./data/y.*')
}

# create y training dataframe
system('awk "NR%2==0" ./data/regression.tra > ./data/y.trn', intern = TRUE)

y.trn <- read.table('./data/y.trn', header = FALSE, sep = '')
View(y.trn)
str(y.trn)
length(y.trn) #7
summary(y.trn)




# create y test dataframe
system('awk "NR%2==0" ./data/regression.tst > ./data/y.tst', intern = TRUE)
y.tst <- read.table('./data/y.tst', header = FALSE, sep = '')
View(y.tst)
str(y.tst)
length(y.tst) #7
summary(y.tst)



# convert factor to numeric if required
# y.trn$V1 <- as.numeric(levels(y.trn$V1))[y.trn$V1]
# y.trn$V2 <- as.numeric(levels(y.trn$V2))[y.trn$V2]
# y.trn$V3 <- as.numeric(levels(y.trn$V3))[y.trn$V3]
# y.trn$V4 <- as.numeric(levels(y.trn$V4))[y.trn$V4]
# y.trn$V5 <- as.numeric(levels(y.trn$V5))[y.trn$V5]
# y.trn$V6 <- as.numeric(levels(y.trn$V6))[y.trn$V6]
# y.trn$V7 <- as.numeric(levels(y.trn$V7))[y.trn$V7]



################################################################################
# create the feature matrix
################################################################################

# read the file line by line to create features
df <- readLines('./data/regression.tra')
# will receive an error about incomplete final line
# View(df)

# read the file line by line to create test features
df.tst <- readLines('./data/regression.tst')


# parse out the odd rows
parser <- function(x){
    first <- unlist(strsplit(x[1], '\\s{2,}'))
return(first)
}


# apply the parse function to the dataset
x.trn <- sapply(split(df, ceiling(seq_along(df)/2)),parser)
# View(x.trn)

x.trn.df <- data.frame(x.trn, stringsAsFactors = FALSE)
# View(x.trn.df)

# write dataframe to disk
write.table(x.trn.df, './data/x.trn', quote = FALSE, sep = ' ', row.names =  FALSE, col.names = FALSE)

# read in new x dataframe
x.trn.new  <- read.table('./data/x.trn', header = FALSE, sep = ' ')
View(x.trn.new)
str(x.trn.new)

# test x dataframe
x.tst <- sapply(split(df.tst, ceiling(seq_along(df.tst)/2)),parser)
x.tst.df <- data.frame(x.tst, stringsAsFactors = FALSE)
write.table(x.tst.df, './data/x.tst', quote = FALSE, sep = ' ', row.names =  FALSE, col.names = FALSE)

# read in new x dataframe
x.tst.new  <- read.table('./data/x.tst', header = FALSE, sep = ' ')
View(x.tst.new)
str(x.tst.new)




################################################################################
# Three Layer Network
################################################################################

# Task 1: Train a neural network with one hidden layer based on the training
# dataset “regression.tra”, which has 8 inputs and 7 outputs. Apply the trained
# network on the testing data “regression.tst”. Let Nh, the number of hidden
# units in the network, be 1, 4 and 8 respectively, and obtain training and
# testing errors.


################################################################################
# Training network
################################################################################
# training feature matrix
x.mat <- as.matrix(x.trn.new)
# View(x.mat)

# training response matrix
y.mat <- as.matrix(y.trn)
# View(y.mat)

# neural network
# nnet function help
# ?nnet

nnFit <- nnet(x.mat, y.mat, size = 1, linout = TRUE)
# weights:  23
# initial  value 24846.837242
# final  value 1814.607274
# converged

nnfit
# a 8-1-7 network with 23 weights
# options were - linear output units

# test feature matrix
x.mat <- as.matrix(x.tst.new)
# test response matrix
y.mat <- as.matrix(y.tst)

# testing model hidden layers


# test data with 1, 4, 8 hidden layers
# 1 hidden layer
nnFit.1 <- nnet(x.mat, y.mat, size = 1, linout = TRUE)

# weights:  23
# initial  value 29357.893447
# final  value 1107.880594
# converged


# 4 hidden layers
nnFit.4 <- nnet(x.mat, y.mat, size = 4, linout = TRUE)

# weights:  71
# initial  value 12871.803244
# iter  10 value 1106.166043
# iter  20 value 922.770314
# iter  30 value 840.011394
# iter  40 value 715.202185
# iter  50 value 656.060758
# iter  60 value 637.283953
# iter  70 value 631.687682
# iter  80 value 630.885494
# iter  90 value 630.762267
# iter 100 value 630.602610
# final  value 630.602610
# stopped after 100 iterations
# a 8-4-7 network with 71 weights
# options were - linear output units


# 8 hidden layers
nnFit.8 <- nnet(x.mat, y.mat, size = 8, linout = TRUE)

# initial  value 11620.054255
# iter  10 value 1056.787146
# iter  20 value 957.737340
# iter  30 value 842.752599
# iter  40 value 751.701143
# iter  50 value 699.934546
# iter  60 value 575.273005
# iter  70 value 485.100183
# iter  80 value 427.350548
# iter  90 value 392.526924
# iter 100 value 384.890486
# final  value 384.890486
# stopped after 100 iterations
# a 8-8-7 network with 135 weights
# options were - linear output units

################################################################################
# Two Class Data
################################################################################

# Task 2: Design a three layer neural network for classification. Train the
# network on the training data classification.tra, which has 2 inputs and 2
# classes.  Apply the trained network on the testing data classification.tst.
# Let Nh, the number of hidden units in the network, be 1, 2 and 4 respectively,
# obtain training and testing classification accuracies.


 # read in the generated training data
trn <-  read.table('./data/train.txt', sep = ' ')
# View(trn)


colnames(trn)<- c('ftr1', 'ftr2', 'rsp')
# View(trn)

# create the feature dataframe
x.trn <- trn[,-3]
# View(x.trn)
# str(x.trn)

# q: are the features the same size?
# a: yes; 400
# length(x.trn[,1]) # 400
# length(x.trn[,2]) # 400

# create the response dataframe
y.trn <- trn[3]
# View(y.trn)
# str(y.trn)


# Multilayer Perceptron for Classification
# MLP help page
# ?mlp

# 3 hidden layers
mlpFit <- mlp(x.trn, y.trn, size = 1)
summary(mlpFit)
# SNNS network definition file V1.4-3D
# generated at Thu Apr  5 21:17:08 2018
#
# network name : RSNNS_untitled
# source files :
# no. of units : 4
# no. of connections : 3
# no. of unit types : 0
# no. of site types : 0
#
#
# learning function : Std_Backpropagation
# update function   : Topological_Order
#
#
# unit default section :
#
# act      | bias     | st | subnet | layer | act func     | out func
# ---------|----------|----|--------|-------|--------------|-------------
#  0.00000 |  0.00000 | i  |      0 |     1 | Act_Logistic | Out_Identity
# ---------|----------|----|--------|-------|--------------|-------------
#
#
#     unit definition section :
#
# no. | typeName | unitName   | act      | bias     | st | position | act func     | out func | sites
# ----|----------|------------|----------|----------|----|----------|--------------|----------|-------
#   1 |          | Input_ftr1 |  2.82523 |  0.10059 | i  | 1,0,0    | Act_Identity |          |
#   2 |          | Input_ftr2 | -2.16075 | -0.28343 | i  | 2,0,0    | Act_Identity |          |
#   3 |          | Hidden_2_1 |  0.99409 | -4.10186 | h  | 1,2,0    |||
#   4 |          | Output_rsp |  0.91737 | -2.33788 | o  | 1,4,0    |||
# ----|----------|------------|----------|----------|----|----------|--------------|----------|-------
#
#
# connection definition section :
#
# target | site | source:weight
# -------|------|------------------------------------------------------------------------------------------------------------
# 3 |    | 2:-0.00161, 1: 3.26454
# 4 |    | 3: 4.77330
# -------|------|------------------------------------------------------------------------------------------------------------

weightMatrix(mlpFit)
# Input_ftr1 Input_ftr2  Hidden_2_1 Output_rsp
# Input_ftr1          0          0  3.26454425   0.000000
# Input_ftr2          0          0 -0.00160933   0.000000
# Hidden_2_1          0          0  0.00000000   4.773295
# Output_rsp          0          0  0.00000000   0.000000


# read in the generated training data
tst <-  read.table('./data/test.txt', sep = ' ')
# View(tst)


colnames(tst)<- c('ftr1', 'ftr2', 'rsp')
# View(tst)

# create the feature dataframe
x.tst <- tst[,-3]
# View(x.tst)
# str(x.tst)


# create the response dataframe
y.tst <- tst[3]
# View(y.tst)
# str(y.tst)

# test mlp 1 hidden layer
mlpFit.1 <- mlp(x.tst, y.tst, size = 1)
extractNetInfo(mlpFit.1)
# $infoHeader
#                 name               value
# 1       no. of units                   4
# 2 no. of connections                   3
# 3  no. of unit types                   0
# 4  no. of site types                   0
# 5 learning function Std_Backpropagation
# 6 update function   Topological_Order
#
# $unitDefinitions
# unitNo   unitName    unitAct    unitBias        type posX posY posZ      actFunc      outFunc sites
# 1      1 Input_ftr1 1.53466904 -0.17774597  UNIT_INPUT    1    0    0 Act_Identity Out_Identity
# 2      2 Input_ftr2 4.82125711 -0.09393275  UNIT_INPUT    2    0    0 Act_Identity Out_Identity
# 3      3 Hidden_2_1 0.09488393  4.41821146 UNIT_HIDDEN    1    2    0 Act_Logistic Out_Identity
# 4      4 Output_rsp 0.85789084  2.25291371 UNIT_OUTPUT    1    4    0 Act_Logistic Out_Identity
#
# $fullWeightMatrix
# Input_ftr1 Input_ftr2 Hidden_2_1 Output_rsp
# Input_ftr1          0          0 -3.5152020   0.000000
# Input_ftr2          0          0 -0.2652728   0.000000
# Hidden_2_1          0          0  0.0000000  -4.795671
# Output_rsp          0          0  0.0000000   0.000000

# test mlp 2 hidden layer
mlpFit.2 <- mlp(x.tst, y.tst, size = 2)
extractNetInfo(mlpFit.2)

# $infoHeader
#                 name               value
# 1       no. of units                   5
# 2 no. of connections                   6
# 3  no. of unit types                   0
# 4  no. of site types                   0
# 5  learning function Std_Backpropagation
# 6    update function   Topological_Order
#
# $unitDefinitions
# unitNo   unitName   unitAct    unitBias        type posX posY posZ      actFunc      outFunc sites
# 1      1 Input_ftr1 1.5346690  0.14532027  UNIT_INPUT    1    0    0 Act_Identity Out_Identity
# 2      2 Input_ftr2 4.8212571  0.28536153  UNIT_INPUT    2    0    0 Act_Identity Out_Identity
# 3      3 Hidden_2_1 0.9848601 -1.85099852 UNIT_HIDDEN    1    2    0 Act_Logistic Out_Identity
# 4      4 Hidden_2_2 0.5300117  4.96654701 UNIT_HIDDEN    2    2    0 Act_Logistic Out_Identity
# 5      5 Output_rsp 0.8723227  0.01731124 UNIT_OUTPUT    1    4    0 Act_Logistic Out_Identity
#
# $fullWeightMatrix
# Input_ftr1 Input_ftr2 Hidden_2_1 Hidden_2_2 Output_rsp
# Input_ftr1          0          0  2.0661259 -3.5085435   0.000000
# Input_ftr2          0          0  0.5922407  0.1116093   0.000000
# Hidden_2_1          0          0  0.0000000  0.0000000   4.015714
# Hidden_2_2          0          0  0.0000000  0.0000000  -3.868923
# Output_rsp          0          0  0.0000000  0.0000000   0.000000

# test mlp 4 hidden layer
mlpFit.4 <- mlp(x.tst, y.tst, size = 4)
extractNetInfo(mlpFit.4)

# $infoHeader
#                 name               value
# 1       no. of units                   7
# 2 no. of connections                  12
# 3  no. of unit types                   0
# 4  no. of site types                   0
# 5  learning function Std_Backpropagation
# 6    update function   Topological_Order
#
# $unitDefinitions
# unitNo   unitName    unitAct    unitBias        type posX posY posZ      actFunc      outFunc sites
# 1      1 Input_ftr1 1.53466904  0.16754857  UNIT_INPUT    1    0    0 Act_Identity Out_Identity
# 2      2 Input_ftr2 4.82125711 -0.07640423  UNIT_INPUT    2    0    0 Act_Identity Out_Identity
# 3      3 Hidden_2_1 0.97571188 -1.61111474 UNIT_HIDDEN    1    2    0 Act_Logistic Out_Identity
# 4      4 Hidden_2_2 0.68408781  2.07331133 UNIT_HIDDEN    2    2    0 Act_Logistic Out_Identity
# 5      5 Hidden_2_3 0.03354631  1.46115279 UNIT_HIDDEN    3    2    0 Act_Logistic Out_Identity
# 6      6 Hidden_2_4 0.03141143  1.23949301 UNIT_HIDDEN    4    2    0 Act_Logistic Out_Identity
# 7      7 Output_rsp 0.85914159  1.08728540 UNIT_OUTPUT    1    4    0 Act_Logistic Out_Identity
#
# $fullWeightMatrix
# Input_ftr1 Input_ftr2 Hidden_2_1 Hidden_2_2 Hidden_2_3 Hidden_2_4 Output_rsp
# Input_ftr1          0          0  1.8766909 -2.4391391 -1.7096649 -1.6513710   0.000000
# Input_ftr2          0          0  0.5028137  0.5066276 -0.4559163 -0.4425927   0.000000
# Hidden_2_1          0          0  0.0000000  0.0000000  0.0000000  0.0000000   2.798597
# Hidden_2_2          0          0  0.0000000  0.0000000  0.0000000  0.0000000  -2.739666
# Hidden_2_3          0          0  0.0000000  0.0000000  0.0000000  0.0000000  -2.090804
# Hidden_2_4          0          0  0.0000000  0.0000000  0.0000000  0.0000000  -2.082677
# Output_rsp          0          0  0.0000000  0.0000000  0.0000000  0.0000000   0.000000




################################################################################
# Zip Code Data
################################################################################
# Task 3: Repeat Task 2 on the training data zipcode.tra, which has 16 inputs
# and 10 classes and test the trained network on the testing data zipcode.tst.
# Let Nh, the number of hidden units in the MLP, be 5, 10 and 13 respectively,
# obtain training and testing classification accuracies.

# ?read.csv
zip.trn <- read.csv('./data/Valid_ZC_train_Data.csv', header = FALSE, sep = ',')
# View(zip.trn)

colnames(zip.trn) <- c('ftr1','ftr2','ftr3','ftr4','ftr5','ftr6','ftr7','ftr8','ftr9'
                       ,'ftr10','ftr11','ftr12','ftr13','ftr14','ftr15','ftr16','rsp')


zip.x.trn <- zip.trn[, -17]
zip.y.trn <- zip.trn[17]

zip.x.mat <- as.matrix(zip.x.trn)
zip.y.mat <- as.matrix(zip.y.trn)

# Training MLP
mlpFitZip <- mlp(zip.x.mat, zip.y.mat, size = 1)
extractNetInfo(mlpFitZip)

# $infoHeader
#                 name               value
# 1       no. of units                  18
# 2 no. of connections                  17
# 3  no. of unit types                   0
# 4  no. of site types                   0
# 5  learning function Std_Backpropagation
# 6    update function   Topological_Order

# $unitDefinitions
# unitNo    unitName    unitAct    unitBias        type posX posY posZ      actFunc      outFunc
# 1       1  Input_ftr1  1.0000000 -0.01744974  UNIT_INPUT    1    0    0 Act_Identity Out_Identity
# 2       2  Input_ftr2  1.0000000  0.25804460  UNIT_INPUT    2    0    0 Act_Identity Out_Identity
# 3       3  Input_ftr3  0.0000000  0.13042754  UNIT_INPUT    3    0    0 Act_Identity Out_Identity
# 4       4  Input_ftr4  1.0000000  0.17468759  UNIT_INPUT    4    0    0 Act_Identity Out_Identity
# 5       5  Input_ftr5  0.0000000  0.02608570  UNIT_INPUT    5    0    0 Act_Identity Out_Identity
# 6       6  Input_ftr6  1.0000000  0.12153730  UNIT_INPUT    6    0    0 Act_Identity Out_Identity
# 7       7  Input_ftr7  0.0000000 -0.07433514  UNIT_INPUT    7    0    0 Act_Identity Out_Identity
# 8       8  Input_ftr8  1.0000000  0.12509498  UNIT_INPUT    8    0    0 Act_Identity Out_Identity
# 9       9  Input_ftr9 10.0000000 -0.16758792  UNIT_INPUT    9    0    0 Act_Identity Out_Identity
# 10     10 Input_ftr10  0.0000000  0.18408278  UNIT_INPUT   10    0    0 Act_Identity Out_Identity
# 11     11 Input_ftr11  0.7500000 -0.11424300  UNIT_INPUT   11    0    0 Act_Identity Out_Identity
# 12     12 Input_ftr12  2.0000000 -0.25217623  UNIT_INPUT   12    0    0 Act_Identity Out_Identity
# 13     13 Input_ftr13  0.0000000  0.26646703  UNIT_INPUT   13    0    0 Act_Identity Out_Identity
# 14     14 Input_ftr14  2.6666670  0.03866094  UNIT_INPUT   14    0    0 Act_Identity Out_Identity
# 15     15 Input_ftr15  2.5833330 -0.03890762  UNIT_INPUT   15    0    0 Act_Identity Out_Identity
# 16     16 Input_ftr16  0.7241379 -0.19695276  UNIT_INPUT   16    0    0 Act_Identity Out_Identity
# 17     17  Hidden_2_1  0.9999283  0.42357603 UNIT_HIDDEN    1    2    0 Act_Logistic Out_Identity
# 18     18  Output_rsp  0.9999982  7.32148600 UNIT_OUTPUT    1    4    0 Act_Logistic Out_Identity



zip.tst <- read.csv('./data/Valid_ZC_test_Data.csv', header = FALSE, sep = ',')
# View(zip.tst)

colnames(zip.tst) <- c('ftr1','ftr2','ftr3','ftr4','ftr5','ftr6','ftr7','ftr8','ftr9'
                       ,'ftr10','ftr11','ftr12','ftr13','ftr14','ftr15','ftr16','rsp')


zip.x.tst <- zip.tst[, -17]
zip.y.tst <- zip.tst[17]

zip.x.mat <- as.matrix(zip.x.tst)
zip.y.mat <- as.matrix(zip.y.tst)

# Test MLP 5 hidden layers
mlpFitZip.5 <- mlp(zip.x.mat, zip.y.mat, size = 5)
extractNetInfo(mlpFitZip.5)
# $infoHeader
#                 name               value
# 1       no. of units                  22
# 2 no. of connections                  85
# 3  no. of unit types                   0
# 4  no. of site types                   0
# 5  learning function Std_Backpropagation
# 6    update function   Topological_Order
#
# $unitDefinitions
# unitNo    unitName    unitAct     unitBias        type posX posY posZ      actFunc      outFunc sites
# 1       1  Input_ftr1  1.0000000  0.009358346  UNIT_INPUT    1    0    0 Act_Identity Out_Identity
# 2       2  Input_ftr2  1.0000000  0.233026743  UNIT_INPUT    2    0    0 Act_Identity Out_Identity
# 3       3  Input_ftr3  0.0000000  0.103718340  UNIT_INPUT    3    0    0 Act_Identity Out_Identity
# 4       4  Input_ftr4  1.0000000 -0.137541354  UNIT_INPUT    4    0    0 Act_Identity Out_Identity
# 5       5  Input_ftr5  0.0000000  0.194667071  UNIT_INPUT    5    0    0 Act_Identity Out_Identity
# 6       6  Input_ftr6  1.0000000 -0.155095607  UNIT_INPUT    6    0    0 Act_Identity Out_Identity
# 7       7  Input_ftr7  0.0000000  0.269827008  UNIT_INPUT    7    0    0 Act_Identity Out_Identity
# 8       8  Input_ftr8  1.0000000 -0.028807998  UNIT_INPUT    8    0    0 Act_Identity Out_Identity
# 9       9  Input_ftr9 10.5000000  0.155121535  UNIT_INPUT    9    0    0 Act_Identity Out_Identity
# 10     10 Input_ftr10  0.0000000 -0.280220091  UNIT_INPUT   10    0    0 Act_Identity Out_Identity
# 11     11 Input_ftr11  0.7368421  0.124357492  UNIT_INPUT   11    0    0 Act_Identity Out_Identity
# 12     12 Input_ftr12  1.0000000  0.054758221  UNIT_INPUT   12    0    0 Act_Identity Out_Identity
# 13     13 Input_ftr13  1.7900000  0.093493491  UNIT_INPUT   13    0    0 Act_Identity Out_Identity
# 14     14 Input_ftr14  1.3076921  0.156245291  UNIT_INPUT   14    0    0 Act_Identity Out_Identity
# 15     15 Input_ftr15  2.7647059  0.045459092  UNIT_INPUT   15    0    0 Act_Identity Out_Identity
# 16     16 Input_ftr16  1.2413790 -0.051479995  UNIT_INPUT   16    0    0 Act_Identity Out_Identity
# 17     17  Hidden_2_1  0.9980718  0.201067358 UNIT_HIDDEN    1    2    0 Act_Logistic Out_Identity
# 18     18  Hidden_2_2  0.9989157  0.161342621 UNIT_HIDDEN    2    2    0 Act_Logistic Out_Identity
# 19     19  Hidden_2_3  0.9939985  0.248922944 UNIT_HIDDEN    3    2    0 Act_Logistic Out_Identity
# 20     20  Hidden_2_4  0.9981739  0.066938601 UNIT_HIDDEN    4    2    0 Act_Logistic Out_Identity
# 21     21  Hidden_2_5  0.9989969  0.282944411 UNIT_HIDDEN    5    2    0 Act_Logistic Out_Identity
# 22     22  Output_rsp  0.9999995  3.429463387 UNIT_OUTPUT    1    4    0 Act_Logistic Out_Identity


# Test MLP 10 hidden layers
mlpFitZip.10 <- mlp(zip.x.mat, zip.y.mat, size = 10)
extractNetInfo(mlpFitZip.10)

# $infoHeader
#                 name               value
# 1       no. of units                  27
# 2 no. of connections                 170
# 3  no. of unit types                   0
# 4  no. of site types                   0
# 5  learning function Std_Backpropagation
# 6    update function   Topological_Order
#
# $unitDefinitions
# unitNo    unitName    unitAct     unitBias        type posX posY posZ      actFunc      outFunc sites
# 1       1  Input_ftr1  1.0000000 -0.103213102  UNIT_INPUT    1    0    0 Act_Identity Out_Identity
# 2       2  Input_ftr2  1.0000000 -0.266847163  UNIT_INPUT    2    0    0 Act_Identity Out_Identity
# 3       3  Input_ftr3  0.0000000 -0.171601623  UNIT_INPUT    3    0    0 Act_Identity Out_Identity
# 4       4  Input_ftr4  1.0000000  0.160543382  UNIT_INPUT    4    0    0 Act_Identity Out_Identity
# 5       5  Input_ftr5  0.0000000  0.125508100  UNIT_INPUT    5    0    0 Act_Identity Out_Identity
# 6       6  Input_ftr6  1.0000000  0.128460974  UNIT_INPUT    6    0    0 Act_Identity Out_Identity
# 7       7  Input_ftr7  0.0000000  0.267045438  UNIT_INPUT    7    0    0 Act_Identity Out_Identity
# 8       8  Input_ftr8  1.0000000 -0.047296524  UNIT_INPUT    8    0    0 Act_Identity Out_Identity
# 9       9  Input_ftr9 10.5000000 -0.214128703  UNIT_INPUT    9    0    0 Act_Identity Out_Identity
# 10     10 Input_ftr10  0.0000000 -0.262436897  UNIT_INPUT   10    0    0 Act_Identity Out_Identity
# 11     11 Input_ftr11  0.7368421  0.299100637  UNIT_INPUT   11    0    0 Act_Identity Out_Identity
# 12     12 Input_ftr12  1.0000000 -0.083453670  UNIT_INPUT   12    0    0 Act_Identity Out_Identity
# 13     13 Input_ftr13  1.7900000  0.264654458  UNIT_INPUT   13    0    0 Act_Identity Out_Identity
# 14     14 Input_ftr14  1.3076921  0.011924773  UNIT_INPUT   14    0    0 Act_Identity Out_Identity
# 15     15 Input_ftr15  2.7647059 -0.299008548  UNIT_INPUT   15    0    0 Act_Identity Out_Identity
# 16     16 Input_ftr16  1.2413790 -0.188131630  UNIT_INPUT   16    0    0 Act_Identity Out_Identity
# 17     17  Hidden_2_1  0.9847873  0.048922565 UNIT_HIDDEN    1    2    0 Act_Logistic Out_Identity
# 18     18  Hidden_2_2  0.9948030  0.322825044 UNIT_HIDDEN    2    2    0 Act_Logistic Out_Identity
# 19     19  Hidden_2_3  0.9998503 -0.004453308 UNIT_HIDDEN    3    2    0 Act_Logistic Out_Identity
# 20     20  Hidden_2_4  0.9984032  0.397551775 UNIT_HIDDEN    4    2    0 Act_Logistic Out_Identity
# 21     21  Hidden_2_5  0.9703342 -0.092358246 UNIT_HIDDEN    5    2    0 Act_Logistic Out_Identity
# 22     22  Hidden_2_6  0.9882665  0.247353345 UNIT_HIDDEN    6    2    0 Act_Logistic Out_Identity
# 23     23  Hidden_2_7  0.9972405  0.169709384 UNIT_HIDDEN    7    2    0 Act_Logistic Out_Identity
# 24     24  Hidden_2_8  0.8154095  0.368596643 UNIT_HIDDEN    8    2    0 Act_Logistic Out_Identity
# 25     25  Hidden_2_9  0.9791656  0.388557225 UNIT_HIDDEN    9    2    0 Act_Logistic Out_Identity
# 26     26 Hidden_2_10  0.4318390 -0.094534963 UNIT_HIDDEN   10    2    0 Act_Logistic Out_Identity
# 27     27  Output_rsp  0.9999998  2.303434610 UNIT_OUTPUT    1    4    0 Act_Logistic Out_Identity

# Test MLP 13 hidden layers
mlpFitZip.13 <- mlp(zip.x.mat, zip.y.mat, size = 13)
extractNetInfo(mlpFitZip.13)
# $infoHeader
#                 name               value
# 1       no. of units                  30
# 2 no. of connections                 221
# 3  no. of unit types                   0
# 4  no. of site types                   0
# 5  learning function Std_Backpropagation
# 6    update function   Topological_Order
#
# $unitDefinitions
# unitNo    unitName     unitAct    unitBias        type posX posY posZ      actFunc      outFunc sites
# 1       1  Input_ftr1  1.00000000 -0.14415908  UNIT_INPUT    1    0    0 Act_Identity Out_Identity
# 2       2  Input_ftr2  1.00000000 -0.27137744  UNIT_INPUT    2    0    0 Act_Identity Out_Identity
# 3       3  Input_ftr3  0.00000000  0.15364906  UNIT_INPUT    3    0    0 Act_Identity Out_Identity
# 4       4  Input_ftr4  1.00000000 -0.22735918  UNIT_INPUT    4    0    0 Act_Identity Out_Identity
# 5       5  Input_ftr5  0.00000000 -0.04581288  UNIT_INPUT    5    0    0 Act_Identity Out_Identity
# 6       6  Input_ftr6  1.00000000  0.11647880  UNIT_INPUT    6    0    0 Act_Identity Out_Identity
# 7       7  Input_ftr7  0.00000000  0.03855786  UNIT_INPUT    7    0    0 Act_Identity Out_Identity
# 8       8  Input_ftr8  1.00000000 -0.25902843  UNIT_INPUT    8    0    0 Act_Identity Out_Identity
# 9       9  Input_ftr9 10.50000000  0.26665258  UNIT_INPUT    9    0    0 Act_Identity Out_Identity
# 10     10 Input_ftr10  0.00000000 -0.12258029  UNIT_INPUT   10    0    0 Act_Identity Out_Identity
# 11     11 Input_ftr11  0.73684210  0.15258712  UNIT_INPUT   11    0    0 Act_Identity Out_Identity
# 12     12 Input_ftr12  1.00000000 -0.17591724  UNIT_INPUT   12    0    0 Act_Identity Out_Identity
# 13     13 Input_ftr13  1.78999996  0.05504215  UNIT_INPUT   13    0    0 Act_Identity Out_Identity
# 14     14 Input_ftr14  1.30769205 -0.23977938  UNIT_INPUT   14    0    0 Act_Identity Out_Identity
# 15     15 Input_ftr15  2.76470590 -0.06103811  UNIT_INPUT   15    0    0 Act_Identity Out_Identity
# 16     16 Input_ftr16  1.24137902  0.14330485  UNIT_INPUT   16    0    0 Act_Identity Out_Identity
# 17     17  Hidden_2_1  0.98455036  0.34058833 UNIT_HIDDEN    1    2    0 Act_Logistic Out_Identity
# 18     18  Hidden_2_2  0.99281728  0.15905468 UNIT_HIDDEN    2    2    0 Act_Logistic Out_Identity
# 19     19  Hidden_2_3  0.03069293  0.13908266 UNIT_HIDDEN    3    2    0 Act_Logistic Out_Identity
# 20     20  Hidden_2_4  0.99817562 -0.13171349 UNIT_HIDDEN    4    2    0 Act_Logistic Out_Identity
# 21     21  Hidden_2_5  0.99745661  0.38329938 UNIT_HIDDEN    5    2    0 Act_Logistic Out_Identity
# 22     22  Hidden_2_6  0.99783081  0.37005305 UNIT_HIDDEN    6    2    0 Act_Logistic Out_Identity
# 23     23  Hidden_2_7  0.01553688 -0.20507269 UNIT_HIDDEN    7    2    0 Act_Logistic Out_Identity
# 24     24  Hidden_2_8  0.91833979 -0.01591811 UNIT_HIDDEN    8    2    0 Act_Logistic Out_Identity
# 25     25  Hidden_2_9  0.99637955  0.37403962 UNIT_HIDDEN    9    2    0 Act_Logistic Out_Identity
# 26     26 Hidden_2_10  0.96162206 -0.07846814 UNIT_HIDDEN   10    2    0 Act_Logistic Out_Identity
# 27     27 Hidden_2_11  0.98612267  0.35485226 UNIT_HIDDEN   11    2    0 Act_Logistic Out_Identity
# 28     28 Hidden_2_12  0.86011481  0.24662767 UNIT_HIDDEN   12    2    0 Act_Logistic Out_Identity
# 29     29 Hidden_2_13  0.99759078  0.17426412 UNIT_HIDDEN   13    2    0 Act_Logistic Out_Identity
# 30     30  Output_rsp  0.99999982  1.89401495 UNIT_OUTPUT    1    4    0 Act_Logistic Out_Identity

