# load data
promote <- read.csv("~/promoted.csv", header = T)
View(promote)
# resp: Responded or not (1=yes, 0=no) 
# card_tenure: Card Tenure in months 
# risk_score: Risk Score 
# num_promoted: Number of times customer was contacted earlier for the same product 
# avg_bal: Average balance 
# geo_group: Geographical Group (NW W, S, E, SE) 
# res_type: Residence Type (SI=single family, CO=cooperative, CN=condominium, RE=rental, TO=townhouse)

# load the required packages
library(nproc) # version >= 2.0.9
library(glmnet) # version >= 2.0.5
library(randomForest) # version >= 4.6.12
library(e1071) # version >= 1.6.7
# library(plotrix) # version >= 3.6.1
library(parallel)

# clean the data 
summary(promote)
df <- promote[complete.cases(promote),]
nrow(df) # 23815
summary(df) #0.068:1
levels(df[,7]) # 5
levels(df[,8]) # 
data <- df[-which(df[,7]=="" | df[,8]==""),]
n <- nrow(data) #22400

# Dummy Variable
df <- data.frame(E=rep(0,n), N=rep(0,n), SE=rep(0,n), W=rep(0,n),
                 CN=rep(0,n), CO=rep(0,n), RE=rep(0,n), SI=rep(0,n), TO=rep(0,n))
for (i in 1:n) {
    str1 <- as.character(data[i,7])
    df[i,str1] <- 1
    str2 <- as.character(data[i,8])
    df[i,str2] <- 1
}

x = as.matrix(cbind(data[,3:6], df))
y = data[,2]
y = as.matrix(1 - y)
dat=cbind(x,y)
colnames(dat)[14] <- "y"
save(dat, file="processed_data")
load("processed_data")
x = dat[,-14]
y = dat[,14]

# setting 1
split = 10
alpha = 0.1
delta = 0.1
randSeed = 2

# nproc method selection
rf.nproc = nproc(x=x, y=y, method = 'randomforest',  
                 delta = delta, split = split, 
                 randSeed = randSeed,  ntree = 1000, 
                 mtry = 3, cutoff = c(0.2,0.8), 
                 n.cores = detectCores())
save(rf.nproc, file = "rf")

penlog.nproc = nproc(x=x, y=y, method = 'penlog', 
                     delta = delta, split = split, 
                     randSeed = randSeed, 
                     n.cores = detectCores())
save(penlog.nproc, file = "penlog")

svm.nproc = nproc(x = x, y = y, method = 'svm', 
                  delta = delta, split = split,
                  randSeed = randSeed, n.cores = detectCores())
save(svm.nproc, file = "svm")

ada.nproc = nproc(x = x, y = y, method = 'ada', 
                  delta = delta, split = split,
                  randSeed = randSeed, n.cores = detectCores())
save(ada.nproc, file = "ada")

lda.nproc = nproc(x = x, y = y, method = 'lda', 
                  delta = delta, split = split,
                  randSeed = randSeed, n.cores = detectCores())
save(lda.nproc, file = "lda")

nb.nproc = nproc(x = x, y = y, method = 'nb', 
                 delta = delta, split = split,
                 randSeed = randSeed, n.cores = detectCores())
save(nb.nproc, file = "nb")

nnb.nproc = nproc(x = x, y = y, method = 'nnb', 
                  delta = delta, split = split,
                  randSeed = randSeed, n.cores = detectCores())
save(nnb.nproc, file = "nnb")

# Figure 1
rf.col = rgb(red=214, green=31, blue=38, maxColorValue=255)
penlog.col = "black"
svm.col = rgb(red=44, green=123, blue=182, maxColorValue=255)
lda.col = rgb(red=15, green=200, blue=50, maxColorValue=255)
nb.col = rgb(red=150, green=77, blue=150, maxColorValue=255)
nnb.col = rgb(red=50, green=70, blue=200, maxColorValue=255)
ada.col = rgb(red=200, green=200, blue=50, maxColorValue=255)
plot(rf.nproc, col=rf.col, lwd=2, main="NP-ROC Bounds Comparison")
lines(penlog.nproc, col=penlog.col, lwd=2)
lines(svm.nproc, col=svm.col, lwd=2)
lines(lda.nproc, col=lda.col, lwd=2)
lines(nb.nproc, col=nb.col, lwd=2)
lines(nnb.nproc, col=nnb.col, lwd=2)
lines(ada.nproc, col=ada.col, lwd=2)
legend('bottomright', c('random forest','penalized logistic','svm','lda','naive bayes','nonparametric naive bayes','ada'), 
       col=c(rf.col,penlog.col,svm.col,lda.col,nb.col,nnb.col,ada.col), lty=rep(1,7), lwd=rep(2,7), cex=0.9, x.intersp = 0.5,  bty="n")

#figure 2
roc.ada = rocCV(x = x, y = y, method = "ada")
roc.rf = rocCV(x = x, y = y, method = "randomforest")
v = compare(rf.nproc, ada.nproc)
legend('topleft', c('adaboost','random forest'), 
       col=c("red","black"), lty=rep(1,2), cex=0.9, x.intersp = 0.1,  bty="n")
lines(roc.ada$fpr, roc.ada$tpr, type = "l", 
      lty=2, col = "red", lwd = 2)
lines(roc.rf$fpr, roc.rf$tpr, type = "l", 
      lty=2, col = "black", lwd = 2)
save(v, file="ada_vs_rf")

# figure 3
plot(ada.nproc$typeI.u,1-ada.nproc$typeII.u, type = "l", lwd=2,
     ylab = "1-conditional type II error", xlab = "type I error upper bound",
     main = "Roc and NP-Roc for Adaboost")
lines(ada.nproc$typeI.u,1-ada.nproc$typeII.l, lwd=2)
lines(roc.ada$fpr, roc.ada$tpr, type = "l", 
      lty=2, col = "blue", lwd = 2)

# figure 4
aucl = c(rf.nproc$auc.l, penlog.nproc$auc.l, svm.nproc$auc.l, lda.nproc$auc.l, nb.nproc$auc.l, nnb.nproc$auc.l, ada.nproc$auc.l)
aucu = c(rf.nproc$auc.u, penlog.nproc$auc.u, svm.nproc$auc.u, lda.nproc$auc.u, nb.nproc$auc.u, nnb.nproc$auc.u, ada.nproc$auc.u)
names(aucl) = c('rf','pen_log','svm','lda','n_b','n_n_b','ada')
names(aucu) = c('rf','pen_log','svm','lda','n_b','n_n_b','ada')
col1 = rgb(red=50, green=132, blue=191, maxColorValue=255)
col2 = rgb(red=255, green=204, blue=51, maxColorValue=255)
Values <- rbind(aucl,aucu-aucl)
barplot(Values, main = "AUC Plot for NP-ROC Bounds", names.arg = names(aucl),
        col = c(col1, col2), ylim = c(0.45, 0.7), xpd = FALSE)


# fit and predict
set.seed(123)
train = sample(1:n, round(3*n/4))
y.tr = y[train]
x.tr = x[train,]
y.te = y[-train]
x.te = x[-train,]

#ada (delta=0.1): 1 for alpha=0.1, 2 for 0.2 
npc.ada1 <- npc(x = x.tr, y = y.tr, method = 'ada', 
                alpha=0.1, delta = 0.1, split = split)
save(npc.ada1, file="npc.ada1")
npc.ada2 <- npc(x = x.tr, y = y.tr, method = 'ada', 
                alpha=0.2, delta = 0.1, split = split)
save(npc.ada2, file="npc.ada2")
pred.ada1 <- predict(npc.ada1, x.te)$pred.label
pred.ada2 <- predict(npc.ada2, x.te)$pred.label
e1.ada1 = mean(pred.ada1[y.te==0]==1) #0.03865979
e2.ada1 = mean(pred.ada1[y.te==1]==0) #0.8397928
error_rate1 = (sum(pred.ada1[y.te==0]==1)+sum(pred.ada1[y.te==1]==0))/length(y.te)
# 0.7842857
e1.ada2 = mean(pred.ada2[y.te==0]==1) #0.1417526
e2.ada2 = mean(pred.ada2[y.te==1]==0) #0.6266309
error_rate2 = (sum(pred.ada2[y.te==0]==1)+sum(pred.ada2[y.te==1]==0))/length(y.te)
# 0.5930357


# original adaboost
library(ada)
ada = ada::ada(x = x.tr, y = y.tr)
scores =  predict(ada, as.data.frame(x.tr[y.tr==0,]), type="probs")
theta = quantile(scores[,2], 0.8)
pred.s = predict(ada, as.data.frame(x.te), type="probs")
pred.y = as.numeric(pred.s[,2]>=theta)
e1.ada = mean(pred.y[y.te==0]==1) #1
e2.ada = mean(pred.y[y.te==1]==0) #0
error_rate = (sum(pred.y[y.te==0]==1)+sum(pred.ada1[y.te==1]==0))/length(y.te)
# 0.8508929

# analysis
npc.ada <- npc(x = x, y = y, method = 'ada', 
               alpha=0.1, delta = 0.1, split = split)
target <- read.csv("~/target.csv", header = T)
View(target)

tgt <- target[complete.cases(target[,2:7]),]
nrow(tgt) # 104629
summary(tgt) #0.068:1
levels(tgt[,7]) # 5
levels(tgt[,6]) # 
data <- tgt[-which(tgt[,6]=="" | tgt[,7]==""),]
n <- nrow(data) # 98506

# Dummy Variable
df <- data.frame(E=rep(0,n), N=rep(0,n), SE=rep(0,n), W=rep(0,n),
                 CN=rep(0,n), CO=rep(0,n), RE=rep(0,n), SI=rep(0,n), TO=rep(0,n))
for (i in 1:n) {
    str1 <- as.character(data[i,6])
    df[i,str1] <- 1
    str2 <- as.character(data[i,7])
    df[i,str2] <- 1
}

x_in = as.matrix(cbind(data[,2:5], df))
save(x_in, file="processed_target")
load(processed_target)
pred.out <- predict(npc.ada, as.data.frame(x_in))
contact = nrow(x_in)-sum(pred.out$pred.label) #77213, 0.78
