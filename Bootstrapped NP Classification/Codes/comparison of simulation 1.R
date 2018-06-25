## Simulation 1

# As this dataset doesn't require training,
# we can directly chose our threshold

alpha <- 0.05
delta <- 0.05
B <- 500

# Set up parallel 
library(doParallel)
cl <- makeCluster(8)
registerDoParallel(cl)

np_order_stat <- function(size, alpha, delta) {
    violation_rates <- pbinom(q=0:(size-1), size=size, prob=1-alpha, lower.tail=F)
    return( which(violation_rates <= delta)[1] )
}

np_threshold <- function(S0, alpha, delta) {
    np_order <- np_order_stat(size=length(S0), alpha=alpha, delta=delta)
    return( sort(S0)[np_order] )
}

start_time <- Sys.time()

thres <- foreach(i = 1:1000, .combine = "rbind")%dopar%{
    
    ### Set up the data ###
    set.seed(i)
    X0 <- rnorm(1000)
    
    ### Pick up the threshold ###
    # candidate threshold
    candi_thre <- sort(X0)
    # get the typeI error of each candidate threshold using bootstrap
    typeI <- replicate(B, {
        boot.ix <- sample(length(X0), replace = T) 
        boot.X <- X0[boot.ix]
        typeI <- vapply(candi_thre, function(x) mean(boot.X > x), numeric(1))
    })
    # get the violation rate for each candidate threshold
    vio_rate <- apply(typeI, 1, function(x) mean(x > alpha))
    # select the threshold satisfying violation rate
    boot_thre <- candi_thre[which(vio_rate<delta)[1]]
    
    ### np_threshold ###
    np_thre <- np_threshold(X0, alpha, delta)
    
    ### Naive classical threshold ###
    nai_thre <- quantile(X0, probs=1-alpha)
    
    ### Collect all three threshold
    thres <- c(nai_thre, boot_thre, np_thre)
}

stopCluster(cl)

names(thres) <- c("naive", "boot", "np")

### Get the true typeI error for the chosen threshold
typeI_true <- matrix(1-pnorm(thres), nrow = nrow(thres))

### Get the violation rate for each method

nai_vio <- mean(typeI_true[ , 1] > alpha)
boot_vio <- mean(typeI_true[ , 2] > alpha)
np_vio <- mean(typeI_true[ , 3] > alpha)

### plot ###

## plot the true type I and II errors for each of the 1000 classical classifiers
# oracle ROC
alphas <- c(0.001, 0.005, seq(0.01, 0.99, by=0.01))
plot(x=alphas, y=pnorm(qnorm(alphas, mean=0, sd=1, lower.tail=FALSE), mean=2, sd=1, lower.tail=FALSE), 
     type="l", xlab="type I error", ylab="1 - type II error", 
     lwd=2,  xlim=c(0,0.5), ylim=c(0.3,1), main="naive classical classifiers")
abline(v = alpha, lty=2, lwd=2)

x <- pnorm(thres[ ,1], mean=0, sd=1, lower.tail=FALSE)
y <- pnorm(thres[ ,1], mean=2, sd=1, lower.tail=FALSE)
points(x=x[which(x<=alpha)], y=y[which(x<=alpha)], 
       col=rgb(red=139, green=26,  blue=26, alpha=120, maxColorValue=255), cex=1, pch=4, lwd=2 )
points(x=x[which(x>alpha)], y=y[which(x>alpha)], 
        col=rgb(red=255, green=48, blue=48, alpha=120, maxColorValue=255), cex=1, pch=4, lwd=2 )
legend( "bottomright", "naive classical classifiers", pch=4, cex=1, bty="n", 
        col=rgb(red=205, green=38, blue=38, alpha=255, maxColorValue=255) )
text(x = 0.3, y= 0.6, labels = paste("Type I error violation rate:", nai_vio),
     col=rgb(red=205, green=38, blue=38, alpha=255, maxColorValue=255))

## plot the true type I and II errors for each of the 1000 bootstrap classifiers
## oracle ROC
alphas <- c(0.001, 0.005, seq(0.01, 0.99, by=0.01))
plot(x=alphas, y=pnorm(qnorm(alphas, mean=0, sd=1, lower.tail=FALSE), mean=2, sd=1, lower.tail=FALSE), 
     type="l", xlab="type I error", ylab="1 - type II error", 
     lwd=2,  xlim=c(0,0.5), ylim=c(0.3,1), main="bootstrap classifiers")
abline(v = alpha, lty=2, lwd=2)

x <- pnorm(thres[ ,2], mean=0, sd=1, lower.tail=FALSE)
y <- pnorm(thres[ ,2], mean=2, sd=1, lower.tail=FALSE)
points( x=x[which(x<=alpha)], y=y[which(x<=alpha)], 
        col=rgb(red=0, green=134,   blue=139, alpha=120, maxColorValue=255), cex=1, pch=4, lwd=2 )
points( x=x[which(x>alpha)], y=y[which(x>alpha)], 
        col=rgb(red=0, green=245,     blue=255, alpha=120, maxColorValue=255), cex=1, pch=4, lwd=2 )

legend("bottomright", "bootstrap classifiers", pch=4, cex=1, bty="n", 
        col=rgb(red=0, green=245, blue=255, alpha=255, maxColorValue=255))
text(x = 0.3, y= 0.6, labels = paste("Type I error violation rate:", boot_vio),
     col=rgb(red=0, green=245, blue=255, alpha=255, maxColorValue=255))


## oracle ROC
alphas <- c(0.001, 0.005, seq(0.01, 0.99, by=0.01))
plot(x=alphas, y=pnorm(qnorm(alphas, mean=0, sd=1, lower.tail=FALSE), mean=2, sd=1, lower.tail=FALSE), 
     type="l", xlab="type I error", ylab="1 - type II error", 
     lwd=2,  xlim=c(0,0.5), ylim=c(0.3,1), main="np classifiers")
abline(v = alpha, lty=2, lwd=2)
## plot the true type I and II errors for each of the 1000 NP classifiers
x <- pnorm(thres[ ,3], mean=0, sd=1, lower.tail=FALSE)
y <- pnorm(thres[ ,3], mean=2, sd=1, lower.tail=FALSE)
points( x=x[which(x<alpha)], y=y[which(x<alpha)], 
        col=rgb(red=39, green=64,     blue=139, alpha=120, maxColorValue=255), cex=1, pch=4, lwd=2 )
points( x=x[which(x>=alpha)], y=y[which(x>=alpha)], 
        col=rgb(red=72, green=118,  blue=255, alpha=120, maxColorValue=255), cex=1, pch=4, lwd=2 )

legend( "bottomright", "NP classifiers", pch=4, cex=1, bty="n", 
        col=rgb(red=39, green=64,   blue=139, alpha=255, maxColorValue=255))
text(x = 0.3, y= 0.6, labels = paste("Type I error violation rate:", np_vio),
     col=rgb(red=39, green=64,   blue=139, alpha=255, maxColorValue=255))

end_time <- Sys.time()
print(end_time - start_time)


################bootstrap 1#################
my_temp <- typeI[959, ]
hist(my_temp,breaks = 40, main = "Histogram of the Empirical Type I Error for a Classifier",
     xlab = "Empirical Type I Error")
abline(v= 0.05, lty = "dashed", col = "red")

### bootstrap 2####
candi_thre[955: 964]
plot(density(typeI[955, ]), xlim = c(0.015, 0.07), ylim = c(0, 65),
     main = "Density Plot of the Empirical Type I Error for 10 Candidate Thresholds")
for(i in 1: 9){
    lines(density(typeI[955+i, ]))
}

### boostrap 3###

###########intitial bootstrap#########
library(doParallel)
cl <- makeCluster(8)
registerDoParallel(cl)

typeI <- foreach(i = 1:1000, .combine = cbind) %dopar% {
    set.seed(i)
    X0 <- rnorm(1000)
    # threshold candidate in each bootstrap 
    threshold <- replicate(1000, {
        boot.ix <- sample(length(X0), replace = T)
        boot.X <- X0[boot.ix]
        threshold <- quantile(boot.X, 1-alpha)
    })
    boot.threshold <- quantile(threshold, 1-delta)
    typeI <- 1-pnorm(boot.threshold)
}
stopCluster(cl)
mean(typeI > 0.05)
# 0.067
plot(sort(threshold), type = "l",
     main= "Thresholds with Empirical Type I Error <= 0.05 in Each bootstrap",
     ylab = "Thershold")

hist(threshold,breaks = 40 )
abline(v = quantile(threshold, 1-delta), lty = "dashed", col = "red")
