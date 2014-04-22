require(Matrix)
logreg = function(y, x) {
    x = as.matrix(x)
    x = apply(x, 2, scale)
    x = cbind(1, x)
    m = nrow(x)
    n = ncol(x)
    alpha = 2/m

    # b = matrix(rnorm(n))
    # b = matrix(summary(lm(y~x))$coef[, 1])
    b = matrix(rep(0, n))
    v = exp(-x %*% b)
    h = 1 / (1 + v)

    J = -(t(y) %*% log(h) + t(1-y) %*% log(1 -h))
    derivJ = t(x) %*% (h-y)


    derivThresh = 0.0000001
    bThresh = 0.001
    while(1) {
        newb = b - alpha * derivJ
        if(max(abs(b - newb)) < bThresh) {
            break
        }
        v = exp(-x %*% newb)
        h = 1 / (1 + v)
        newderivJ = t(x) %*% (h-y)
        if(max(abs(newderivJ - derivJ)) < derivThresh) {
            break
        }
        newJ = -(t(y) %*% log(h) + t(0-y) %*% log(1 -h))
        if(newJ > J) {
            alpha = alpha/2
        }
        b = newb
        J = newJ
        derivJ = newderivJ
    }
    w = h^2 * v
    # # hessian matrix of cost function
    hess = t(x) %*% Diagonal(x = as.vector(w)) %*% x
    seMat = sqrt(diag(solve(hess)))
    zscore = b / seMat
    cbind(b, zscore)
}

nr = 5000
nc = 2
# set.seed(17)
x = matrix(rnorm(nr*nc, 3, 9), nr)
# x = apply(x, 2, scale)
# y = matrix(sample(0:1, nr, repl=T), nr)
h = 1/(1 + exp(-x %*% rnorm(nc)))
y = round(h)
y[1:round(nr/2)] = sample(0:1, round(nr/2), repl=T)


ntests = 13
testglm = function() {
    nr = 5000
    nc = 2
    # set.seed(17)
    x = matrix(rnorm(nr*nc, 3, 9), nr)
    # x = apply(x, 2, scale)
    # y = matrix(sample(0:1, nr, repl=T), nr)
    h = 1/(1 + exp(-x %*% rnorm(nc)))
    y = round(h)
    y[1:round(nr/2)] = sample(0:1, round(nr/2), repl=T)
    for(i in 1:ntests) {
        res = summary(glm(y~x, family=binomial))$coef[, c(1, 3)]
    }
    res
}

testlogreg = function() {
    nr = 5000
    nc = 2
    # set.seed(17)
    x = matrix(rnorm(nr*nc, 3, 9), nr)
    # x = apply(x, 2, scale)
    # y = matrix(sample(0:1, nr, repl=T), nr)
    h = 1/(1 + exp(-x %*% rnorm(nc)))
    y = round(h)
    y[1:round(nr/2)] = sample(0:1, round(nr/2), repl=T)
    for(i in 1:ntests) {
        res = logreg(y, x)
    }
    res
}

print(system.time(testlogreg()))
print(system.time(testglm()))

# nbench = 100
# benchres = matrix(NA, nbench, 2)
# bothres = matrix(NA, 2*nbench, nc)
# for(i in 1:nbench) {
#     x = matrix(rnorm(nr*nc, 3, 9), nr)
#     h = 1/(1 + exp(-x %*% rnorm(nc)))
#     y = round(h)
#     y[1:round(nr/2)] = sample(0:1, round(nr/2), repl=T)
#     myTime = system.time(myres <- testlogreg())[1]
#     glmTime = system.time(glmres <- testglm())[1]
#     timeratio = myTime/glmTime
#     resdiff = abs(myres[, 2] - glmres[, 2])
#     resdiff = sum(resdiff[2:length(resdiff)])
#     benchres[i, ] = c(timeratio, resdiff)
#     bothres[2*i-1, ] = myres[, 2][2:(nc+1)]
#     bothres[2*i, ] = glmres[, 2][2:(nc+1)]
# }
# head(benchres)
# head(bothres)

