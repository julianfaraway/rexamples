Randomized Block Design
================
[Julian Faraway](https://julianfaraway.github.io/)
07 July 2022

-   <a href="#data" id="toc-data">Data</a>
-   <a href="#questions" id="toc-questions">Questions</a>
-   <a href="#linear-model" id="toc-linear-model">Linear Model</a>
-   <a href="#mixed-effect-model" id="toc-mixed-effect-model">Mixed Effect
    Model</a>
-   <a href="#inla" id="toc-inla">INLA</a>
    -   <a href="#half-normal-priors-on-the-sds"
        id="toc-half-normal-priors-on-the-sds">Half-normal priors on the SDs</a>
    -   <a href="#informative-gamma-priors-on-the-precisions"
        id="toc-informative-gamma-priors-on-the-precisions">Informative gamma
        priors on the precisions</a>
    -   <a href="#penalized-complexity-prior"
        id="toc-penalized-complexity-prior">Penalized Complexity Prior</a>
-   <a href="#stan" id="toc-stan">STAN</a>
    -   <a href="#diagnostics" id="toc-diagnostics">Diagnostics</a>
    -   <a href="#output-summaries" id="toc-output-summaries">Output
        summaries</a>
    -   <a href="#posterior-distributions"
        id="toc-posterior-distributions">Posterior Distributions</a>
-   <a href="#brms" id="toc-brms">BRMS</a>
-   <a href="#mgcv" id="toc-mgcv">MGCV</a>
    -   <a href="#ginla" id="toc-ginla">GINLA</a>
-   <a href="#discussion" id="toc-discussion">Discussion</a>
-   <a href="#package-version-info" id="toc-package-version-info">Package
    version info</a>

See the [introduction](index.md) for an overview.

This example is discussed in more detail in my book [Extending the
Linear Model with R](https://julianfaraway.github.io/faraway/ELM/)

Required libraries:

``` r
library(faraway)
library(ggplot2)
library(lme4)
library(INLA)
library(knitr)
library(rstan, quietly=TRUE)
library(brms)
library(mgcv)
```

# Data

Load in and plot the data:

``` r
data(penicillin, package="faraway")
summary(penicillin)
```

     treat    blend       yield   
     A:5   Blend1:4   Min.   :77  
     B:5   Blend2:4   1st Qu.:81  
     C:5   Blend3:4   Median :87  
     D:5   Blend4:4   Mean   :86  
           Blend5:4   3rd Qu.:89  
                      Max.   :97  

``` r
ggplot(penicillin,aes(x=blend,y=yield,group=treat,linetype=treat))+geom_line()
```

![](figs/peni-1..svg)<!-- -->

``` r
ggplot(penicillin,aes(x=treat,y=yield,group=blend,linetype=blend))+geom_line()
```

![](figs/peni-2..svg)<!-- -->

The production of penicillin uses a raw material, corn steep liquor,
which is quite variable and can only be made in blends sufficient for
four runs. There are four processes, A, B, C and D, for the production.
See `help(penicillin)` for more information about the data.

In this example, the treatments are the four processes. These are the
specific four processes of interest that we wish to compare. The five
blends are five among many blends that would be randomly created during
production. We are not interested in these five specific blends but are
interested in how the blends vary. An interaction between blends and
treatments would complicate matters. But (a) there is no reason to
expect this exists and (b) with only one replicate per treatment and
blend combination, it is difficult to check for an interaction.

The plots show no outliers, no skewness, no obviously unequal variances
and no clear evidence of interaction. Let’s proceed.

# Questions

1.  Is there a difference between treatments? If so, what?
2.  Is there variation between the blends? What is the extent of this
    variation?

# Linear Model

Consider the model:

![y\_{ijk} = \mu + \tau_i + v_j + \epsilon\_{ijk}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;y_%7Bijk%7D%20%3D%20%5Cmu%20%2B%20%5Ctau_i%20%2B%20v_j%20%2B%20%5Cepsilon_%7Bijk%7D "y_{ijk} = \mu + \tau_i + v_j + \epsilon_{ijk}")

where the
![\mu](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu "\mu"),
![\tau_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau_i "\tau_i")
and
![v_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;v_j "v_j")
are fixed effects and the error
![\epsilon\_{ijk}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cepsilon_%7Bijk%7D "\epsilon_{ijk}")
is independent and identically distributed
![N(0,\sigma^2)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;N%280%2C%5Csigma%5E2%29 "N(0,\sigma^2)").
We can fit the model with:

``` r
lmod <- aov(yield ~ blend + treat, penicillin)
summary(lmod)
```

                Df Sum Sq Mean Sq F value Pr(>F)
    blend        4    264    66.0    3.50  0.041
    treat        3     70    23.3    1.24  0.339
    Residuals   12    226    18.8               

There is no significant difference between the treatments. The blends do
meet the 5% level for statistical significance. But this asserts a
difference between these particular five blends. It’s less clear what
this means about blends in general. We can get the estimated parameters
with:

``` r
coef(lmod)
```

    (Intercept) blendBlend2 blendBlend3 blendBlend4 blendBlend5      treatB      treatC      treatD 
             90          -9          -7          -4         -10           1           5           2 

Blend 1 and treatment A are the reference levels. We can also use a sum
(or deviation) coding:

``` r
op <- options(contrasts=c("contr.sum", "contr.poly"))
lmod <- aov(yield ~ blend + treat, penicillin)
coef(lmod)
```

    (Intercept)      blend1      blend2      blend3      blend4      treat1      treat2      treat3 
             86           6          -3          -1           2          -2          -1           3 

``` r
options(op)
```

The fit is the same but the parameterization is different. We can get
the full set of estimated effects as:

``` r
model.tables(lmod)
```

    Tables of effects

     blend 
    blend
    Blend1 Blend2 Blend3 Blend4 Blend5 
         6     -3     -1      2     -4 

     treat 
    treat
     A  B  C  D 
    -2 -1  3  0 

# Mixed Effect Model

Since we are not interested in the blends specifically, we may wish to
treat it as a random effect. The model becomes:

![y\_{ijk} = \mu + \tau_i + v_j + \epsilon\_{ijk}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;y_%7Bijk%7D%20%3D%20%5Cmu%20%2B%20%5Ctau_i%20%2B%20v_j%20%2B%20%5Cepsilon_%7Bijk%7D "y_{ijk} = \mu + \tau_i + v_j + \epsilon_{ijk}")

where the
![\mu](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu "\mu")
and![\tau_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau_i "\tau_i")
are fixed effects and the error
![\epsilon\_{ijk}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cepsilon_%7Bijk%7D "\epsilon_{ijk}")
is independent and identically distributed
![N(0,\sigma^2)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;N%280%2C%5Csigma%5E2%29 "N(0,\sigma^2)").
The
![v_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;v_j "v_j")
are now random effects and are independent and identically distributed
![N(0,\sigma^2_v)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;N%280%2C%5Csigma%5E2_v%29 "N(0,\sigma^2_v)").
We fit the model using REML: (again using sum coding)

``` r
op <- options(contrasts=c("contr.sum", "contr.poly"))
mmod <- lmer(yield ~ treat + (1|blend), penicillin)
faraway::sumary(mmod)
```

    Fixed Effects:
                coef.est coef.se
    (Intercept) 86.00     1.82  
    treat1      -2.00     1.68  
    treat2      -1.00     1.68  
    treat3       3.00     1.68  

    Random Effects:
     Groups   Name        Std.Dev.
     blend    (Intercept) 3.43    
     Residual             4.34    
    ---
    number of obs: 20, groups: blend, 5
    AIC = 118.6, DIC = 128
    deviance = 117.3 

``` r
options(op)
```

We get the same fixed effect estimates but now we have an estimated
blend SD. We can get random effect estimates:

``` r
ranef(mmod)$blend
```

           (Intercept)
    Blend1     4.28788
    Blend2    -2.14394
    Blend3    -0.71465
    Blend4     1.42929
    Blend5    -2.85859

which are a shrunk version of the fixed effect estimates. We can test
for a difference of the fixed effects with:

``` r
anova(mmod)
```

    Analysis of Variance Table
          npar Sum Sq Mean Sq F value
    treat    3     70    23.3    1.24

No p-value is supplied because there is some doubt in general over the
validity of the null F-distribution. In this specific example, with a
simple balanced design, it can be shown that the null F is correct. As
it happens, it is the same as that produced in the all fixed effects
analysis earlier:

``` r
anova(lmod)
```

    Analysis of Variance Table

    Response: yield
              Df Sum Sq Mean Sq F value Pr(>F)
    blend      4    264    66.0    3.50  0.041
    treat      3     70    23.3    1.24  0.339
    Residuals 12    226    18.8               

So no evidence of a difference between the treatments. More general
tests are available such as the Kenward-Roger method which adjusts the
degrees of freedom - see [Extending the Linear Model with
R](https://julianfaraway.github.io/faraway/ELM/) for details.

We can test the hypothesis
![H_0: \sigma^2_v = 0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;H_0%3A%20%5Csigma%5E2_v%20%3D%200 "H_0: \sigma^2_v = 0")
using a parametric bootstrap method:

``` r
rmod <- lmer(yield ~ treat + (1|blend), penicillin)
nlmod <- lm(yield ~ treat, penicillin)
as.numeric(2*(logLik(rmod)-logLik(nlmod,REML=TRUE)))
```

    [1] 2.7629

``` r
lrstatf <- numeric(1000)
for(i in 1:1000){
   ryield <-  unlist(simulate(nlmod))
   nlmodr <- lm(ryield ~ treat, penicillin)
   rmodr <- lmer(ryield ~ treat + (1|blend), penicillin)
   lrstatf[i] <- 2*(logLik(rmodr)-logLik(nlmodr,REML=TRUE))
  }
mean(lrstatf > 2.7629)
```

    [1] 0.043

The result falls just below the 5% level for significance. Because of
resampling variability, we should repeat with more boostrap samples. At
any rate, the evidence for variation between the blends is not decisive.

# INLA

Integrated nested Laplace approximation is a method of Bayesian
computation which uses approximation rather than simulation. More can be
found on this topic in [Bayesian Regression Modeling with
INLA](http://julianfaraway.github.io/brinla/) and the [chapter on
GLMMs](https://julianfaraway.github.io/brinlabook/chaglmm.html)

Use the most recent computational methodology:

``` r
inla.setOption(inla.mode="experimental")
inla.setOption("short.summary",TRUE)
```

Fit the default INLA model:

``` r
formula = yield ~ treat+f(blend, model="iid")
result = inla(formula, family="gaussian", data=penicillin)
summary(result)
```

    Fixed effects:
                  mean    sd 0.025quant 0.5quant 0.975quant   mode kld
    (Intercept) 84.046 2.394     79.318   84.043     88.787 84.040   0
    treatB       0.949 3.386     -5.754    0.951      7.637  0.956   0
    treatC       4.926 3.386     -1.781    4.929     11.610  4.936   0
    treatD       1.943 3.386     -4.761    1.946      8.630  1.951   0

    Model hyperparameters:
                                                mean      sd 0.025quant 0.5quant 0.975quant     mode
    Precision for the Gaussian observations 3.70e-02 1.2e-02      0.017 3.50e-02   6.50e-02    0.032
    Precision for blend                     2.17e+04 2.3e+04   1590.153 1.48e+04   8.25e+04 4372.481

     is computed 

Precision for the blend effect looks implausibly large. There is a
problem with default gamma prior (it needs to be more informative).

## Half-normal priors on the SDs

Try a half-normal prior on the blend precision. I have used variance of
the response to help with the scaling so these are more informative.

``` r
resprec <- 1/var(penicillin$yield)
formula = yield ~ treat+f(blend, model="iid", prior="logtnormal", param=c(0, resprec))
result = inla(formula, family="gaussian", data=penicillin)
summary(result)
```

    Fixed effects:
                  mean    sd 0.025quant 0.5quant 0.975quant   mode kld
    (Intercept) 84.029 2.758     78.542   84.028     89.520 84.026   0
    treatB       0.968 2.679     -4.352    0.970      6.274  0.974   0
    treatC       4.953 2.679     -0.370    4.957     10.257  4.962   0
    treatD       1.964 2.679     -3.357    1.967      7.270  1.971   0

    Model hyperparameters:
                                             mean    sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations 0.063 0.023      0.027    0.060      0.117 0.053
    Precision for blend                     0.104 0.120      0.013    0.068      0.416 0.032

     is computed 

Looks more plausible. Compute the transforms to an SD scale for the
blend and error. Make a table of summary statistics for the posteriors:

``` r
sigmaalpha <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[2]])
sigmaepsilon <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[1]])
restab=sapply(result$marginals.fixed, function(x) inla.zmarginal(x,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaalpha,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaepsilon,silent=TRUE))
restab=cbind(restab, sapply(result$marginals.random$blend,function(x) inla.zmarginal(x, silent=TRUE)))
colnames(restab) = c("mu","B-A","C-A","D-A","blend","error",levels(penicillin$blend))
data.frame(restab) |> kable()
```

|            | mu     | B.A      | C.A      | D.A     | blend  | error  | Blend1   | Blend2   | Blend3   | Blend4   | Blend5  |
|:-----------|:-------|:---------|:---------|:--------|:-------|:-------|:---------|:---------|:---------|:---------|:--------|
| mean       | 84.029 | 0.96792  | 4.9534   | 1.9643  | 4.1792 | 4.1966 | 4.3692   | -2.1837  | -0.7278  | 1.4558   | -2.9119 |
| sd         | 2.7568 | 2.677    | 2.6771   | 2.677   | 1.8807 | 0.8039 | 2.822    | 2.6164   | 2.552    | 2.5765   | 2.6714  |
| quant0.025 | 78.541 | -4.3532  | -0.37081 | -3.3576 | 1.5598 | 2.9273 | -0.64405 | -7.6795  | -5.9881  | -3.5195  | -8.5354 |
| quant0.25  | 82.275 | -0.76197 | 3.2242   | 0.23456 | 2.8237 | 3.619  | 2.4151   | -3.792   | -2.2642  | -0.16487 | -4.5747 |
| quant0.5   | 84.021 | 0.96369  | 4.9502   | 1.9603  | 3.831  | 4.0835 | 4.2545   | -2.0546  | -0.66455 | 1.3337   | -2.7805 |
| quant0.75  | 85.768 | 2.6876   | 6.6737   | 3.6841  | 5.1369 | 4.6599 | 6.1496   | -0.47936 | 0.81255  | 3.0081   | -1.1111 |
| quant0.975 | 89.508 | 6.2626   | 10.245   | 7.2582  | 8.8196 | 6.067  | 10.25    | 2.7147   | 4.3179   | 6.8181   | 1.9519  |

Also construct a plot the SD posteriors:

``` r
ddf <- data.frame(rbind(sigmaalpha,sigmaepsilon),errterm=gl(2,nrow(sigmaalpha),labels = c("blend","error")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("yield")+ylab("density")+xlim(0,15)
```

![](figs/penipdsd-1..svg)<!-- -->

Posterior for the blend SD is more diffuse than the error SD. Posterior
for the blend SD has zero density at zero.

Is there any variation between blends? We framed this question as an
hypothesis test previously but that is not sensible in this framework.
We might ask the probability that the blend SD is zero. Since we have
posited a continuous prior that places no discrete mass on zero, the
posterior probability will be zero, regardless of the data. Instead we
might ask the probability that the operator SD is small. Given the
response is measured to the nearest integer, 1 is a reasonable
representation of *small* if we take this to mean the smallest amount we
care about. (Clearly you cannot rely on the degree of rounding to make
such decisions in general).

We can compute the probability that the operator SD is smaller than 1:

``` r
inla.pmarginal(1, sigmaalpha)
```

    [1] 0.0013882

The probability is very small.

## Informative gamma priors on the precisions

Now try more informative gamma priors for the precisions. Define it so
the mean value of gamma prior is set to the inverse of the variance of
the fixed-effects model residuals. We expect the two error variances to
be lower than this variance so this is an overestimate. The variance of
the gamma prior (for the precision) is controlled by the `apar` shape
parameter - smaller values are less informative.

``` r
apar <- 0.5
lmod <- lm(yield ~ treat, data=penicillin)
bpar <- apar*var(residuals(lmod))
lgprior <- list(prec = list(prior="loggamma", param = c(apar,bpar)))
formula = yield ~ treat+f(blend, model="iid", hyper = lgprior)
result <- inla(formula, family="gaussian", data=penicillin)
summary(result)
```

    Fixed effects:
                  mean    sd 0.025quant 0.5quant 0.975quant   mode kld
    (Intercept) 84.028 2.878     78.320   84.027     89.742 84.025   0
    treatB       0.969 2.648     -4.285    0.971      6.210  0.974   0
    treatC       4.954 2.648     -0.302    4.957     10.193  4.962   0
    treatD       1.965 2.648     -3.289    1.967      7.206  1.971   0

    Model hyperparameters:
                                             mean    sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations 0.063 0.023      0.027    0.060      0.118 0.053
    Precision for blend                     0.072 0.057      0.012    0.056      0.222 0.032

     is computed 

Compute the summaries as before:

``` r
sigmaalpha <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[2]])
sigmaepsilon <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[1]])
restab=sapply(result$marginals.fixed, function(x) inla.zmarginal(x,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaalpha,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaepsilon,silent=TRUE))
restab=cbind(restab, sapply(result$marginals.random$blend,function(x) inla.zmarginal(x, silent=TRUE)))
colnames(restab) = c("mu","B-A","C-A","D-A","blend","error",levels(penicillin$blend))
data.frame(restab) |> kable()
```

|            | mu     | B.A      | C.A      | D.A     | blend  | error   | Blend1   | Blend2   | Blend3   | Blend4   | Blend5  |
|:-----------|:-------|:---------|:---------|:--------|:-------|:--------|:---------|:---------|:---------|:---------|:--------|
| mean       | 84.028 | 0.96865  | 4.9545   | 1.9651  | 4.5529 | 4.1993  | 4.6872   | -2.343   | -0.78083 | 1.5624   | -3.1242 |
| sd         | 2.8769 | 2.6461   | 2.6461   | 2.6461  | 1.7587 | 0.80031 | 2.8308   | 2.75     | 2.7258   | 2.7352   | 2.771   |
| quant0.025 | 78.319 | -4.2856  | -0.30263 | -3.2899 | 2.1301 | 2.9183  | -0.63291 | -7.9756  | -6.2717  | -3.7763  | -8.837  |
| quant0.25  | 82.217 | -0.74709 | 3.2393   | 0.24951 | 3.3022 | 3.6251  | 2.8526   | -4.0205  | -2.4534  | -0.15004 | -4.8132 |
| quant0.5   | 84.02  | 0.9643   | 4.9511   | 1.961   | 4.2028 | 4.0933  | 4.5851   | -2.2923  | -0.76736 | 1.5154   | -3.0581 |
| quant0.75  | 85.824 | 2.6741   | 6.6605   | 3.6707  | 5.423  | 4.6655  | 6.4006   | -0.62001 | 0.89837  | 3.2204   | -1.3721 |
| quant0.975 | 89.729 | 6.198    | 10.181   | 7.1938  | 8.9418 | 6.0469  | 10.565   | 2.9535   | 4.586    | 7.1087   | 2.1579  |

Make the plots:

``` r
ddf <- data.frame(rbind(sigmaalpha,sigmaepsilon),errterm=gl(2,nrow(sigmaalpha),labels = c("blend","error")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("yield")+ylab("density")+xlim(0,15)
```

![](figs/penigam-1..svg)<!-- -->

Posterior for blend SD has no weight near zero.

We can compute the probability that the operator SD is smaller than 1:

``` r
inla.pmarginal(1, sigmaalpha)
```

    [1] 2.6614e-05

The probability is very small.

## Penalized Complexity Prior

In [Simpson et al (2015)](http://arxiv.org/abs/1403.4630v3), penalized
complexity priors are proposed. This requires that we specify a scaling
for the SDs of the random effects. We use the SD of the residuals of the
fixed effects only model (what might be called the base model in the
paper) to provide this scaling.

``` r
lmod <- lm(yield ~ treat, data=penicillin)
sdres <- sd(residuals(lmod))
pcprior <- list(prec = list(prior="pc.prec", param = c(3*sdres,0.01)))
formula <- yield ~ treat + f(blend, model="iid", hyper = pcprior)
result <- inla(formula, family="gaussian", data=penicillin)
summary(result)
```

    Fixed effects:
                  mean    sd 0.025quant 0.5quant 0.975quant   mode kld
    (Intercept) 84.030 2.622     78.826   84.029     89.239 84.027   0
    treatB       0.967 2.724     -4.446    0.969      6.365  0.973   0
    treatC       4.952 2.724     -0.464    4.955     10.347  4.961   0
    treatD       1.963 2.724     -3.450    1.966      7.361  1.970   0

    Model hyperparameters:
                                             mean    sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations 0.062 0.023      0.027    0.059      0.118 0.052
    Precision for blend                     0.143 0.180      0.016    0.089      0.602 0.040

     is computed 

Compute the summaries as before:

``` r
sigmaalpha <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[2]])
sigmaepsilon <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[1]])
restab=sapply(result$marginals.fixed, function(x) inla.zmarginal(x,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaalpha,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaepsilon,silent=TRUE))
restab=cbind(restab, sapply(result$marginals.random$blend,function(x) inla.zmarginal(x, silent=TRUE)))
colnames(restab) = c("mu","B-A","C-A","D-A","blend","error",levels(penicillin$blend))
data.frame(restab) |> kable()
```

|            | mu     | B.A      | C.A      | D.A     | blend  | error   | Blend1   | Blend2   | Blend3   | Blend4   | Blend5   |
|:-----------|:-------|:---------|:---------|:--------|:-------|:--------|:---------|:---------|:---------|:---------|:---------|
| mean       | 84.03  | 0.96684  | 4.9519   | 1.9631  | 3.6588 | 4.2227  | 3.9866   | -1.9935  | -0.66442 | 1.3292   | -2.6592  |
| sd         | 2.6203 | 2.7222   | 2.7223   | 2.7222  | 1.6989 | 0.81716 | 2.7294   | 2.4339   | 2.3391   | 2.3754   | 2.5136   |
| quant0.025 | 78.826 | -4.4468  | -0.46515 | -3.4514 | 1.2961 | 2.9245  | -0.64689 | -7.1803  | -5.5456  | -3.1876  | -8.0116  |
| quant0.25  | 82.354 | -0.79008 | 3.1956   | 0.20633 | 2.4298 | 3.636   | 2.0132   | -3.4899  | -2.0471  | -0.16424 | -4.2359  |
| quant0.5   | 84.022 | 0.96262  | 4.9487   | 1.9592  | 3.3461 | 4.1109  | 3.8755   | -1.8219  | -0.56926 | 1.1624   | -2.4953  |
| quant0.75  | 85.692 | 2.7134   | 6.6991   | 3.7099  | 4.5282 | 4.6957  | 5.7381   | -0.36832 | 0.71771  | 2.7471   | -0.89293 |
| quant0.975 | 89.227 | 6.353    | 10.335   | 7.3485  | 7.8448 | 6.1176  | 9.681    | 2.4329   | 3.9459   | 6.3474   | 1.732    |

Make the plots:

``` r
ddf <- data.frame(rbind(sigmaalpha,sigmaepsilon),errterm=gl(2,nrow(sigmaalpha),labels = c("blend","error")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("yield")+ylab("density")+xlim(0,15)
```

![](figs/penipc-1..svg)<!-- -->

Posterior for blend SD has no weight at zero. Results are comparable to
previous analyses.

We can plot the posterior marginals of the random effects:

``` r
nlevels = length(unique(penicillin$blend))
rdf = data.frame(do.call(rbind,result$marginals.random$blend))
rdf$blend = gl(nlevels,nrow(rdf)/nlevels,labels=1:nlevels)
ggplot(rdf,aes(x=x,y=y,group=blend, color=blend)) + 
  geom_line() +
  xlab("") + ylab("Density") 
```

![](figs/penirandeffpden-1..svg)<!-- -->

There is substantial overlap and we cannot distinguish the blends.

We can compute the probability that the operator SD is smaller than 1:

``` r
inla.pmarginal(1, sigmaalpha)
```

    [1] 0.0061713

The probability is still very small.

# STAN

[STAN](https://mc-stan.org/) performs Bayesian inference using MCMC.

Set up STAN to use multiple cores. Set the random number seed for
reproducibility.

``` r
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
set.seed(123)
```

We need to use a STAN command file
[penicillin.stan](stancode/penicillin.stan) which we view here:

``` r
writeLines(readLines("stancode/penicillin.stan"))
```

    data {
      int<lower=0> N;
      int<lower=0> Nt;
      int<lower=0> Nb;
      int<lower=1,upper=Nt> treat[N];
      int<lower=1,upper=Nb> blk[N];
      vector[N] y;
    }
    parameters {
      vector[Nb] eta;
      vector[Nt] trt;
      real<lower=0> sigmablk;
      real<lower=0> sigmaepsilon;
    }
    transformed parameters {
      vector[Nb] bld;
      vector[N] yhat;

      bld <- sigmablk * eta;

      for (i in 1:N)
        yhat[i] <- trt[treat[i]]+bld[blk[i]];

    }
    model {
      eta ~ normal(0, 1);

      y ~ normal(yhat, sigmaepsilon);
    }

We have used uninformative priors for the treatment effects and the two
variances. We prepare data in a format consistent with the command file.
Needs to be a list.

``` r
ntreat <- as.numeric(penicillin$treat)
blk <- as.numeric(penicillin$blend)
penidat <- list(N=nrow(penicillin), Nt=max(ntreat), Nb=max(blk), treat=ntreat, blk=blk, y=penicillin$yield)
```

``` r
rt <- stanc(file="stancode/penicillin.stan")
suppressMessages(sm <- stan_model(stanc_ret = rt, verbose=FALSE))
```

    Running /Library/Frameworks/R.framework/Resources/bin/R CMD SHLIB foo.c
    clang -mmacosx-version-min=10.13 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG   -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/Rcpp/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/unsupported"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/BH/include" -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/src/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppParallel/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/rstan/include" -DEIGEN_NO_DEBUG  -DBOOST_DISABLE_ASSERTS  -DBOOST_PENDING_INTEGER_LOG2_HPP  -DSTAN_THREADS  -DUSE_STANC3 -DSTRICT_R_HEADERS  -DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION  -DBOOST_NO_AUTO_PTR  -include '/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp'  -D_REENTRANT -DRCPP_PARALLEL_USE_TBB=1   -I/usr/local/include   -fPIC  -Wall -g -O2  -c foo.c -o foo.o
    In file included from <built-in>:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Dense:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Core:88:
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:628:1: error: unknown type name 'namespace'
    namespace Eigen {
    ^
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:628:16: error: expected ';' after top level declarator
    namespace Eigen {
                   ^
                   ;
    In file included from <built-in>:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Dense:1:
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Core:96:10: fatal error: 'complex' file not found
    #include <complex>
             ^~~~~~~~~
    3 errors generated.
    make: *** [foo.o] Error 1

``` r
system.time(fit <- sampling(sm, data=penidat))
```

       user  system elapsed 
      1.840   0.254   0.888 

We get some warnings but nothing too serious.

## Diagnostics

Plot the chains for the block SD

``` r
pname <- "sigmablk"
muc <- rstan::extract(fit, pars=pname,  permuted=FALSE, inc_warmup=FALSE)
mdf <- reshape2::melt(muc)
ggplot(mdf,aes(x=iterations,y=value,color=chains)) + geom_line() + ylab(mdf$parameters[1])
```

![](figs/penisigmablk-1..svg)<!-- -->

which is satistfactory. The same for the error SD:

``` r
pname <- "sigmaepsilon"
muc <- rstan::extract(fit, pars=pname,  permuted=FALSE, inc_warmup=FALSE)
mdf <- reshape2::melt(muc)
ggplot(mdf,aes(x=iterations,y=value,color=chains)) + geom_line() + ylab(mdf$parameters[1])
```

![](figs/penisigmaepsilon-1..svg)<!-- -->

which also looks reasonable.

## Output summaries

Examine the output:

``` r
fit
```

    Inference for Stan model: penicillin.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.

                   mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
    eta[1]         0.92    0.02 0.72  -0.48   0.46   0.91   1.38   2.37  2056    1
    eta[2]        -0.53    0.02 0.68  -1.89  -0.94  -0.51  -0.09   0.77  1991    1
    eta[3]        -0.20    0.02 0.67  -1.53  -0.63  -0.20   0.21   1.14  1928    1
    eta[4]         0.28    0.02 0.67  -0.99  -0.13   0.26   0.71   1.61  1840    1
    eta[5]        -0.68    0.02 0.68  -2.12  -1.10  -0.66  -0.24   0.60  2000    1
    trt[1]        84.23    0.08 3.04  78.04  82.27  84.26  86.18  90.39  1451    1
    trt[2]        85.25    0.09 3.08  79.15  83.24  85.23  87.18  91.57  1265    1
    trt[3]        89.20    0.09 3.17  83.14  87.17  89.18  91.18  95.65  1284    1
    trt[4]        86.18    0.09 3.06  79.96  84.31  86.14  88.13  92.32  1200    1
    sigmablk       4.82    0.11 3.11   0.66   2.80   4.22   6.08  12.71   795    1
    sigmaepsilon   4.92    0.03 1.15   3.22   4.10   4.73   5.52   7.56  1529    1
    bld[1]         3.91    0.09 3.13  -1.51   1.70   3.74   5.92  10.58  1153    1
    bld[2]        -2.24    0.09 2.91  -8.65  -4.03  -2.01  -0.29   3.01  1118    1
    bld[3]        -0.92    0.08 2.87  -7.10  -2.55  -0.75   0.80   4.73  1306    1
    bld[4]         1.17    0.08 2.81  -4.11  -0.53   0.99   2.80   7.12  1246    1
    bld[5]        -2.96    0.09 2.97  -9.64  -4.74  -2.65  -0.89   2.20  1198    1
    yhat[1]       88.14    0.07 3.28  81.46  86.00  88.21  90.35  94.41  2190    1
    yhat[2]       89.16    0.06 3.24  82.62  87.02  89.19  91.38  95.32  2492    1
    yhat[3]       93.11    0.07 3.29  86.35  90.97  93.24  95.32  99.45  2405    1
    yhat[4]       90.09    0.07 3.19  83.47  88.08  90.24  92.15  96.29  2296    1
    yhat[5]       81.99    0.05 2.97  76.12  79.97  81.98  83.88  87.98  3770    1
    yhat[6]       83.01    0.05 2.87  77.35  81.13  83.05  84.86  88.65  3460    1
    yhat[7]       86.96    0.05 3.03  80.97  85.01  86.95  88.91  92.98  4052    1
    yhat[8]       83.93    0.05 2.95  78.22  81.95  83.90  85.81  89.72  3446    1
    yhat[9]       83.31    0.04 2.86  77.76  81.38  83.34  85.12  89.19  4809    1
    yhat[10]      84.33    0.04 2.87  78.63  82.56  84.36  86.11  90.01  4251    1
    yhat[11]      88.28    0.04 2.96  82.36  86.43  88.26  90.15  94.13  4410    1
    yhat[12]      85.26    0.04 2.88  79.61  83.38  85.27  87.12  90.88  4130    1
    yhat[13]      85.40    0.05 2.91  79.56  83.54  85.40  87.30  91.17  3927    1
    yhat[14]      86.42    0.04 2.87  80.94  84.59  86.35  88.33  92.00  4270    1
    yhat[15]      90.37    0.05 2.97  84.27  88.51  90.40  92.32  95.88  4325    1
    yhat[16]      87.35    0.05 2.89  81.59  85.51  87.36  89.16  93.16  4047    1
    yhat[17]      81.27    0.05 2.96  75.32  79.39  81.22  83.17  87.25  3659    1
    yhat[18]      82.29    0.05 2.99  76.31  80.38  82.32  84.29  88.13  3187    1
    yhat[19]      86.24    0.05 3.05  80.16  84.27  86.17  88.23  92.37  3519    1
    yhat[20]      83.22    0.06 3.00  77.31  81.31  83.18  85.17  89.27  2789    1
    lp__         -39.95    0.14 3.54 -48.21 -42.08 -39.54 -37.33 -34.30   682    1

    Samples were drawn using NUTS(diag_e) at Wed Jul  6 14:25:59 2022.
    For each parameter, n_eff is a crude measure of effective sample size,
    and Rhat is the potential scale reduction factor on split chains (at 
    convergence, Rhat=1).

We are not interested in the `yhat` values. In bigger datasets, there
might be a lot of these so we can select which parameters we view:

``` r
print(fit, pars=c("trt","sigmablk","sigmaepsilon","bld"))
```

    Inference for Stan model: penicillin.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.

                  mean se_mean   sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
    trt[1]       84.23    0.08 3.04 78.04 82.27 84.26 86.18 90.39  1451    1
    trt[2]       85.25    0.09 3.08 79.15 83.24 85.23 87.18 91.57  1265    1
    trt[3]       89.20    0.09 3.17 83.14 87.17 89.18 91.18 95.65  1284    1
    trt[4]       86.18    0.09 3.06 79.96 84.31 86.14 88.13 92.32  1200    1
    sigmablk      4.82    0.11 3.11  0.66  2.80  4.22  6.08 12.71   795    1
    sigmaepsilon  4.92    0.03 1.15  3.22  4.10  4.73  5.52  7.56  1529    1
    bld[1]        3.91    0.09 3.13 -1.51  1.70  3.74  5.92 10.58  1153    1
    bld[2]       -2.24    0.09 2.91 -8.65 -4.03 -2.01 -0.29  3.01  1118    1
    bld[3]       -0.92    0.08 2.87 -7.10 -2.55 -0.75  0.80  4.73  1306    1
    bld[4]        1.17    0.08 2.81 -4.11 -0.53  0.99  2.80  7.12  1246    1
    bld[5]       -2.96    0.09 2.97 -9.64 -4.74 -2.65 -0.89  2.20  1198    1

    Samples were drawn using NUTS(diag_e) at Wed Jul  6 14:25:59 2022.
    For each parameter, n_eff is a crude measure of effective sample size,
    and Rhat is the potential scale reduction factor on split chains (at 
    convergence, Rhat=1).

We see the posterior mean, SE and SD of the samples. We see some
quantiles from which we could construct a 95% credible interval (for
example). The `n_eff` is a rough measure of the sample size taking into
account the correlation in the samples. The effective sample sizes for
the primary parameters is adequate for most purposes. The
![\hat R](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%20R "\hat R")
statistics are good.

We can also get the posterior means alone.

``` r
(get_posterior_mean(fit, pars=c("eta","trt","sigmablk","sigmaepsilon","bld")))
```

                 mean-chain:1 mean-chain:2 mean-chain:3 mean-chain:4 mean-all chains
    eta[1]            0.85154      0.91723      0.97356      0.94087         0.92080
    eta[2]           -0.57892     -0.57664     -0.49303     -0.45731        -0.52648
    eta[3]           -0.22657     -0.23535     -0.18038     -0.17078        -0.20327
    eta[4]            0.22122      0.25797      0.33333      0.31521         0.28193
    eta[5]           -0.70092     -0.75471     -0.65218     -0.62864        -0.68411
    trt[1]           84.43379     84.33595     84.01538     84.14854        84.23341
    trt[2]           85.42798     85.58686     84.96935     85.01704        85.25031
    trt[3]           89.45047     89.54111     88.91072     88.90254        89.20121
    trt[4]           86.41803     86.41739     86.01618     85.86442        86.17900
    sigmablk          5.18674      4.65618      4.46743      4.96364         4.81850
    sigmaepsilon      4.92640      4.87123      4.93308      4.96310         4.92345
    bld[1]            3.80367      3.63264      4.05038      4.15474         3.91036
    bld[2]           -2.54002     -2.45345     -1.95192     -2.03254        -2.24448
    bld[3]           -1.14556     -1.04827     -0.69713     -0.79113        -0.92052
    bld[4]            0.91220      0.97052      1.41382      1.38849         1.17126
    bld[5]           -3.22688     -3.18165     -2.63792     -2.79434        -2.96020

We see that we get this information for each chain as well as overall.
This gives a sense of why running more than one chain might be helpful
in assessing the uncertainty in the posterior inference.

## Posterior Distributions

We can use extract to get at various components of the STAN fit.

``` r
postsig <- rstan::extract(fit, pars=c("sigmablk","sigmaepsilon"))
ref <- reshape2::melt(postsig,value.name="yield")
ggplot(data=ref,aes(x=yield, color=L1))+geom_density()+guides(color=guide_legend(title="SD"))
```

![](figs/penistanpdsig-1..svg)<!-- -->

We see that the error SD can be localized much more than the block SD.
We can compute the chance that the block SD is less than one. We’ve
chosen 1 as the response is only measured to the nearest integer so an
SD of less than one would not be particularly noticeable.

``` r
mean(postsig$sigmablk < 1)
```

    [1] 0.042

We see that this probability is small and would be smaller if we had
specified a lower threshold.

We can also look at the blend effects:

``` r
opre <- rstan::extract(fit, pars="bld")
ref <- reshape2::melt(opre, value.name="yield")
ggplot(data=ref,aes(x=yield, color=factor(Var2)))+geom_density()+guides(color=guide_legend(title="blend"))
```

![](figs/penistanblendrf-1..svg)<!-- -->

We see that all five blend distributions clearly overlap zero. We can
also look at the treatment effects:

``` r
opre <- rstan::extract(fit, pars="trt")
ref <- reshape2::melt(opre, value.name="yield")
ref[,2] <- (LETTERS[1:4])[ref[,2]]
ggplot(data=ref,aes(x=yield, color=factor(Var2)))+geom_density()+guides(color=guide_legend(title="treatment"))
```

![](figs/penistantrt-1..svg)<!-- -->

We did not include an intercept so the treatment effects are not
differences from zero. We see the distributions overlap substantially.

# BRMS

BRMS stands for Bayesian Regression Models with STAN. It provides a
convenient wrapper to STAN functionality.

Fitting the model is very similar to `lmer` as seen above:

``` r
suppressMessages(bmod <- brm(yield ~ treat + (1|blend), penicillin, cores=4))
```

    Running /Library/Frameworks/R.framework/Resources/bin/R CMD SHLIB foo.c
    clang -mmacosx-version-min=10.13 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG   -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/Rcpp/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/unsupported"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/BH/include" -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/src/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppParallel/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/rstan/include" -DEIGEN_NO_DEBUG  -DBOOST_DISABLE_ASSERTS  -DBOOST_PENDING_INTEGER_LOG2_HPP  -DSTAN_THREADS  -DUSE_STANC3 -DSTRICT_R_HEADERS  -DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION  -DBOOST_NO_AUTO_PTR  -include '/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp'  -D_REENTRANT -DRCPP_PARALLEL_USE_TBB=1   -I/usr/local/include   -fPIC  -Wall -g -O2  -c foo.c -o foo.o
    In file included from <built-in>:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Dense:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Core:88:
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:628:1: error: unknown type name 'namespace'
    namespace Eigen {
    ^
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:628:16: error: expected ';' after top level declarator
    namespace Eigen {
                   ^
                   ;
    In file included from <built-in>:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Dense:1:
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Core:96:10: fatal error: 'complex' file not found
    #include <complex>
             ^~~~~~~~~
    3 errors generated.
    make: *** [foo.o] Error 1

We get some warnings but not as severe as seen with our STAN fit above.
We can obtain some posterior densities and diagnostics with:

``` r
plot(bmod)
```

![](figs/penibrmsdiag-1..svg)<!-- -->![](figs/penibrmsdiag-2..svg)<!-- -->

Looks OK.

We can look at the STAN code that `brms` used with:

``` r
stancode(bmod)
```

    // generated with brms 2.17.0
    functions {
    }
    data {
      int<lower=1> N;  // total number of observations
      vector[N] Y;  // response variable
      int<lower=1> K;  // number of population-level effects
      matrix[N, K] X;  // population-level design matrix
      // data for group-level effects of ID 1
      int<lower=1> N_1;  // number of grouping levels
      int<lower=1> M_1;  // number of coefficients per level
      int<lower=1> J_1[N];  // grouping indicator per observation
      // group-level predictor values
      vector[N] Z_1_1;
      int prior_only;  // should the likelihood be ignored?
    }
    transformed data {
      int Kc = K - 1;
      matrix[N, Kc] Xc;  // centered version of X without an intercept
      vector[Kc] means_X;  // column means of X before centering
      for (i in 2:K) {
        means_X[i - 1] = mean(X[, i]);
        Xc[, i - 1] = X[, i] - means_X[i - 1];
      }
    }
    parameters {
      vector[Kc] b;  // population-level effects
      real Intercept;  // temporary intercept for centered predictors
      real<lower=0> sigma;  // dispersion parameter
      vector<lower=0>[M_1] sd_1;  // group-level standard deviations
      vector[N_1] z_1[M_1];  // standardized group-level effects
    }
    transformed parameters {
      vector[N_1] r_1_1;  // actual group-level effects
      real lprior = 0;  // prior contributions to the log posterior
      r_1_1 = (sd_1[1] * (z_1[1]));
      lprior += student_t_lpdf(Intercept | 3, 87, 5.9);
      lprior += student_t_lpdf(sigma | 3, 0, 5.9)
        - 1 * student_t_lccdf(0 | 3, 0, 5.9);
      lprior += student_t_lpdf(sd_1 | 3, 0, 5.9)
        - 1 * student_t_lccdf(0 | 3, 0, 5.9);
    }
    model {
      // likelihood including constants
      if (!prior_only) {
        // initialize linear predictor term
        vector[N] mu = Intercept + rep_vector(0.0, N);
        for (n in 1:N) {
          // add more terms to the linear predictor
          mu[n] += r_1_1[J_1[n]] * Z_1_1[n];
        }
        target += normal_id_glm_lpdf(Y | Xc, mu, b, sigma);
      }
      // priors including constants
      target += lprior;
      target += std_normal_lpdf(z_1[1]);
    }
    generated quantities {
      // actual population-level intercept
      real b_Intercept = Intercept - dot_product(means_X, b);
    }

We see that `brms` is using student t distributions with 3 degrees of
freedom for the priors. For the two error SDs, this will be truncated at
zero to form half-t distributions. You can get a more explicit
description of the priors with `prior_summary(bmod)`. These are
qualitatively similar to the half-normal and the PC prior used in the
INLA fit.

We examine the fit:

``` r
summary(bmod)
```

     Family: gaussian 
      Links: mu = identity; sigma = identity 
    Formula: yield ~ treat + (1 | blend) 
       Data: penicillin (Number of observations: 20) 
      Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
             total post-warmup draws = 4000

    Group-Level Effects: 
    ~blend (Number of levels: 5) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     3.66      2.16     0.27     8.89 1.00      925      833

    Population-Level Effects: 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    Intercept    84.19      2.68    78.84    89.55 1.00     1439     1294
    treatB        1.00      3.21    -5.46     7.31 1.00     2321     2415
    treatC        4.99      3.18    -1.53    11.24 1.00     2405     2510
    treatD        1.99      3.17    -4.31     8.31 1.00     2395     2391

    Family Specific Parameters: 
          Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sigma     4.84      1.07     3.22     7.27 1.00     1359     1577

    Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
    and Tail_ESS are effective sample size measures, and Rhat is the potential
    scale reduction factor on split chains (at convergence, Rhat = 1).

The parameterisation of the treatment effects is different from the STAN
version but not in an important way.

We can estimate the tail probability as before

``` r
bps = posterior_samples(bmod)
mean(bps$sd_blend__Intercept < 1)
```

    [1] 0.086

A somewhat higher value than seen previously. The priors used here put
greater weight on smaller values of the SD.

# MGCV

It is possible to fit some GLMMs within the GAM framework of the `mgcv`
package. An explanation of this can be found in this
[blog](https://fromthebottomoftheheap.net/2021/02/02/random-effects-in-gams/)

The `blend` term must be a factor for this to work:

``` r
gmod = gam(yield ~ treat + s(blend, bs = 're'), data=penicillin, method="REML")
```

and look at the summary output:

``` r
summary(gmod)
```


    Family: gaussian 
    Link function: identity 

    Formula:
    yield ~ treat + s(blend, bs = "re")

    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)
    (Intercept)    84.00       2.47   33.94  3.5e-14
    treatB          1.00       2.74    0.36    0.721
    treatC          5.00       2.74    1.82    0.091
    treatD          2.00       2.74    0.73    0.479

    Approximate significance of smooth terms:
              edf Ref.df   F p-value
    s(blend) 2.86      4 2.5   0.038

    R-sq.(adj) =  0.361   Deviance explained = 55.8%
    -REML = 51.915  Scale est. = 18.833    n = 20

We get the fixed effect estimates. We also get a test on the random
effect (as described in this
[article](https://doi.org/10.1093/biomet/ast038). The hypothesis of no
variation between the operators is rejected.

We can get an estimate of the operator and error SD:

``` r
gam.vcomp(gmod)
```


    Standard deviations and 0.95 confidence intervals:

             std.dev  lower  upper
    s(blend)  3.4339 1.2853 9.1744
    scale     4.3397 2.9088 6.4746

    Rank: 2/2

which is the same as the REML estimate from `lmer` earlier.

The random effect estimates for the four operators can be found with:

``` r
coef(gmod)
```

    (Intercept)      treatB      treatC      treatD  s(blend).1  s(blend).2  s(blend).3  s(blend).4  s(blend).5 
       84.00000     1.00000     5.00000     2.00000     4.28789    -2.14394    -0.71465     1.42930    -2.85859 

which is again the same as before.

## GINLA

In [Wood (2019)](https://doi.org/10.1093/biomet/asz044), a simplified
version of INLA is proposed. The first construct the GAM model without
fitting and then use the `ginla()` function to perform the computation.

``` r
gmod = gam(yield ~ treat + s(blend, bs = 're'), data=penicillin, fit = FALSE)
gimod = ginla(gmod)
```

We get the posterior density for the intercept as:

``` r
plot(gimod$beta[1,],gimod$density[1,],type="l",xlab="yield",ylab="density")
```

![](figs/peniginlaint-1..svg)<!-- -->

and for the treatment effects as:

``` r
xmat = t(gimod$beta[2:4,])
ymat = t(gimod$density[2:4,])
matplot(xmat, ymat,type="l",xlab="yield",ylab="density")
legend("right",c("B","C","D"),col=1:3,lty=1:3)
```

![](figs/peniginlateff-1..svg)<!-- -->

``` r
xmat = t(gimod$beta[5:9,])
ymat = t(gimod$density[5:9,])
matplot(xmat, ymat,type="l",xlab="yield",ylab="density")
legend("right",paste0("blend",1:5),col=1:5,lty=1:5)
```

![](figs/peniginlareff-1..svg)<!-- -->

It is not straightforward to obtain the posterior densities of the
hyperparameters.

# Discussion

See the [Discussion of the single random effect
model](pulp.md#Discussion) for general comments. In this example, the
default model for INLA failed due to a default prior that was
insufficiently informative. But the default prior in STAN produced more
credible results. As in the simple single random effect sample, the
conclusions were very sensitive to the choice of prior. There was a
substantive difference between STAN and INLA, particularly regarding the
lower tail of the blend SD. Although the priors are not identical,
preventing direct comparison, STAN does give higher weight to the lower
tail. This is concerning.

# Package version info

``` r
sessionInfo()
```

    R version 4.2.0 (2022-04-22)
    Platform: x86_64-apple-darwin17.0 (64-bit)
    Running under: macOS Big Sur/Monterey 10.16

    Matrix products: default
    BLAS:   /Library/Frameworks/R.framework/Versions/4.2/Resources/lib/libRblas.0.dylib
    LAPACK: /Library/Frameworks/R.framework/Versions/4.2/Resources/lib/libRlapack.dylib

    locale:
    [1] en_GB.UTF-8/en_GB.UTF-8/en_GB.UTF-8/C/en_GB.UTF-8/en_GB.UTF-8

    attached base packages:
    [1] parallel  stats     graphics  grDevices utils     datasets  methods   base     

    other attached packages:
     [1] mgcv_1.8-40         nlme_3.1-157        brms_2.17.0         Rcpp_1.0.8.3        rstan_2.26.13      
     [6] StanHeaders_2.26.13 knitr_1.39          INLA_22.06.20-2     sp_1.4-7            foreach_1.5.2      
    [11] lme4_1.1-29         Matrix_1.4-1        ggplot2_3.3.6       faraway_1.0.8      

    loaded via a namespace (and not attached):
      [1] minqa_1.2.4          colorspace_2.0-3     ellipsis_0.3.2       ggridges_0.5.3       markdown_1.1        
      [6] base64enc_0.1-3      rstudioapi_0.13      Deriv_4.1.3          farver_2.1.0         DT_0.23             
     [11] fansi_1.0.3          mvtnorm_1.1-3        bridgesampling_1.1-2 codetools_0.2-18     splines_4.2.0       
     [16] shinythemes_1.2.0    bayesplot_1.9.0      jsonlite_1.8.0       nloptr_2.0.3         shiny_1.7.1         
     [21] compiler_4.2.0       backports_1.4.1      assertthat_0.2.1     fastmap_1.1.0        cli_3.3.0           
     [26] later_1.3.0          htmltools_0.5.2      prettyunits_1.1.1    tools_4.2.0          igraph_1.3.1        
     [31] coda_0.19-4          gtable_0.3.0         glue_1.6.2           reshape2_1.4.4       dplyr_1.0.9         
     [36] posterior_1.2.2      V8_4.2.0             vctrs_0.4.1          svglite_2.1.0        iterators_1.0.14    
     [41] crosstalk_1.2.0      tensorA_0.36.2       xfun_0.31            stringr_1.4.0        ps_1.7.0            
     [46] mime_0.12            miniUI_0.1.1.1       lifecycle_1.0.1      gtools_3.9.2.1       MASS_7.3-57         
     [51] zoo_1.8-10           scales_1.2.0         colourpicker_1.1.1   promises_1.2.0.1     Brobdingnag_1.2-7   
     [56] inline_0.3.19        shinystan_2.6.0      yaml_2.3.5           curl_4.3.2           gridExtra_2.3       
     [61] loo_2.5.1            stringi_1.7.6        highr_0.9            dygraphs_1.1.1.6     checkmate_2.1.0     
     [66] boot_1.3-28          pkgbuild_1.3.1       rlang_1.0.2          pkgconfig_2.0.3      systemfonts_1.0.4   
     [71] matrixStats_0.62.0   distributional_0.3.0 evaluate_0.15        lattice_0.20-45      purrr_0.3.4         
     [76] rstantools_2.2.0     htmlwidgets_1.5.4    labeling_0.4.2       processx_3.5.3       tidyselect_1.1.2    
     [81] plyr_1.8.7           magrittr_2.0.3       R6_2.5.1             generics_0.1.2       DBI_1.1.2           
     [86] pillar_1.7.0         withr_2.5.0          xts_0.12.1           abind_1.4-5          tibble_3.1.7        
     [91] crayon_1.5.1         utf8_1.2.2           rmarkdown_2.14       grid_4.2.0           callr_3.7.0         
     [96] threejs_0.3.3        digest_0.6.29        xtable_1.8-4         httpuv_1.6.5         RcppParallel_5.1.5  
    [101] stats4_4.2.0         munsell_0.5.0        shinyjs_2.1.0       
