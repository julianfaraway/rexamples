# Poisson GLMM using frequentist methods
[Julian Faraway](https://julianfaraway.github.io/)
2024-10-11

- [Data](#data)
- [Model](#model)
- [LME4](#lme4)
- [NLME](#nlme)
- [MMRM](#mmrm)
- [GLMMTMB](#glmmtmb)
- [Discussion](#discussion)
- [Package version info](#package-version-info)

See the [introduction](../index.md) for an overview.

See a [mostly Bayesian analysis](nitrofen.md) analysis of the same data.

This example is discussed in more detail in the book [Bayesian
Regression Modeling with
INLA](https://julianfaraway.github.io/brinlabook/chaglmm.html#sec:poissonglmm)

Packages used:

``` r
library(ggplot2)
```

# Data

In [Davison and Hinkley,
1997](https://doi.org/10.1017/CBO9780511802843), the results of a study
on Nitrofen, a herbicide, are reported. Due to concern regarding the
effect on animal life, 50 female water fleas were divided into five
groups of ten each and treated with different concentrations of the
herbicide. The number of offspring in three subsequent broods for each
flea was recorded. We start by loading the data from the `boot` package:
(the `boot` package comes with base R distribution so there is no need
to download this)

``` r
data(nitrofen, package="boot")
head(nitrofen)
```

      conc brood1 brood2 brood3 total
    1    0      3     14     10    27
    2    0      5     12     15    32
    3    0      6     11     17    34
    4    0      6     12     15    33
    5    0      6     15     15    36
    6    0      5     14     15    34

We need to rearrange the data to have one response value per line:

``` r
lnitrofen = data.frame(conc = rep(nitrofen$conc,each=3),
  live = as.numeric(t(as.matrix(nitrofen[,2:4]))),
  id = rep(1:50,each=3),
  brood = rep(1:3,50))
head(lnitrofen)
```

      conc live id brood
    1    0    3  1     1
    2    0   14  1     2
    3    0   10  1     3
    4    0    5  2     1
    5    0   12  2     2
    6    0   15  2     3

Make a plot of the data:

``` r
lnitrofen$jconc <- lnitrofen$conc + rep(c(-10,0,10),50)
lnitrofen$fbrood = factor(lnitrofen$brood)
ggplot(lnitrofen, aes(x=jconc,y=live, shape=fbrood, color=fbrood)) + 
       geom_point(position = position_jitter(w = 0, h = 0.5)) + 
       xlab("Concentration") + labs(shape = "Brood")
```

<img src="figs/fig-nitrodat-1..svg" id="fig-nitrodat"
alt="Figure 1: The number of live offspring varies with the concentration of Nitrofen and the brood number." />

# Model

Since the response is a small count, a Poisson model is a natural
choice. We expect the rate of the response to vary with the brood and
concentration level. The plot of the data suggests these two predictors
may have an interaction. The three observations for a single flea are
likely to be correlated. We might expect a given flea to tend to produce
more, or less, offspring over a lifetime. We can model this with an
additive random effect. The linear predictor is:

``` math
\eta_i = x_i^T \beta + u_{j(i)}, \quad i=1, \dots, 150. \quad j=1, \dots 50,
```

where $x_i$ is a vector from the design matrix encoding the information
about the $i^{th}$ observation and $u_j$ is the random affect associated
with the $j^{th}$ flea. The response has distribution
$Y_i \sim Poisson(\exp(\eta_i))$.

# LME4

We fit a model using penalized quasi-likelihood (PQL) using the `lme4`
package:

``` r
library(lme4)
glmod <- glmer(live ~ I(conc/300)*brood + (1|id), nAGQ=25, 
             family=poisson, data=lnitrofen)
summary(glmod, correlation = FALSE)
```

    Generalized linear mixed model fit by maximum likelihood (Adaptive Gauss-Hermite Quadrature, nAGQ = 25) ['glmerMod']
     Family: poisson  ( log )
    Formula: live ~ I(conc/300) * brood + (1 | id)
       Data: lnitrofen

         AIC      BIC   logLik deviance df.resid 
       334.5    349.5   -162.2    324.5      145 

    Scaled residuals: 
       Min     1Q Median     3Q    Max 
    -2.285 -0.858  0.068  0.706  2.866 

    Random effects:
     Groups Name        Variance Std.Dev.
     id     (Intercept) 0.0835   0.289   
    Number of obs: 150, groups:  id, 50

    Fixed effects:
                      Estimate Std. Error z value Pr(>|z|)
    (Intercept)         1.3451     0.1590    8.46  < 2e-16
    I(conc/300)         0.3581     0.2801    1.28      0.2
    brood               0.5815     0.0592    9.83  < 2e-16
    I(conc/300):brood  -0.7957     0.1158   -6.87  6.3e-12

We scaled the concentration by dividing by 300 (the maximum value is
310) to avoid scaling problems encountered with `glmer()`. This is
helpful in any case since it puts all the parameter estimates on a
similar scale. The first brood is the reference level so the slope for
this group is estimated as $-0.0437$ and is not statistically
significant, confirming the impression from the plot. We can see that
numbers of offspring in the second and third broods start out
significantly higher for zero concentration of the herbicide, with
estimates of $1.1688$ and $1.3512$. But as concentration increases, we
see that the numbers decrease significantly, with slopes of $-1.6730$
and $-1.8312$ relative to the first brood. The individual SD is
estimated at $0.302$ which is noticeably smaller than the estimates
above, indicating that the brood and concentration effects outweigh the
individual variation.

We can make a plot of the mean predicted response as concentration and
brood vary. I have chosen not specify a particular individual in the
random effects with the option `re.form=~0` . We have $u_i = 0$ and so
this represents the the response for a `typical` individual.

``` r
predf = data.frame(conc=rep(c(0,80,160,235,310),each=3),brood=rep(1:3,5))
predf$live = predict(glmod, newdata=predf, re.form=~0, type="response")
predf$brood = factor(predf$brood)
ggplot(predf, aes(x=conc,y=live,group=brood,color=brood)) + 
  geom_line() + xlab("Concentration")
```

<img src="figs/fig-prednitro-1..svg" id="fig-prednitro"
alt="Figure 2: Predicted number of live offspring" />

We see that if only the first brood were considered, the herbicide does
not have a large effect. In the second and third broods, the (negative)
effect of the herbicide becomes more apparent with fewer live offspring
being produced as the concentration rises.

# NLME

Cannot fit GLMMs (other than normal response).

# MMRM

Cannot fit GLMMs (other than normal response).

# GLMMTMB

See the discussion for the [single random effect
example](pulpfreq.md#GLMMTMB) for some introduction.

``` r
library(glmmTMB)
```

We can fit the model with:

``` r
gtmod <- glmmTMB(live ~ I(conc/300)*brood + (1|id), 
             family=poisson, data=lnitrofen)
summary(gtmod)
```

     Family: poisson  ( log )
    Formula:          live ~ I(conc/300) * brood + (1 | id)
    Data: lnitrofen

         AIC      BIC   logLik deviance df.resid 
       830.5    845.6   -410.3    820.5      145 

    Random effects:

    Conditional model:
     Groups Name        Variance Std.Dev.
     id     (Intercept) 0.0831   0.288   
    Number of obs: 150, groups:  id, 50

    Conditional model:
                      Estimate Std. Error z value Pr(>|z|)
    (Intercept)         1.3451     0.1589    8.47  < 2e-16
    I(conc/300)         0.3582     0.2799    1.28      0.2
    brood               0.5814     0.0591    9.83  < 2e-16
    I(conc/300):brood  -0.7955     0.1157   -6.87  6.3e-12

Almost the same output as with `lme4` although a different optimizer has
been used for the fit.

# Discussion

Only `lme4` and `glmmTMB` can manage this.

# Package version info

``` r
sessionInfo()
```

    R version 4.4.1 (2024-06-14)
    Platform: x86_64-apple-darwin20
    Running under: macOS Sonoma 14.7

    Matrix products: default
    BLAS:   /Library/Frameworks/R.framework/Versions/4.4-x86_64/Resources/lib/libRblas.0.dylib 
    LAPACK: /Library/Frameworks/R.framework/Versions/4.4-x86_64/Resources/lib/libRlapack.dylib;  LAPACK version 3.12.0

    locale:
    [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

    time zone: Europe/London
    tzcode source: internal

    attached base packages:
    [1] stats     graphics  grDevices utils     datasets  methods   base     

    other attached packages:
    [1] glmmTMB_1.1.9 lme4_1.1-35.5 Matrix_1.7-0  ggplot2_3.5.1

    loaded via a namespace (and not attached):
     [1] utf8_1.2.4          generics_0.1.3      lattice_0.22-6      digest_0.6.37       magrittr_2.0.3     
     [6] evaluate_0.24.0     grid_4.4.1          estimability_1.5.1  mvtnorm_1.2-6       fastmap_1.2.0      
    [11] jsonlite_1.8.8      mgcv_1.9-1          fansi_1.0.6         scales_1.3.0        numDeriv_2016.8-1.1
    [16] cli_3.6.3           rlang_1.1.4         munsell_0.5.1       splines_4.4.1       withr_3.0.1        
    [21] yaml_2.3.10         tools_4.4.1         nloptr_2.1.1        coda_0.19-4.1       minqa_1.2.8        
    [26] dplyr_1.1.4         colorspace_2.1-1    boot_1.3-31         vctrs_0.6.5         R6_2.5.1           
    [31] lifecycle_1.0.4     emmeans_1.10.4      MASS_7.3-61         pkgconfig_2.0.3     pillar_1.9.0       
    [36] gtable_0.3.5        glue_1.7.0          Rcpp_1.0.13         systemfonts_1.1.0   xfun_0.47          
    [41] tibble_3.2.1        tidyselect_1.2.1    rstudioapi_0.16.0   knitr_1.48          farver_2.1.2       
    [46] xtable_1.8-4        htmltools_0.5.8.1   nlme_3.1-166        rmarkdown_2.28      svglite_2.1.3      
    [51] labeling_0.4.3      TMB_1.9.14          compiler_4.4.1     
