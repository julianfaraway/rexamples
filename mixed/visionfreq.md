# Repeated Measures with Vision Data using Frequentist Methods
[Julian Faraway](https://julianfaraway.github.io/)
2024-10-09

- [Data](#data)
- [Mixed Effect Model](#mixed-effect-model)
- [LME4](#lme4)
- [NLME](#nlme)
- [MMRM](#mmrm)
- [GLMMTMB](#glmmtmb)
- [Discussion](#discussion)
- [Package version info](#package-version-info)

See the [introduction](index.md) for an overview.

See a [mostly Bayesian analysis](vision.md) analysis of the same data.

This example is discussed in more detail in my book [Extending the
Linear Model with R](https://julianfaraway.github.io/faraway/ELM/)

Required libraries:

``` r
library(faraway)
library(ggplot2)
```

# Data

The acuity of vision for seven subjects was tested. The response is the
lag in milliseconds between a light flash and a response in the cortex
of the eye. Each eye is tested at four different powers of lens. An
object at the distance of the second number appears to be at distance of
the first number.

Load in and look at the data:

``` r
data(vision, package="faraway")
ftable(xtabs(acuity ~ eye + subject + power, data=vision))
```

                  power 6/6 6/18 6/36 6/60
    eye   subject                         
    left  1             116  119  116  124
          2             110  110  114  115
          3             117  118  120  120
          4             112  116  115  113
          5             113  114  114  118
          6             119  115   94  116
          7             110  110  105  118
    right 1             120  117  114  122
          2             106  112  110  110
          3             120  120  120  124
          4             115  116  116  119
          5             114  117  116  112
          6             100   99   94   97
          7             105  105  115  115

We create a numeric version of the power to make a plot:

``` r
vision$npower <- rep(1:4,14)
ggplot(vision, aes(y=acuity, x=npower, linetype=eye)) + geom_line() + facet_wrap(~ subject, ncol=4) + scale_x_continuous("Power",breaks=1:4,labels=c("6/6","6/18","6/36","6/60"))
```

![](figs/visplot-1..svg)

# Mixed Effect Model

The power is a fixed effect. In the model below, we have treated it as a
nominal factor, but we could try fitting it in a quantitative manner.
The subjects should be treated as random effects. Since we do not
believe there is any consistent right-left eye difference between
individuals, we should treat the eye factor as nested within subjects.

The model can be written as:

``` math
y_{ijk} = \mu + p_j + s_i + e_{ik} + \epsilon_{ijk}
```

where $i=1, \dots ,7$ runs over individuals, $j=1, \dots ,4$ runs over
power and $k=1,2$ runs over eyes. The $p_j$ term is a fixed effect, but
the remaining terms are random. Let $s_i \sim N(0,\sigma^2_s)$,
$e_{ik} \sim N(0,\sigma^2_e)$ and
$\epsilon_{ijk} \sim N(0,\sigma^2\Sigma)$ where we take $\Sigma=I$.

# LME4

``` r
library(lme4)
library(pbkrtest)
```

We can fit the model with:

``` r
mmod <- lmer(acuity~power + (1|subject) + (1|subject:eye),vision)
faraway::sumary(mmod)
```

    Fixed Effects:
                coef.est coef.se
    (Intercept) 112.64     2.23 
    power6/18     0.79     1.54 
    power6/36    -1.00     1.54 
    power6/60     3.29     1.54 

    Random Effects:
     Groups      Name        Std.Dev.
     subject:eye (Intercept) 3.21    
     subject     (Intercept) 4.64    
     Residual                4.07    
    ---
    number of obs: 56, groups: subject:eye, 14; subject, 7
    AIC = 342.7, DIC = 349.6
    deviance = 339.2 

We can check for a power effect using a Kenward-Roger adjusted $F$-test:

``` r
mmod <- lmer(acuity~power+(1|subject)+(1|subject:eye),vision,REML=FALSE)
nmod <- lmer(acuity~1+(1|subject)+(1|subject:eye),vision,REML=FALSE)
KRmodcomp(mmod, nmod)
```

    large : acuity ~ power + (1 | subject) + (1 | subject:eye)
    small : acuity ~ 1 + (1 | subject) + (1 | subject:eye)
           stat   ndf   ddf F.scaling p.value
    Ftest  2.83  3.00 39.00         1   0.051

The power just fails to meet the 5% level of significance (although note
that there is a clear outlier in the data which deserves some
consideration). We can also compute bootstrap confidence intervals:

``` r
set.seed(123)
print(confint(mmod, method="boot", oldNames=FALSE, nsim=1000),digits=3)
```

                                 2.5 % 97.5 %
    sd_(Intercept)|subject:eye   0.172   5.27
    sd_(Intercept)|subject       0.000   6.88
    sigma                        2.943   4.70
    (Intercept)                108.623 116.56
    power6/18                   -1.950   3.94
    power6/36                   -3.868   1.88
    power6/60                    0.388   6.22

We see that lower ends of the CIs for random effect SDs are zero or
close to it.

# NLME

See the discussion for the [single random effect
example](pulpfreq.md#NLME) for some introduction.

The syntax for specifying the nested/heirarchical model is different
from `lme4`:

``` r
library(nlme)
nlmod = lme(acuity ~ power, 
            vision, 
            ~ 1 | subject/eye)
summary(nlmod)
```

    Linear mixed-effects model fit by REML
      Data: vision 
         AIC    BIC  logLik
      342.71 356.37 -164.35

    Random effects:
     Formula: ~1 | subject
            (Intercept)
    StdDev:      4.6396

     Formula: ~1 | eye %in% subject
            (Intercept) Residual
    StdDev:      3.2052   4.0746

    Fixed effects:  acuity ~ power 
                  Value Std.Error DF t-value p-value
    (Intercept) 112.643    2.2349 39  50.401  0.0000
    power6/18     0.786    1.5400 39   0.510  0.6128
    power6/36    -1.000    1.5400 39  -0.649  0.5199
    power6/60     3.286    1.5400 39   2.134  0.0392
     Correlation: 
              (Intr) pw6/18 pw6/36
    power6/18 -0.345              
    power6/36 -0.345  0.500       
    power6/60 -0.345  0.500  0.500

    Standardized Within-Group Residuals:
          Min        Q1       Med        Q3       Max 
    -3.424009 -0.323213  0.010949  0.440812  2.466186 

    Number of Observations: 56
    Number of Groups: 
             subject eye %in% subject 
                   7               14 

The results are presented somewhat differently but match those presented
by `lme4` earlier. We do get p-values for the fixed effects but these
are not so useful.

We can get tests on the fixed effects with:

``` r
anova(nlmod)
```

                numDF denDF F-value p-value
    (Intercept)     1    39 3132.91  <.0001
    power           3    39    2.83  0.0511

In this case, the results are the same as with the `pbkrtest` output
because the degrees of freedom adjustment has no effect. This is not
always the case.

# MMRM

This package is not designed to fit models with a hierarchical
structure. In principle, it should be possible to specify a
parameterised covariance structure corresponding to this design but the
would reduce to the previous computations.

# GLMMTMB

See the discussion for the [single random effect
example](pulpfreq.md#GLMMTMB) for some introduction.

``` r
library(glmmTMB)
```

The default fit uses ML (not REML)

``` r
gtmod <- glmmTMB(acuity~power+(1|subject)+(1|subject:eye),data=vision)
summary(gtmod)
```

     Family: gaussian  ( identity )
    Formula:          acuity ~ power + (1 | subject) + (1 | subject:eye)
    Data: vision

         AIC      BIC   logLik deviance df.resid 
       353.2    367.4   -169.6    339.2       49 

    Random effects:

    Conditional model:
     Groups      Name        Variance Std.Dev.
     subject     (Intercept) 17.4     4.17    
     subject:eye (Intercept) 10.6     3.25    
     Residual                15.4     3.93    
    Number of obs: 56, groups:  subject, 7; subject:eye, 14

    Dispersion estimate for gaussian family (sigma^2): 15.4 

    Conditional model:
                Estimate Std. Error z value Pr(>|z|)
    (Intercept)  112.643      2.084    54.0   <2e-16
    power6/18      0.786      1.484     0.5    0.596
    power6/36     -1.000      1.484    -0.7    0.500
    power6/60      3.286      1.484     2.2    0.027

Another option is to use REML with:

``` r
gtmodr = glmmTMB(acuity~power+(1|subject)+(1|subject:eye),data=vision,
                REML=TRUE)
summary(gtmodr)
```

     Family: gaussian  ( identity )
    Formula:          acuity ~ power + (1 | subject) + (1 | subject:eye)
    Data: vision

         AIC      BIC   logLik deviance df.resid 
       342.7    356.9   -164.4    328.7       53 

    Random effects:

    Conditional model:
     Groups      Name        Variance Std.Dev.
     subject     (Intercept) 21.5     4.64    
     subject:eye (Intercept) 10.3     3.21    
     Residual                16.6     4.07    
    Number of obs: 56, groups:  subject, 7; subject:eye, 14

    Dispersion estimate for gaussian family (sigma^2): 16.6 

    Conditional model:
                Estimate Std. Error z value Pr(>|z|)
    (Intercept)  112.643      2.235    50.4   <2e-16
    power6/18      0.786      1.540     0.5    0.610
    power6/36     -1.000      1.540    -0.6    0.516
    power6/60      3.286      1.540     2.1    0.033

The result is appears identical with the previous REML fits.

If we want to test the significance of the `power` fixed effect, we have
the same methods available as for the `lme4` fit.

# Discussion

See the [Discussion of the single random effect
model](pulp.md#Discussion) for general comments.

`lme4`, `nlme` and `glmmTMB` were all able to fit this model. `mmrm` was
not designed for this type of model.

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
    [1] glmmTMB_1.1.9  nlme_3.1-166   pbkrtest_0.5.3 lme4_1.1-35.5  Matrix_1.7-0   ggplot2_3.5.1  faraway_1.0.8 

    loaded via a namespace (and not attached):
     [1] utf8_1.2.4          generics_0.1.3      tidyr_1.3.1         lattice_0.22-6      digest_0.6.37      
     [6] magrittr_2.0.3      estimability_1.5.1  evaluate_0.24.0     grid_4.4.1          mvtnorm_1.2-6      
    [11] fastmap_1.2.0       jsonlite_1.8.8      backports_1.5.0     mgcv_1.9-1          purrr_1.0.2        
    [16] fansi_1.0.6         scales_1.3.0        numDeriv_2016.8-1.1 cli_3.6.3           rlang_1.1.4        
    [21] munsell_0.5.1       splines_4.4.1       withr_3.0.1         yaml_2.3.10         tools_4.4.1        
    [26] parallel_4.4.1      coda_0.19-4.1       nloptr_2.1.1        minqa_1.2.8         dplyr_1.1.4        
    [31] colorspace_2.1-1    boot_1.3-31         broom_1.0.6         vctrs_0.6.5         R6_2.5.1           
    [36] emmeans_1.10.4      lifecycle_1.0.4     MASS_7.3-61         pkgconfig_2.0.3     pillar_1.9.0       
    [41] gtable_0.3.5        glue_1.7.0          Rcpp_1.0.13         systemfonts_1.1.0   xfun_0.47          
    [46] tibble_3.2.1        tidyselect_1.2.1    rstudioapi_0.16.0   knitr_1.48          xtable_1.8-4       
    [51] farver_2.1.2        htmltools_0.5.8.1   rmarkdown_2.28      svglite_2.1.3       labeling_0.4.3     
    [56] TMB_1.9.14          compiler_4.4.1     
