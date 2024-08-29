# Crossed Effects Design using Frequentist Methods
[Julian Faraway](https://julianfaraway.github.io/)
2024-08-29

- [Data](#data)
- [LME4](#lme4)
- [NLME](#nlme)
- [MMRM](#mmrm)
- [GLMMTMB](#glmmtmb)
- [Discussion](#discussion)
- [Package version info](#package-version-info)

See the [introduction](index.md) for an overview.

See a [mostly Bayesian analysis](abrasion.md) analysis of the same data.

This example is discussed in more detail in my book [Extending the
Linear Model with R](https://julianfaraway.github.io/faraway/ELM/)

Required libraries:

``` r
library(faraway)
library(ggplot2)
```

# Data

Effects are said to be crossed when they are not nested. When at least
some crossing occurs, methods for nested designs cannot be used. We
consider a latin square example.

In an experiment, four materials, A, B, C and D, were fed into a
wear-testing machine. The response is the loss of weight in 0.1 mm over
the testing period. The machine could process four samples at a time and
past experience indicated that there were some differences due to the
position of these four samples. Also some differences were suspected
from run to run. Four runs were made. The latin square structure of the
design may be observed:

``` r
data(abrasion, package="faraway")
matrix(abrasion$material,4,4)
```

         [,1] [,2] [,3] [,4]
    [1,] "C"  "A"  "D"  "B" 
    [2,] "D"  "B"  "C"  "A" 
    [3,] "B"  "D"  "A"  "C" 
    [4,] "A"  "C"  "B"  "D" 

We can plot the data

``` r
ggplot(abrasion,aes(x=material, y=wear, shape=run, color=position))+geom_point(position = position_jitter(width=0.1, height=0.0))
```

![](figs/abrasionplot-1..svg)

# LME4

See the discussion for the [single random effect
example](pulpfreq.md#LME4) for some introduction.

Since we are most interested in the choice of material, treating this as
a fixed effect is natural. We must account for variation due to the run
and the position but were not interested in their specific values
because we believe these may vary between experiments. We treat these as
random effects.

We fit this model with:

``` r
library(lme4)
mmod <- lmer(wear ~ material + (1|run) + (1|position), abrasion)
summary(mmod, cor=FALSE)
```

    Linear mixed model fit by REML ['lmerMod']
    Formula: wear ~ material + (1 | run) + (1 | position)
       Data: abrasion

    REML criterion at convergence: 100.3

    Scaled residuals: 
       Min     1Q Median     3Q    Max 
    -1.090 -0.302  0.027  0.422  1.210 

    Random effects:
     Groups   Name        Variance Std.Dev.
     run      (Intercept)  66.9     8.18   
     position (Intercept) 107.1    10.35   
     Residual              61.3     7.83   
    Number of obs: 16, groups:  run, 4; position, 4

    Fixed effects:
                Estimate Std. Error t value
    (Intercept)   265.75       7.67   34.66
    materialB     -45.75       5.53   -8.27
    materialC     -24.00       5.53   -4.34
    materialD     -35.25       5.53   -6.37

We test the random effects:

``` r
library(RLRsim)
mmodp <- lmer(wear ~ material + (1|position), abrasion)
mmodr <- lmer(wear ~ material + (1|run), abrasion)
exactRLRT(mmodp, mmod, mmodr)
```


        simulated finite sample distribution of RLRT.
        
        (p-value based on 10000 simulated values)

    data:  
    RLRT = 4.59, p-value = 0.013

``` r
exactRLRT(mmodr, mmod, mmodp)
```


        simulated finite sample distribution of RLRT.
        
        (p-value based on 10000 simulated values)

    data:  
    RLRT = 3.05, p-value = 0.034

We see both are statistically significant.

We can test the fixed effect:

``` r
library(pbkrtest)
mmod <- lmer(wear ~ material + (1|run) + (1|position), abrasion,REML=FALSE)
nmod <- lmer(wear ~ 1+ (1|run) + (1|position), abrasion,REML=FALSE)
KRmodcomp(mmod, nmod)
```

    large : wear ~ material + (1 | run) + (1 | position)
    small : wear ~ 1 + (1 | run) + (1 | position)
          stat  ndf  ddf F.scaling p.value
    Ftest 25.1  3.0  6.0         1 0.00085

We see the fixed effect is significant.

# NLME

See the discussion for the [single random effect
example](pulpfreq.md#NLME) for some introduction.

The short answer is that `nlme` is not designed to fit crossed effects
and you should use `lme4`. But it is possible - as explained in this
[StackOverflow answer by Ben
Bolker](https://stackoverflow.com/a/38805602)

``` r
library(nlme)
abrasion$dummy = factor(1)
nlmod = lme(wear ~ material,
          random=list(dummy =
                pdBlocked(list(pdIdent(~run-1),
                               pdIdent(~position-1)))),
          data=abrasion)
nlmod
```

    Linear mixed-effects model fit by REML
      Data: abrasion 
      Log-restricted-likelihood: -50.128
      Fixed: wear ~ material 
    (Intercept)   materialB   materialC   materialD 
         265.75      -45.75      -24.00      -35.25 

    Random effects:
     Composite Structure: Blocked

     Block 1: run1, run2, run3, run4
     Formula: ~run - 1 | dummy
     Structure: Multiple of an Identity
             run1  run2  run3  run4
    StdDev: 8.179 8.179 8.179 8.179

     Block 2: position1, position2, position3, position4
     Formula: ~position - 1 | dummy
     Structure: Multiple of an Identity
            position1 position2 position3 position4 Residual
    StdDev:    10.347    10.347    10.347    10.347   7.8262

    Number of Observations: 16
    Number of Groups: 1 

The output contains the fixed and random effect estimates as found with
`lme4` albeit presented in an unfamiliar way. Of course, it is much
easier to just use `lme4`.

# MMRM

See the discussion for the [single random effect
example](pulpfreq.md#MMRM) for some introduction.

`mmrm` is not designed to handle crossed effects.

# GLMMTMB

See the discussion for the [single random effect
example](pulpfreq.md#GLMMTMB) for some introduction.

``` r
library(glmmTMB)
```

The default fit uses ML (not REML)

``` r
gtmod = glmmTMB(wear ~ material + (1|run) + (1|position), abrasion)
summary(gtmod)
```

     Family: gaussian  ( identity )
    Formula:          wear ~ material + (1 | run) + (1 | position)
    Data: abrasion

         AIC      BIC   logLik deviance df.resid 
       134.3    139.7    -60.2    120.3        9 

    Random effects:

    Conditional model:
     Groups   Name        Variance Std.Dev.
     run      (Intercept) 61.4     7.84    
     position (Intercept) 91.1     9.54    
     Residual             41.1     6.41    
    Number of obs: 16, groups:  run, 4; position, 4

    Dispersion estimate for gaussian family (sigma^2): 41.1 

    Conditional model:
                Estimate Std. Error z value Pr(>|z|)
    (Intercept)   265.75       6.96    38.2  < 2e-16
    materialB     -45.75       4.53   -10.1  < 2e-16
    materialC     -24.00       4.53    -5.3  1.2e-07
    materialD     -35.25       4.53    -7.8  7.6e-15

This is identical with the `lme4` fit using ML.

# Discussion

The `lme4` package is the obvious choice for this model type. Although
`nlme` can be tricked into fitting the model, it’s not convenient.
`mmrm` was not designed with this model type in mind. `glmmTMB` would be
valuable for less common response types but is aligned with `lme4` in
this instance.

# Package version info

``` r
sessionInfo()
```

    R version 4.4.1 (2024-06-14)
    Platform: x86_64-apple-darwin20
    Running under: macOS Sonoma 14.6.1

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
    [1] glmmTMB_1.1.9  nlme_3.1-166   pbkrtest_0.5.3 RLRsim_3.1-8   lme4_1.1-35.5  Matrix_1.7-0   ggplot2_3.5.1 
    [8] faraway_1.0.8 

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
