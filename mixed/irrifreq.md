# Split Plot Design
[Julian Faraway](https://julianfaraway.github.io/)
2024-08-23

- [Data](#data)
- [Mixed Effect Model](#mixed-effect-model)
- [LME4](#lme4)
- [NLME](#nlme)
- [MMRM](#mmrm)
- [GLMMTMB](#glmmtmb)
- [Discussion](#discussion)
- [Package version info](#package-version-info)

See the [introduction](index.md) for an overview.

See a [mostly Bayesian analysis](irrigation.md) analysis of the same
data.

This example is discussed in more detail in my book [Extending the
Linear Model with R](https://julianfaraway.github.io/faraway/ELM/)

``` r
library(faraway)
library(ggplot2)
```

# Data

In an agricultural field trial, the objective was to determine the
effects of two crop varieties and four different irrigation methods.
Eight fields were available, but only one type of irrigation may be
applied to each field. The fields may be divided into two parts with a
different variety planted in each half. The whole plot factor is the
method of irrigation, which should be randomly assigned to the fields.
Within each field, the variety is randomly assigned.

Load in and plot the data:

``` r
data(irrigation, package="faraway")
summary(irrigation)
```

         field   irrigation variety     yield     
     f1     :2   i1:4       v1:8    Min.   :34.8  
     f2     :2   i2:4       v2:8    1st Qu.:37.6  
     f3     :2   i3:4               Median :40.1  
     f4     :2   i4:4               Mean   :40.2  
     f5     :2                      3rd Qu.:42.7  
     f6     :2                      Max.   :47.6  
     (Other):4                                    

``` r
ggplot(irrigation, aes(y=yield, x=field, shape=variety, color=irrigation)) + geom_point()
```

![](figs/irriplot-1..svg)

# Mixed Effect Model

The irrigation and variety are fixed effects, but the field is a random
effect. We must also consider the interaction between field and variety,
which is necessarily also a random effect because one of the two
components is random. The fullest model that we might consider is:

``` math
y_{ijk} = \mu + i_i + v_j + (iv)_{ij} + f_k + (vf)_{jk} + \epsilon_{ijk}
```

where $\mu, i_i, v_j, (iv)_{ij}$ are fixed effects; the rest are random
having variances $\sigma^2_f$, $\sigma^2_{vf}$ and $\sigma^2_\epsilon$.
Note that we have no $(if)_{ik}$ term in this model. It would not be
possible to estimate such an effect since only one type of irrigation is
used on a given field; the factors are not crossed. Unfortunately, it is
not possible to distinguish the variety within the field variation. We
would need more than one observation per variety within each field for
us to separate the two variabilities. (This means that we are not
demonstrating split plot modeling in the wider sense). We resort to a
simpler model that omits the variety by field interaction random effect:

``` math
y_{ijk} = \mu + i_i + v_j + (iv)_{ij} + f_k +  \epsilon_{ijk}
```

# LME4

See the discussion for the [single random effect
example](pulpfreq.md#LME4) for some introduction.

We fit this model with:

``` r
library(lme4)
lmod4 = lmer(yield ~ irrigation * variety + (1|field), irrigation)
summary(lmod4, cor=FALSE)
```

    Linear mixed model fit by REML ['lmerMod']
    Formula: yield ~ irrigation * variety + (1 | field)
       Data: irrigation

    REML criterion at convergence: 45.4

    Scaled residuals: 
       Min     1Q Median     3Q    Max 
    -0.745 -0.551  0.000  0.551  0.745 

    Random effects:
     Groups   Name        Variance Std.Dev.
     field    (Intercept) 16.20    4.02    
     Residual              2.11    1.45    
    Number of obs: 16, groups:  field, 8

    Fixed effects:
                           Estimate Std. Error t value
    (Intercept)               38.50       3.03   12.73
    irrigationi2               1.20       4.28    0.28
    irrigationi3               0.70       4.28    0.16
    irrigationi4               3.50       4.28    0.82
    varietyv2                  0.60       1.45    0.41
    irrigationi2:varietyv2    -0.40       2.05   -0.19
    irrigationi3:varietyv2    -0.20       2.05   -0.10
    irrigationi4:varietyv2     1.20       2.05    0.58

The fixed effects don’t look very significant. For testing the fixed
effects, we might try:

``` r
anova(lmod4)
```

    Analysis of Variance Table
                       npar Sum Sq Mean Sq F value
    irrigation            3   2.46   0.818    0.39
    variety               1   2.25   2.250    1.07
    irrigation:variety    3   1.55   0.517    0.25

The small values of the F-statistics suggest a lack of significance but
no p-values or degrees of freedom for the error are supplied due to
previously mentioned concerns regarding correctness.

We use the `pbkrtest` package for testing which implements the
Kenward-Roger approximation for the degrees of freedom. First test the
interaction:

``` r
library(pbkrtest)
lmoda = lmer(yield ~ irrigation + variety + (1|field),data=irrigation)
summary(lmoda, cor=FALSE)
```

    Linear mixed model fit by REML ['lmerMod']
    Formula: yield ~ irrigation + variety + (1 | field)
       Data: irrigation

    REML criterion at convergence: 54.8

    Scaled residuals: 
       Min     1Q Median     3Q    Max 
    -0.875 -0.561 -0.109  0.689  1.093 

    Random effects:
     Groups   Name        Variance Std.Dev.
     field    (Intercept) 16.54    4.07    
     Residual              1.43    1.19    
    Number of obs: 16, groups:  field, 8

    Fixed effects:
                 Estimate Std. Error t value
    (Intercept)    38.425      2.952   13.02
    irrigationi2    1.000      4.154    0.24
    irrigationi3    0.600      4.154    0.14
    irrigationi4    4.100      4.154    0.99
    varietyv2       0.750      0.597    1.26

``` r
KRmodcomp(lmod4, lmoda)
```

    large : yield ~ irrigation * variety + (1 | field)
    small : yield ~ irrigation + variety + (1 | field)
          stat  ndf  ddf F.scaling p.value
    Ftest 0.25 3.00 4.00         1    0.86

The interaction is not significant. Does the variety make a difference
to the yield? We fit a model with no variety effect:

``` r
lmodi <- lmer(yield ~ irrigation + (1|field), irrigation)
```

We have some choices now. One idea is to compare this model to the main
effects only model with:

``` r
KRmodcomp(lmoda,lmodi)
```

    large : yield ~ irrigation + variety + (1 | field)
    small : yield ~ irrigation + (1 | field)
          stat  ndf  ddf F.scaling p.value
    Ftest 1.58 1.00 7.00         1    0.25

This assumes that the interaction effect is zero and absorbs all the
variation and degrees of freedom into the error term. Although we did
not find a significant interaction effect, this is a stronger assumption
and we may not have the best estimate of $\sigma^2$ for the denominator
in the F-test.

Another idea is to compare to the full model with:

``` r
KRmodcomp(lmod4,lmodi)
```

    large : yield ~ irrigation * variety + (1 | field)
    small : yield ~ irrigation + (1 | field)
          stat  ndf  ddf F.scaling p.value
    Ftest 0.45 4.00 4.00         1    0.77

The problem with this is that we are not just testing the variety
effect. This is testing the variety effect and its interaction with
irrigation. That’s OK but not what we meant to test.

We can take the ANOVA approach to testing the variety but we cannot
construct this a model comparison. The `lme4` package did provide us
with the ANOVA table. We can get the (possibly adjusted) degrees of
freedom from `pbkrtest` output as 4, get the F-statistic from the ANOVA
table and compute the p-value using the F-distribution (the numerator DF
also comes from the table):

``` r
1-pf(1.07,1,4)
```

    [1] 0.35938

A not significant result. In this case, all the nitpicking does not make
much difference.

Now check the irrigation method using the same ANOVA-based approach:

``` r
1-pf(0.39,3,4)
```

    [1] 0.7674

We find no significant irrigation effect.

As a final check, lets compare the null model with no fixed effects to
the full model.

``` r
lmodn <- lmer(yield ~  1 + (1|field), irrigation)
KRmodcomp(lmod4, lmodn)
```

    large : yield ~ irrigation * variety + (1 | field)
    small : yield ~ 1 + (1 | field)
          stat  ndf  ddf F.scaling p.value
    Ftest 0.38 7.00 4.48     0.903    0.88

This confirms the lack of statistical significance for the variety and
irrigation factors.

We can check the significance of the random effect (field) term with:

``` r
library(RLRsim)
exactRLRT(lmod4)
```


        simulated finite sample distribution of RLRT.
        
        (p-value based on 10000 simulated values)

    data:  
    RLRT = 6.11, p-value = 0.0075

We can see that there is a significant variation among the fields.

# NLME

See the discussion for the [single random effect
example](pulpfreq.md#NLME) for some introduction.

``` r
library(nlme)
```

``` r
nlmod = lme(yield ~ irrigation * variety, irrigation, ~ 1 | field)
summary(nlmod)
```

    Linear mixed-effects model fit by REML
      Data: irrigation 
         AIC    BIC  logLik
      65.395 66.189 -22.697

    Random effects:
     Formula: ~1 | field
            (Intercept) Residual
    StdDev:      4.0249   1.4517

    Fixed effects:  yield ~ irrigation * variety 
                           Value Std.Error DF t-value p-value
    (Intercept)             38.5    3.0255  4 12.7251  0.0002
    irrigationi2             1.2    4.2787  4  0.2805  0.7930
    irrigationi3             0.7    4.2787  4  0.1636  0.8780
    irrigationi4             3.5    4.2787  4  0.8180  0.4593
    varietyv2                0.6    1.4517  4  0.4133  0.7006
    irrigationi2:varietyv2  -0.4    2.0530  4 -0.1948  0.8550
    irrigationi3:varietyv2  -0.2    2.0530  4 -0.0974  0.9271
    irrigationi4:varietyv2   1.2    2.0530  4  0.5845  0.5903
     Correlation: 
                           (Intr) irrgt2 irrgt3 irrgt4 vrtyv2 irr2:2 irr3:2
    irrigationi2           -0.707                                          
    irrigationi3           -0.707  0.500                                   
    irrigationi4           -0.707  0.500  0.500                            
    varietyv2              -0.240  0.170  0.170  0.170                     
    irrigationi2:varietyv2  0.170 -0.240 -0.120 -0.120 -0.707              
    irrigationi3:varietyv2  0.170 -0.120 -0.240 -0.120 -0.707  0.500       
    irrigationi4:varietyv2  0.170 -0.120 -0.120 -0.240 -0.707  0.500  0.500

    Standardized Within-Group Residuals:
            Min          Q1         Med          Q3         Max 
    -7.4484e-01 -5.5094e-01  4.9127e-15  5.5094e-01  7.4484e-01 

    Number of Observations: 16
    Number of Groups: 8 

The estimates and standard errors are the same for the corresponding
`lme4` output. `nlme` also p-values for the fixed effects but these are
not useful to us here. We can test the fixed with:

``` r
anova(nlmod)
```

                       numDF denDF F-value p-value
    (Intercept)            1     4  750.24  <.0001
    irrigation             3     4    0.39  0.7685
    variety                1     4    1.07  0.3599
    irrigation:variety     3     4    0.25  0.8612

The result for the interaction corresponds to the calculation for `lme4`
using `pbkrtest`. The tests for the main effects are the same as the
directly calculated p-values above. This clarifies what is being tested
in the ANOVA table.

We can also take the `gls` approach with:

``` r
gmod = gls(yield ~ irrigation * variety ,
           data= irrigation,
           correlation = corCompSymm(form = ~ 1|field))
summary(gmod)
```

    Generalized least squares fit by REML
      Model: yield ~ irrigation * variety 
      Data: irrigation 
         AIC    BIC  logLik
      65.395 66.189 -22.697

    Correlation Structure: Compound symmetry
     Formula: ~1 | field 
     Parameter estimate(s):
        Rho 
    0.88488 

    Coefficients:
                           Value Std.Error t-value p-value
    (Intercept)             38.5    3.0255 12.7251  0.0000
    irrigationi2             1.2    4.2787  0.2805  0.7862
    irrigationi3             0.7    4.2787  0.1636  0.8741
    irrigationi4             3.5    4.2787  0.8180  0.4370
    varietyv2                0.6    1.4517  0.4133  0.6902
    irrigationi2:varietyv2  -0.4    2.0530 -0.1948  0.8504
    irrigationi3:varietyv2  -0.2    2.0530 -0.0974  0.9248
    irrigationi4:varietyv2   1.2    2.0530  0.5845  0.5750

     Correlation: 
                           (Intr) irrgt2 irrgt3 irrgt4 vrtyv2 irr2:2 irr3:2
    irrigationi2           -0.707                                          
    irrigationi3           -0.707  0.500                                   
    irrigationi4           -0.707  0.500  0.500                            
    varietyv2              -0.240  0.170  0.170  0.170                     
    irrigationi2:varietyv2  0.170 -0.240 -0.120 -0.120 -0.707              
    irrigationi3:varietyv2  0.170 -0.120 -0.240 -0.120 -0.707  0.500       
    irrigationi4:varietyv2  0.170 -0.120 -0.120 -0.240 -0.707  0.500  0.500

    Standardized residuals:
            Min          Q1         Med          Q3         Max 
    -1.0283e+00 -7.0699e-01  4.1356e-15  7.0699e-01  1.0283e+00 

    Residual standard error: 4.2787 
    Degrees of freedom: 16 total; 8 residual

We attempt a test for the fixed effects with:

``` r
anova(gmod)
```

    Denom. DF: 8 
                       numDF F-value p-value
    (Intercept)            1  750.24  <.0001
    irrigation             3    0.39  0.7647
    variety                1    1.07  0.3317
    irrigation:variety     3    0.25  0.8625

But the denominator DFs are wrong - should be 4 and not 8 as discussed
earlier. We could fix the p-values manually.

# MMRM

See the discussion for the [single random effect
example](pulpfreq.md#MMRM) for some introduction.

``` r
library(mmrm)
```

As discussed in the `pulp` example, we need to create a `visit` factor
to distinguish the replicates within the fields:

``` r
irrigation$visit = factor(rep(1:2,8))
```

``` r
mmmod = mmrm(yield ~ irrigation * variety + cs(visit|field), irrigation)
summary(mmmod)
```

    mmrm fit

    Formula:     yield ~ irrigation * variety + cs(visit | field)
    Data:        irrigation (used 16 observations from 8 subjects with maximum 2 timepoints)
    Covariance:  compound symmetry (2 variance parameters)
    Method:      Satterthwaite
    Vcov Method: Asymptotic
    Inference:   REML

    Model selection criteria:
         AIC      BIC   logLik deviance 
        49.4     49.6    -22.7     45.4 

    Coefficients: 
                           Estimate Std. Error    df t value Pr(>|t|)
    (Intercept)               38.50       3.03  4.49   12.72  0.00011
    irrigationi2               1.20       4.28  4.49    0.28  0.79159
    irrigationi3               0.70       4.28  4.49    0.16  0.87716
    irrigationi4               3.50       4.28  4.49    0.82  0.45459
    varietyv2                  0.60       1.45  4.00    0.41  0.70058
    irrigationi2:varietyv2    -0.40       2.05  4.00   -0.19  0.85502
    irrigationi3:varietyv2    -0.20       2.05  4.00   -0.10  0.92708
    irrigationi4:varietyv2     1.20       2.05  4.00    0.58  0.59027

    Covariance estimate:
           1      2
    1 18.308 16.200
    2 16.200 18.308

The random component is expressed as a covariance matrix. The SD down
the diagonal will give the estimated error variance from previous
models:

``` r
cm = mmmod$cov
sqrt(cm[1,1])
```

    [1] 4.2788

We can compute the correlation as:

``` r
cm[1,2]/cm[1,1]
```

    [1] 0.88488

This agrees with the GLS output as expected.

We can test the fixed effects with:

``` r
library(car)
Anova(mmmod)
```

    Analysis of Fixed Effect Table (Type II F tests)
                       Num Df Denom Df F Statistic Pr(>=F)
    irrigation              3        4       0.388    0.77
    variety                 1        4       1.068    0.36
    irrigation:variety      3        4       0.245    0.86

and get the same results as previously.

# GLMMTMB

See the discussion for the [single random effect
example](pulpfreq.md#GLMMTMB) for some introduction.

``` r
library(glmmTMB)
```

The default fit uses ML (not REML)

``` r
gtmod <- glmmTMB(yield ~ irrigation*variety + (1|field), irrigation)
summary(gtmod)
```

     Family: gaussian  ( identity )
    Formula:          yield ~ irrigation * variety + (1 | field)
    Data: irrigation

         AIC      BIC   logLik deviance df.resid 
        88.6     96.3    -34.3     68.6        6 

    Random effects:

    Conditional model:
     Groups   Name        Variance Std.Dev.
     field    (Intercept) 8.10     2.85    
     Residual             1.05     1.03    
    Number of obs: 16, groups:  field, 8

    Dispersion estimate for gaussian family (sigma^2): 1.05 

    Conditional model:
                           Estimate Std. Error z value Pr(>|z|)
    (Intercept)               38.50       2.14   18.00   <2e-16
    irrigationi2               1.20       3.03    0.40     0.69
    irrigationi3               0.70       3.03    0.23     0.82
    irrigationi4               3.50       3.03    1.16     0.25
    varietyv2                  0.60       1.03    0.58     0.56
    irrigationi2:varietyv2    -0.40       1.45   -0.28     0.78
    irrigationi3:varietyv2    -0.20       1.45   -0.14     0.89
    irrigationi4:varietyv2     1.20       1.45    0.83     0.41

This is identical with the `lme4` fit using ML.

We can use the `car` package to test the treatment effects:

``` r
Anova(gtmod)
```

    Analysis of Deviance Table (Type II Wald chisquare tests)

    Response: yield
                       Chisq Df Pr(>Chisq)
    irrigation          2.33  3       0.51
    variety             2.14  1       0.14
    irrigation:variety  1.47  3       0.69

but this gives chi-square tests whereas we prefer F-tests.

# Discussion

No new issues are raised by this analysis. There are some choices with
the execution of the tests of the fixed effects but these are not unique
to this type of example.

In the [Bayesian analyses of this data](irrigation.md), there was more
analysis of the random effects but there’s not much we can do with these
in the Frequentist analyses so there’s nothing to be said.

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
     [1] glmmTMB_1.1.9  car_3.1-2      carData_3.0-5  mmrm_0.3.12    nlme_3.1-165   RLRsim_3.1-8   pbkrtest_0.5.3
     [8] lme4_1.1-35.5  Matrix_1.7-0   ggplot2_3.5.1  faraway_1.0.8 

    loaded via a namespace (and not attached):
     [1] gtable_0.3.5        TMB_1.9.14          xfun_0.46           lattice_0.22-6      numDeriv_2016.8-1.1
     [6] vctrs_0.6.5         tools_4.4.1         Rdpack_2.6          generics_0.1.3      parallel_4.4.1     
    [11] tibble_3.2.1        fansi_1.0.6         pkgconfig_2.0.3     checkmate_2.3.2     lifecycle_1.0.4    
    [16] compiler_4.4.1      farver_2.1.2        stringr_1.5.1       munsell_0.5.1       htmltools_0.5.8.1  
    [21] yaml_2.3.10         pillar_1.9.0        nloptr_2.1.1        tidyr_1.3.1         MASS_7.3-61        
    [26] boot_1.3-30         abind_1.4-5         tidyselect_1.2.1    digest_0.6.36       mvtnorm_1.2-5      
    [31] stringi_1.8.4       dplyr_1.1.4         purrr_1.0.2         labeling_0.4.3      splines_4.4.1      
    [36] fastmap_1.2.0       grid_4.4.1          colorspace_2.1-1    cli_3.6.3           magrittr_2.0.3     
    [41] utf8_1.2.4          broom_1.0.6         withr_3.0.1         scales_1.3.0        backports_1.5.0    
    [46] estimability_1.5.1  rmarkdown_2.27      emmeans_1.10.3      coda_0.19-4.1       evaluate_0.24.0    
    [51] knitr_1.48          rbibutils_2.2.16    mgcv_1.9-1          rlang_1.1.4         Rcpp_1.0.13        
    [56] xtable_1.8-4        glue_1.7.0          svglite_2.1.3       rstudioapi_0.16.0   minqa_1.2.7        
    [61] jsonlite_1.8.8      R6_2.5.1            systemfonts_1.1.0  
