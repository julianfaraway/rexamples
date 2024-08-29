# Multilevel Design
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

See a [mostly Bayesian analysis](jspmultilevel.md) analysis of the same
data.

This example is discussed in more detail in my book [Extending the
Linear Model with R](https://julianfaraway.github.io/faraway/ELM/)

Required libraries:

``` r
library(faraway)
library(ggplot2)
```

# Data

*Multilevel* models is a term used for models for data with hierarchical
structure. The term is most commonly used in the social sciences. We can
use the methodology we have already developed to fit some of these
models.

We take as our example some data from the Junior School Project
collected from primary (U.S. term is elementary) schools in inner
London. We math test score result from year two as the response and try
to model this as a function of gender, social class and the Raven’s test
score from the first year which might be taken as a measure of ability
when entering the school. We subset the data to ignore the math scores
from the first two years, we centre the Raven score and create a
combined class-by-school label:

``` r
data(jsp, package="faraway")
jspr <- jsp[jsp$year==2,]
jspr$craven <- jspr$raven-mean(jspr$raven)
jspr$classch <- paste(jspr$school,jspr$class,sep=".")
```

We can plot the data

``` r
ggplot(jspr, aes(x=raven, y=math))+xlab("Raven Score")+ylab("Math Score")+geom_point(position = position_jitter())
```

![](figs/jspplot-1..svg)

``` r
ggplot(jspr, aes(x=social, y=math))+xlab("Social Class")+ylab("Math Score")+geom_boxplot()
```

![](figs/jspplot-2..svg)

Although the data supports a more complex model, we simplify to having
the centred Raven score and the social class as fixed effects and the
school and class nested within school as random effects. See [Extending
the Linear Model with R](https://julianfaraway.github.io/faraway/ELM/),

# LME4

See the discussion for the [single random effect
example](pulpfreq.md#LME4) for some introduction.

``` r
library(lme4)
```

    Loading required package: Matrix

``` r
mmod = lmer(math ~ craven + social+(1|school)+(1|school:class),jspr)
summary(mmod, cor=FALSE)
```

    Linear mixed model fit by REML ['lmerMod']
    Formula: math ~ craven + social + (1 | school) + (1 | school:class)
       Data: jspr

    REML criterion at convergence: 5923.7

    Scaled residuals: 
       Min     1Q Median     3Q    Max 
    -3.943 -0.548  0.147  0.631  2.853 

    Random effects:
     Groups       Name        Variance Std.Dev.
     school:class (Intercept)  1.03    1.02    
     school       (Intercept)  3.23    1.80    
     Residual                 27.57    5.25    
    Number of obs: 953, groups:  school:class, 90; school, 48

    Fixed effects:
                Estimate Std. Error t value
    (Intercept)  32.0108     1.0350   30.93
    craven        0.5841     0.0321   18.21
    social2      -0.3611     1.0948   -0.33
    social3      -0.7768     1.1649   -0.67
    social4      -2.1197     1.0396   -2.04
    social5      -1.3632     1.1585   -1.18
    social6      -2.3703     1.2330   -1.92
    social7      -3.0482     1.2703   -2.40
    social8      -3.5473     1.7027   -2.08
    social9      -0.8864     1.1031   -0.80

We can see the math score is strongly related to the entering Raven
score. We see that the math score tends to be lower as social class goes
down. We also see the most substantial variation at the individual level
with smaller amounts of variation at the school and class level.

We test the random effects:

``` r
library(RLRsim)
mmodc <- lmer(math ~ craven + social+(1|school:class),jspr)
mmods <- lmer(math ~ craven + social+(1|school),jspr)
exactRLRT(mmodc, mmod, mmods)
```


        simulated finite sample distribution of RLRT.
        
        (p-value based on 10000 simulated values)

    data:  
    RLRT = 1.85, p-value = 0.077

``` r
exactRLRT(mmods, mmod, mmodc)
```


        simulated finite sample distribution of RLRT.
        
        (p-value based on 10000 simulated values)

    data:  
    RLRT = 7.64, p-value = 0.0021

The first test is for the class effect which fails to meet the 5%
significance level. The second test is for the school effect and shows
strong evidence of differences between schools.

We can test the social fixed effect:

``` r
library(pbkrtest)
mmodm <- lmer(math ~ craven + (1|school)+(1|school:class),jspr)
KRmodcomp(mmod, mmodm)
```

    large : math ~ craven + social + (1 | school) + (1 | school:class)
    small : math ~ craven + (1 | school) + (1 | school:class)
            stat    ndf    ddf F.scaling p.value
    Ftest   2.76   8.00 930.34         1  0.0052

We see the social effect is significant.

We can compute confidence intervals for the parameters:

``` r
confint(mmod, method="boot")
```

    Computing bootstrap confidence intervals ...


    74 message(s): boundary (singular) fit: see help('isSingular')

                   2.5 %    97.5 %
    .sig01       0.00000  1.801269
    .sig02       0.83958  2.379912
    .sigma       5.01346  5.504037
    (Intercept) 30.20470 34.270570
    craven       0.52172  0.640309
    social2     -2.49560  1.551274
    social3     -3.21533  1.372575
    social4     -4.15223 -0.092000
    social5     -3.59398  0.815693
    social6     -4.71045 -0.022172
    social7     -5.31468 -0.512631
    social8     -7.08861 -0.285387
    social9     -2.99426  1.081667

The lower end of the class confidence interval is zero while the school
random effect is clearly larger. This is consistent with the earlier
tests.

# NLME

See the discussion for the [single random effect
example](pulpfreq.md#NLME) for some introduction.

The syntax for specifying the nested/heirarchical model is different
from `lme4`:

``` r
library(nlme)
```


    Attaching package: 'nlme'

    The following object is masked from 'package:lme4':

        lmList

``` r
nlmod = lme(math ~ craven + social, 
            jspr, 
            ~ 1 | school/class)
summary(nlmod)
```

    Linear mixed-effects model fit by REML
      Data: jspr 
         AIC    BIC  logLik
      5949.7 6012.7 -2961.8

    Random effects:
     Formula: ~1 | school
            (Intercept)
    StdDev:      1.7968

     Formula: ~1 | class %in% school
            (Intercept) Residual
    StdDev:       1.016   5.2509

    Fixed effects:  math ~ craven + social 
                 Value Std.Error  DF t-value p-value
    (Intercept) 32.011   1.03499 854 30.9285  0.0000
    craven       0.584   0.03208 854 18.2053  0.0000
    social2     -0.361   1.09477 854 -0.3298  0.7416
    social3     -0.777   1.16489 854 -0.6668  0.5051
    social4     -2.120   1.03963 854 -2.0389  0.0418
    social5     -1.363   1.15850 854 -1.1767  0.2396
    social6     -2.370   1.23302 854 -1.9224  0.0549
    social7     -3.048   1.27027 854 -2.3997  0.0166
    social8     -3.547   1.70273 854 -2.0833  0.0375
    social9     -0.886   1.10314 854 -0.8035  0.4219
     Correlation: 
            (Intr) craven socil2 socil3 socil4 socil5 socil6 socil7 socil8
    craven  -0.078                                                        
    social2 -0.855 -0.001                                                 
    social3 -0.818  0.058  0.763                                          
    social4 -0.923  0.086  0.856  0.820                                   
    social5 -0.825  0.089  0.764  0.726  0.825                            
    social6 -0.784  0.088  0.720  0.695  0.787  0.698                     
    social7 -0.765  0.091  0.699  0.678  0.765  0.685  0.652              
    social8 -0.560  0.054  0.521  0.501  0.560  0.498  0.471  0.465       
    social9 -0.871  0.089  0.803  0.772  0.872  0.776  0.742  0.723  0.524

    Standardized Within-Group Residuals:
         Min       Q1      Med       Q3      Max 
    -3.94276 -0.54831  0.14712  0.63089  2.85260 

    Number of Observations: 953
    Number of Groups: 
               school class %in% school 
                   48                90 

The results are presented somewhat differently but match those presented
by `lme4` earlier. We do get p-values for the fixed effects but these
are only useful for `craven` and not so much for `social` as it has 9
levels.

We can get tests on the fixed effects with:

``` r
anova(nlmod)
```

                numDF denDF F-value p-value
    (Intercept)     1   854  8103.8  <.0001
    craven          1   854   369.3  <.0001
    social          8   854     2.8  0.0049

The denominator degrees of freedom are not adjusted explaining the
difference with the `pbkrtest`-computed result earlier (which we
prefer). But since the dfs are large, it makes little difference here.

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

    Warning in checkDepPackageVersion(dep_pkg = "TMB"): Package version inconsistency detected.
    glmmTMB was built with TMB version 1.9.11
    Current TMB version is 1.9.14
    Please re-install glmmTMB from source or restore original 'TMB' package (see '?reinstalling' for more information)

The default fit uses ML (not REML)

``` r
gtmod <- glmmTMB(math ~ craven + social+(1|school)+(1|school:class),data=jspr)
```

    Warning in finalizeTMB(TMBStruc, obj, fit, h, data.tmb.old): Model convergence problem; non-positive-definite Hessian
    matrix. See vignette('troubleshooting')

``` r
summary(gtmod)
```

     Family: gaussian  ( identity )
    Formula:          math ~ craven + social + (1 | school) + (1 | school:class)
    Data: jspr

         AIC      BIC   logLik deviance df.resid 
          NA       NA       NA       NA      940 

    Random effects:

    Conditional model:
     Groups       Name        Variance Std.Dev.
     school       (Intercept)  3.8350  1.96    
     school:class (Intercept)  0.0001  0.01    
     Residual                 27.6996  5.26    
    Number of obs: 953, groups:  school, 48; school:class, 90

    Dispersion estimate for gaussian family (sigma^2): 27.7 

    Conditional model:
                Estimate Std. Error z value Pr(>|z|)
    (Intercept)  32.0636     1.0298   31.13   <2e-16
    craven        0.5828     0.0318   18.34   <2e-16
    social2      -0.3764     1.0888   -0.35    0.730
    social3      -0.8560     1.1573   -0.74    0.460
    social4      -2.1994     1.0339   -2.13    0.033
    social5      -1.3761     1.1539   -1.19    0.233
    social6      -2.4073     1.2263   -1.96    0.050
    social7      -3.0469     1.2635   -2.41    0.016
    social8      -3.5872     1.6939   -2.12    0.034
    social9      -0.9467     1.0983   -0.86    0.389

We get a warning about convergence and we see that the class random
effect variance is estimated as zero (or very close to it).

We can get some advice via the suggested vignette or:

``` r
diagnose(gtmod)
```

    Unusually large Z-statistics (|x|>5):

      (Intercept)        craven d~(Intercept) 
           31.135        18.341        70.673 

    Large Z-statistics (estimate/std err) suggest a *possible* failure of the Wald approximation - often also
    associated with parameters that are at or near the edge of their range (e.g. random-effects standard
    deviations approaching 0).  (Alternately, they may simply represent very well-estimated parameters;
    intercepts of non-centered models may fall in this category.) While the Wald p-values and standard errors
    listed in summary() may be unreliable, profile confidence intervals (see ?confint.glmmTMB) and likelihood
    ratio test p-values derived by comparing models (e.g. ?drop1) are probably still OK.  (Note that the LRT is
    conservative when the null value is on the boundary, e.g. a variance or zero-inflation value of 0 (Self and
    Liang 1987; Stram and Lee 1994; Goldman and Whelan 2000); in simple cases these p-values are approximately
    twice as large as they should be.)


    Non-positive definite (NPD) Hessian

    The Hessian matrix represents the curvature of the log-likelihood surface at the maximum likelihood
    estimate (MLE) of the parameters (its inverse is the estimate of the parameter covariance matrix).  A
    non-positive-definite Hessian means that the likelihood surface is approximately flat (or upward-curving)
    at the MLE, which means the model is overfitted or poorly posed in some way. NPD Hessians are often
    associated with extreme parameter estimates.


    parameters with non-finite standard deviations:
    theta_1|school:class.1



    recomputing Hessian via Richardson extrapolation. If this is too slow, consider setting check_hessian = FALSE 

    The next set of diagnostics attempts to determine which elements of the Hessian are causing the
    non-positive-definiteness.  Components with very small eigenvalues represent 'flat' directions, i.e.,
    combinations of parameters for which the data may contain very little information.  So-called 'bad
    elements' represent the dominant components (absolute values >0.01) of the eigenvectors corresponding to
    the 'flat' directions


    maximum Hessian eigenvalue = 1.82e+03 
    Hessian eigenvalue 13 = -0.000924 (relative val = -5.08e-07) 
       bad elements: theta_1|school:class.1 

We are not concerned about the large t-value since it is for the
intercept term which we know to be very different from zero. The
boundary effect for the class variance is the the source of our
problems.

We did not have this difficulty with `lme4` and `nlme` (although
encountering this kind of problem is annoyingly common when fitting
mixed effect models). We don’t know the true model but it seems
reasonable to assume that class variance is not zero i.e. that classes
within the same school would tend to vary (perhaps due to the teacher
effect). Even so, we can’t say that `glmmTMB` has failed because it may
be finding a larger likelihood than the previous fits. We could try
tinkering with the settings such as the optimization methods and
starting values but this is often tricky.

Another option is to use REML with:

``` r
gtmodr = glmmTMB(math ~ craven + social+(1|school)+(1|school:class),
                data=jspr,
                REML=TRUE)
summary(gtmodr)
```

     Family: gaussian  ( identity )
    Formula:          math ~ craven + social + (1 | school) + (1 | school:class)
    Data: jspr

         AIC      BIC   logLik deviance df.resid 
      5949.7   6012.8  -2961.8   5923.7      950 

    Random effects:

    Conditional model:
     Groups       Name        Variance Std.Dev.
     school       (Intercept)  3.23    1.80    
     school:class (Intercept)  1.03    1.02    
     Residual                 27.57    5.25    
    Number of obs: 953, groups:  school, 48; school:class, 90

    Dispersion estimate for gaussian family (sigma^2): 27.6 

    Conditional model:
                Estimate Std. Error z value Pr(>|z|)
    (Intercept)  32.0108     1.0364   30.89   <2e-16
    craven        0.5841     0.0321   18.20   <2e-16
    social2      -0.3611     1.0955   -0.33    0.742
    social3      -0.7768     1.1671   -0.67    0.506
    social4      -2.1197     1.0423   -2.03    0.042
    social5      -1.3632     1.1600   -1.18    0.240
    social6      -2.3703     1.2334   -1.92    0.055
    social7      -3.0482     1.2703   -2.40    0.016
    social8      -3.5473     1.7034   -2.08    0.037
    social9      -0.8864     1.1049   -0.80    0.422

The result is very similar but not identical with the previous fits.

# Discussion

See the [Discussion of the single random effect
model](pulp.md#Discussion) for general comments.

`lme4` and `nlme` were both able to fit this model. `mmrm` was not in
the game. `glmmTMB` gives us a REML fit without complaint but ML gives
us a puzzle to solve.

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
    [16] fansi_1.0.6         scales_1.3.0        numDeriv_2016.8-1.1 codetools_0.2-20    cli_3.6.3          
    [21] rlang_1.1.4         munsell_0.5.1       splines_4.4.1       withr_3.0.1         yaml_2.3.10        
    [26] tools_4.4.1         parallel_4.4.1      coda_0.19-4.1       nloptr_2.1.1        minqa_1.2.8        
    [31] dplyr_1.1.4         colorspace_2.1-1    boot_1.3-31         broom_1.0.6         vctrs_0.6.5        
    [36] R6_2.5.1            emmeans_1.10.4      lifecycle_1.0.4     MASS_7.3-61         pkgconfig_2.0.3    
    [41] pillar_1.9.0        gtable_0.3.5        glue_1.7.0          Rcpp_1.0.13         systemfonts_1.1.0  
    [46] xfun_0.47           tibble_3.2.1        tidyselect_1.2.1    rstudioapi_0.16.0   knitr_1.48         
    [51] xtable_1.8-4        farver_2.1.2        htmltools_0.5.8.1   rmarkdown_2.28      svglite_2.1.3      
    [56] labeling_0.4.3      TMB_1.9.14          compiler_4.4.1     
