# %% [code]
---
title: "Regression with an Abalone Dataset"
date: '`r Sys.Date()`'
output:
  html_document:
    number_sections: true
    fig_caption: true
    toc: true
    fig_width: 7
    fig_height: 4.5
    theme: cosmo
    highlight: tango
    code_folding: hide
---
  
# Introduction  {.tabset .tabset-fade .tabset-pills}

The goal of this competition is to predict the age of abalone from various physical measurements.

My notebook serves as a demonstration of some of the possible techniques available to arrive at a solution.  I intend to add to this as I have time available. Your questions and comments are welcome.

If you fork this on kaggle, be sure to choose the kernel Environment setting for "Always use latest environment"

Lets dive right in.

The Kaggle kernels have many of the common r packages built in.  

## Load libraries

In addition to `tidymodels` we will load the `bonsai` interface to lightgbm.

```{r }
#| label: setup
#| warning: false
#| message: false
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/focal/2021-05-17"))
 
remotes::install_version("ranger", quiet = TRUE)
remotes::install_github("mayer79/outForest", quiet = TRUE)

suppressPackageStartupMessages({
library(tidyverse) # metapackage of all tidyverse packages
library(tidymodels) # metapackage see https://www.tidymodels.org/
  
# library(extrasteps)

library(bonsai) # interface with lightgbm
library(poissonreg) # interface with poisson regression glmnet
library(brulee) # interface with Torch deep learning    

})

tidymodels_prefer()

options(tidymodels.dark = TRUE)

theme_set(cowplot::theme_minimal_grid())

```


## Interchangeability

I prefer to be able to run the same code locally and on a Kaggle kernel.

```{r}
#| label: interchangeability
#| warning: false
#| message: false

if (dir.exists("/kaggle")){
  path <- "/kaggle/input/playground-series-s4e4/"
} else {
  path <- str_c(here::here("data"),"/")
}

path

```

## Load Data

```{r }
#| label: load data
#| warning: false
#| message: false

preprocessor <- function(dataframe) {
  dataframe %>%
    janitor::clean_names() %>% 
    mutate(across(c(where(is.character)), ~ as.factor(.x)),)
}

raw_df <- read_csv(str_c(path, "train.csv"),
                   show_col_types = TRUE) %>%

  preprocessor() %>%
  filter(between(height, 0.0001, 0.3))

features <- raw_df %>%
  select(-id, -rings) %>%
  names()

raw_df <- raw_df %>% 
  distinct(pick(all_of(features)), .keep_all = TRUE)

nom_features <- raw_df %>%
  select(all_of(features)) %>%
  select(where(is.character), where(is.factor)) %>%
  names() 

logical_features <- raw_df %>%
  select(all_of(features)) %>%
  select(where(is.logical)) %>%
  names() 

num_features <- raw_df %>%
  select(all_of(features)) %>%
  select(where(is.numeric)) %>%
  names()

competition_df <- read_csv(str_c(path, "test.csv"),
                   show_col_types = FALSE)  %>% 
  preprocessor() 

all_df <-
    bind_rows(raw_df %>% mutate(source = "train"),
            competition_df %>% mutate(source = "test"))

```

Nominal features:

`r nom_features`

Numeric features: 

`r num_features`


# EDA {.tabset .tabset-fade .tabset-pills}

## Numeric features

Consider where features require univariate transformation, or clipping outliers.

I initially implemented an Outlier section, but went back to preprocess to remove the high and low `height` figures from training.

There are a couple outlier `height` values in test as well.

Other kernels make some effort to impute zero `height` figures.


```{r}
#| label: numeric
#| warning: false
#| message: false
#| fig.height: 12
#| fig.width: 12

raw_df %>% 
  select(all_of(num_features), rings) %>% 
  pivot_longer(-rings,
    names_to = "metric",
    values_to = "value"
  ) %>%
  ggplot(aes(value, fill = ggplot2::cut_number(rings,5))) +
  geom_histogram(alpha = 0.6, bins = 50) +
   facet_wrap(vars(metric), scales = "free", ncol = 3) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  labs(color = NULL, fill = "Rings",
       title = "Numeric Feature Univariate Distributions",
       caption = "Data: Kaggle.com | Visual: Jim Gruman")

raw_df %>% 
  select(all_of(num_features), rings) %>% 
  pivot_longer(-rings,
    names_to = "metric",
    values_to = "value"
  ) %>%
  ggplot(aes(sample = value, color = ggplot2::cut_number(rings,5))) + 
  stat_qq(show.legend = FALSE) + 
  stat_qq_line() +
  facet_wrap(vars(metric), scales = "free", ncol = 3) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  labs(color = NULL, 
       title = "Numeric Feature Q-Q Distributions",
       caption = "Data: Kaggle.com | Visual: Jim Gruman")

my_bin <- function(data, mapping, ..., low = "#132B43", high = "#56B1F7") {
  ggplot(data = data, mapping = mapping) +
    geom_bin2d(...) +
    scale_fill_gradient(low = low, high = high)
}

raw_df %>%
  select(all_of(num_features), rings, sex) %>% 
  GGally::ggpairs(
    mapping = aes(color = sex),
    lower = list(
      combo = GGally::wrap("facethist", binwidth = 1)
    )
  )

```

## Outforest

`outForest` is a multivariate anomaly detection method. Each numeric variable is regressed onto all other variables using a random forest. If the scaled absolute difference between observed value and out-of-bag prediction is larger than a prespecified threshold, then a value is considered an outlier.

A benefit of this technique is noise reduction, at least at the most anomylous observations.  A drawback is of applying it at this point to the training data and the cross validation folds is that the CV out of sample estimates of error are smaller than we would see with raw.
                   
Computationally, it is faster to run it here.                   

```{r}
#| label: outforest anomaly detection and interpolation
                   
outforest_model <- outForest::outForest(
  data = raw_df,
  formula =  formula(paste0(str_c(features,  collapse = " + "), "~ rings")),
  max_prop_outliers = 0.0005,
  threshold = 3,
  impute_multivariate_control = list(
    pmm.k = 3L,
    num.trees = 250L,
    maxiter = 3L
  ),
  allow_predictions = TRUE,
  verbose = 0
)

plot(outforest_model, what = "scores")
plot(outforest_model, what = "counts")

outforest_preds <- predict(outforest_model, newdata = raw_df)

train_df <- outForest::Data(outforest_preds) %>% as_tibble()

```


## Counts of Missingness
                  
```{r}
#| label: counts of missingness

raw_df %>% 
  summarize(across(all_of(features), function(x) sum(is.na(x)))) %>% 
  pivot_longer(everything(),
              names_to = "feature",
              values_to = "Count of Missing") %>% 
                   knitr::kable()
```

## Counts of Distinct
               
```{r}
#| label: counts of distinct
               
raw_df %>% 
  summarize(
    across(all_of(features), n_distinct)
  ) %>%
  pivot_longer(everything(),
               names_to = "feature",
               values_to = "Count of distinct") %>% 
                   knitr::kable()
               
```

## Duplicated

Is this competition transaction already in the training data with a correct label?

```{r}
#| label: duplicates
#| warning: false
#| message: false
all_df %>%
    select(all_of(features), source) %>% 
    group_by_at(features) %>%
    mutate(num_dups = n(),
           dup_id = row_number()) %>%
    ungroup() %>%
    group_by(source) %>%
    mutate(is_duplicated = dup_id > 1) %>%
    count(is_duplicated) %>% 
                   knitr::kable()
               

```
                   
## Pairwise Correlations
                   
`ggcorrplot` provides a quick look at numeric features where the correlation may be significant. 
                         

```{r}
#| label: pairwise correlations
# Leave blank on no significant coefficient

corr <- raw_df %>%
  select(all_of(num_features), rings) %>%
  cor()

p.mat <-raw_df %>%
  select(all_of(num_features), rings) %>%
  ggcorrplot::cor_pmat() 

ggcorrplot::ggcorrplot(
    corr,
    hc.order = TRUE,
    lab = TRUE,
    type = "lower",
    insig = "blank",
    p.mat = p.mat
  ) +
  labs(title = "Pairwise Correlations Training Set")

ggcorrplot::ggcorrplot(
   competition_df %>%
    select(all_of(num_features)) %>%
    cor(),
    hc.order = TRUE,
    lab = TRUE,
    type = "lower",
    insig = "blank",
    p.mat = competition_df %>%
  select(all_of(num_features)) %>%
  ggcorrplot::cor_pmat() 
  ) +
  labs(title = "Pairwise Correlations Competition Set")

```      


## Target

```{r}
#| label: outcome 
#| warning: false
#| message: false
#| fig.width: 12

raw_df %>% 
 ggplot(aes(rings)) +
  geom_histogram(bins = 100) +
  labs(title = "Outcome: Rings",
       caption = "Data: Kaggle.com | Visual: Jim Gruman")

```
                              
           
                
# Machine Learning {.tabset .tabset-fade .tabset-pills}

## Recipe
                   
We will tune each modeling algorithm against 10 fold cross validation, stratified for the outcome.
                 

```{r}
#| label: recipe
#| fig.height: 16
#| fig.width: 12

folds <- vfold_cv(raw_df, 
                  v = 11,
                  repeats = 1,
                  strata = rings)

rec <- recipe(
    
    formula(paste0("rings ~ ", 
               str_c(features,  collapse = " + "))),
    data = raw_df 
  ) %>%
 step_ratio(shell_weight, denom = denom_vars(length)) %>%                   
 step_novel(all_nominal_predictors()) %>% 
 step_nzv(all_numeric_predictors())

rec
                  
```

## Metric

Root Mean Squared Log Error


```{r}
#| label: custom RMSLE metric

rmsle_impl <- function(truth, estimate, case_weights = NULL) {
        sqrt(mean((log(abs(estimate) + 1) - log(abs(truth) + 1))^2))
}

rmsle_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
  check_numeric_metric(truth, estimate, case_weights)

  if (na_rm) {
    result <- yardstick_remove_missing(truth, estimate, case_weights)

    truth <- result$truth
    estimate <- result$estimate
    case_weights <- result$case_weights
  } else if (yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }

  rmsle_impl(truth, estimate, case_weights = case_weights)
}

rmsle <- function(data, ...) {
  UseMethod("rmsle")
}

rmsle <- new_numeric_metric(rmsle, direction = "minimize")

rmsle.data.frame <- function(data, truth, estimate, case_weights = NULL, na_rm = TRUE, ...) {
  
  numeric_metric_summarizer(
    name = "rmsle",
    fn = rmsle_vec,
    data = data,
    truth = !!enquo(truth),
    estimate = !!enquo(estimate),
    na_rm = na_rm,
    case_weights = !!enquo(case_weights)
  )
  
}

metrics <- metric_set(rmsle)


```

## LightGBM 

```{r}
#| label: lightgbm engine

boost_tree_lgbm_spec <- 
  boost_tree(
    trees = 2500L,
   tree_depth = tune(),
   learn_rate =  tune(),
   min_n = tune(),
    loss_reduction = 0
  ) %>% 
  set_engine(engine = "lightgbm",
             is_unbalance = TRUE,
             num_leaves = tune(),
             num_threads = future::availableCores()
             ) %>%
  set_mode(mode = "regression") 

boost_tree_lgbm_spec

```

                                                                     

```{r}
#| label: lgbm fit 
#| warning: false
#| message: false
#| fig.height: 6
#| fig.width: 12
tictoc::tic()
wf <- workflow(rec,
               boost_tree_lgbm_spec) 

set.seed(42)

ctrl <- finetune::control_sim_anneal(     
     verbose = FALSE,
     verbose_iter = TRUE,
     parallel_over = "everything",
     save_pred = TRUE,
     save_workflow = TRUE)

param <- wf %>%
   extract_parameter_set_dials() %>%
   recipes::update(
      min_n = min_n(range = c(10, 40)),
      tree_depth = tree_depth(range = c(6, 40)),
      learn_rate = learn_rate(range = c(-1.5, -2.5)),
      num_leaves = num_leaves(range = c(100, 250))
   ) %>%
   dials::finalize(raw_df)                   
                   
burnin <- tune_grid(
  wf,
  grid = 4,
  resamples = folds,
  control = ctrl,
  metrics = metrics,
  param_info = param)

lgbm_rs <- finetune::tune_sim_anneal(
  wf,
  resamples = folds,
  iter = 18,
  initial = burnin,
  control = ctrl,
  metrics = metrics,
  param_info = param) 

show_best(lgbm_rs)                   
                                     
autoplot(lgbm_rs)                   

tictoc::toc()                    
```
                   
## Poisson GLMnet                   

```{r}
#| label: poisson reg  
#| warning: false
#| message: false
#| fig.height: 6
#| fig.width: 12
tictoc::tic()
                   
all_cores <- parallelly::availableCores()

future::plan("multisession", workers = all_cores)                    

poisson_rec <- rec %>%             
  step_poly(all_numeric_predictors()) %>%   
  step_dummy_multi_choice(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

poisson_reg_glmnet_spec <-
  poisson_reg(penalty = tune(), mixture = tune()) %>%
  set_engine('glmnet')

wf <- workflow(poisson_rec  ,
               poisson_reg_glmnet_spec) 

set.seed(42)

param <- wf %>%
   extract_parameter_set_dials() %>%
   recipes::update(
           mixture = mixture(),
           penalty = penalty(range = c(-4, -2))
        ) %>%
   dials::finalize(raw_df)

poisson_rs <- tune_grid(
  wf,
  resamples = folds,
  grid = 10,
  control = ctrl,
  metrics = metrics,
  param_info = param
)

autoplot(poisson_rs) +
  labs(title = "Poisson GLMnet hyperparameter tuning",
       caption = "Data: Kaggle.com | Visual: Jim Gruman")      
                   
show_best(poisson_rs)                          
                   
future::plan("sequential")                                   
tictoc::toc()                    
```
                   
## XGBoost                

```{r}
#| label: xgboost
#| warning: false
#| message: false
#| fig.height: 6
#| fig.width: 12
tictoc::tic()

all_cores <- parallelly::availableCores()

future::plan("multisession", workers = all_cores) 
                   
                   
xgb_spec <-
  boost_tree(
    trees = 3000,
    tree_depth = 7,
    learn_rate = 0.03,
    min_n = 25) %>%
  set_engine(engine = "xgboost") %>%
  set_mode(mode = "regression")

wf <- workflow(rec %>%
       step_dummy_multi_choice(all_nominal_predictors())  ,
               xgb_spec) 

xgb_rs <- fit_resamples(
  wf,
  resamples = folds,
  control = ctrl,
  metrics = metrics)

collect_metrics(xgb_rs)                   
                   
future::plan("sequential") 
                   
tictoc::toc()                    
```      
                   
## Torch                   

```{r}
#| label: brulee
#| warning: false
#| message: false
#| fig.height: 6
#| fig.width: 12
tictoc::tic()

mlp_brulee_spec <-
  mlp(
    hidden_units = 18,
    epochs = 1500,
    learn_rate = 0.1) %>%
  set_engine('brulee',
            stop_iter = 100) %>%
  set_mode('regression')

wf <- workflow(rec %>%
  step_dummy_multi_choice(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())  ,
               mlp_brulee_spec) 

set.seed(42)

brulee_rs <- fit_resamples(
  wf,
  resamples = folds,
  control = ctrl,
  metrics = metrics)

collect_metrics(brulee_rs)

tictoc::toc()                    
``` 
                   
## SVM
                   
```{r}
#| label: svm
#| warning: false
#| message: false
#| fig.height: 6
#| fig.width: 12
tictoc::tic()

svm_poly_kernlab_spec <-
  svm_poly(
    cost = 6.66,
    degree = 2,
    scale_factor = 0.003,
    margin = 0.126
  ) %>%
  set_engine('kernlab') %>%
  set_mode('regression')

wf <- workflow(rec %>%
  step_dummy_multi_choice(all_nominal_predictors()) %>%               
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) ,
               svm_poly_kernlab_spec)

set.seed(42)

svm_rs <- fit_resamples(
  wf,
  resamples = folds,
  control = ctrl,
  metrics = metrics)

collect_metrics(svm_rs)

tictoc::toc()                    
```                  
                   

# Ensemble Stacks

```{r}
#| label: stacking ensemble
#| warning: false
#| message: false
#| fig.height: 6
#| fig.width: 12

all_cores <- parallelly::availableCores()

future::plan("multisession", workers = all_cores) 


ens <- stacks::stacks() %>%
  stacks::add_candidates(lgbm_rs) %>%
  stacks::add_candidates(poisson_rs) %>%    
  stacks::add_candidates(xgb_rs) %>%
  stacks::add_candidates(brulee_rs) %>%    
  stacks::add_candidates(svm_rs) %>%
  stacks::blend_predictions(
    metric = metrics,
    penalty = c(seq(0.0001, 0.01, 0.0002)),
    control = tune::control_grid(allow_par = TRUE)
  ) 

autoplot(ens)

autoplot(ens, "weights")

ensemble <- stacks::fit_members(ens)
                   
future::plan("sequential") 
```    
                   
 
                   
# The moment of Truth

```{r}
#| label: performance checks
#| warning: false
#| message: false


predict(ensemble, raw_df) %>%
  bind_cols(raw_df) %>%
  rmsle(rings, .pred)

predict(ensemble, raw_df) %>%
  bind_cols(raw_df) %>%
  ggplot(aes(rings, .pred - rings, color = sex), shape = 21, alpha = 0.05) + 
                   geom_jitter() +
  labs(title = "Residuals")
                   
```
                   
## Submission
                   
This submission will be without the OutForest anomaly removal.                    
                   
```{r}
#| label: submission
#| warning: false
#| message: false
                   
submit_df <- predict(ensemble, competition_df) %>%
  bind_cols(competition_df) %>%
  mutate(Rings = if_else(.pred > 0, round(.pred), 1)) %>%
  select(id, Rings)

head(submit_df)  %>% 
     knitr::kable()      
                   
submit_df  %>% 
  write_csv("submission.csv")
  

```