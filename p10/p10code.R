# ---- BEGIN R Script (fast version) ----
fn <- tempfile()
download.file("https://github.com/dmcable/BIOSTAT620/raw/refs/heads/main/data/pset-10-mnist.rds", fn)
dat <- readRDS(fn)
file.remove(fn)

trainX <- as.matrix(dat$train$images)
trainY <- dat$train$labels
testX  <- as.matrix(dat$test$images)

library(xgboost)
dtrain <- xgb.DMatrix(data = trainX, label = trainY)
dtest  <- xgb.DMatrix(data = testX)

params <- list(
  objective = "multi:softmax",
  num_class = 10,
  eval_metric = "merror",
  max_depth = 8,
  eta = 0.1,
  subsample = 0.9,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 120,
  verbose = 1
)

digit_predictions <- predict(xgb_model, dtest)
digit_predictions <- as.integer(digit_predictions)
saveRDS(digit_predictions, file = "digit_predictions.rds")
# ---- END R Script ----

