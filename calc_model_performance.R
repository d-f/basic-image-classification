library("rjson")
library("caret")
library("ROCR")
library("psych")

read_json <- function(json_path) {
  return (fromJSON(file = json_path))
}

parse_accuracy <- function(cm) {
  accuracy <- cm[[3]][[1]]
  return (accuracy)
}

parse_specificity <- function(cm) {
  specificity <- cm[[4]][[2]]
  return (specificity)
}

parse_sensitivity <- function(cm) {
  sensitivity <- cm[[4]][[1]]
  return (sensitivity)
}

calc_roc_auc <- function(predictions, labels) {
  roc_pred <- prediction(predictions = predictions, labels = labels)
  auc <- performance(roc_pred, measure = "auc")
  return (auc@y.values[[1]])
}

calc_cohen_kappa <- function(labels, preds) {
  cks <- cohen.kappa(x=cbind(labels, preds))
  return (cks$kappa)
}

save_results <- function(accuracy, sensitivity, specificity, roc_auc, filename, cks) {
  json_list = list()
  json_list[[1]] = c("Accuracy", "Sensitivity", "Specificity", "ROC-AUC", "Cohen's Kappa")
  json_list[[2]] = c(accuracy, sensitivity, specificity, roc_auc, cks)
  json_file = toJSON(json_list)
  write(json_file, filename)
}

print_results <- function(accuracy, sensitivity, specificity, roc_auc, cks) {
  print(c("accuracy", accuracy))
  print(c("sensitivity", sensitivity))
  print(c("specificity", specificity))
  print(c("ROC-AUC", roc_auc))
  print(c("Cophen's Kappa", cks))
}

calc_results <- function(test_json_path, perf_json_filename) {
  
  opened_json = read_json(json_path=test_json_path)
  targets <- factor(c(opened_json[["labels"]]))
  preds <- factor(c(opened_json[["predictions"]]))
  pred_probs <- opened_json[["predictive probabilities"]]
  label_probs <- opened_json[["label probabilities"]]
  cm <- confusionMatrix(data=preds, reference=targets)
  print(cm)
  accuracy <- parse_accuracy(cm = cm)
  sensitivity <- parse_sensitivity(cm = cm)
  specificity <- parse_specificity(cm = cm)
  roc_auc <- calc_roc_auc(predictions=pred_probs, labels=label_probs)
  cks <- calc_cohen_kappa(labels=targets, preds=preds)
  save_results(
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    roc_auc = roc_auc,
    filename = perf_json_filename
  )
  print_results(
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    roc_auc = roc_auc,
    cks=cks
  )
}    

main <- function() {
  calc_results(
    test_json_path = "",
    perf_json_filename = ""
  )
}

main()

