protect_CATEs = function(percent, CATE, CATE_estimates, n, epsilons = c(0.05,0.5,1,3,5)){
  top = floor(n * percent)
  selection_true = rep(0, n)
  selection_tau = rep(0, n)
  selection_tau[as.data.frame(sort(CATE_estimates,decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  if(length(CATE) > 0){
    selection_true[as.data.frame(sort(CATE, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  }
  
  # now with local dp
  collection = data.frame(customer = 1:n)
  for (epsilon in epsilons){
    print(epsilon)
    protected_selection = protect_selection(epsilon = epsilon, selection = selection_tau, top = top)
    collection = cbind(collection, protected_selection)
  }
  colnames(collection) = c("customer", paste0("", gsub("\\.", "", as.character(paste0("epsilon_", epsilons)))))
  collection$random = sample(x = c(0,1), size = n, replace = TRUE, prob= c(1-percent,percent))
  collection$percentage = percent
  collection$selection_true = selection_true
  collection$selection_tau = selection_tau
  if(length(CATE) > 0){
    collection$tau = CATE
  }
  return(collection)
}

protect_selection = function(epsilon, selection, top, seed = 1){
  # privacy settings
  P = matrix(nrow = 2, ncol = 2)
  diag(P) = (exp(epsilon))/(2-1+exp(epsilon))
  P[is.na(P)==T] = (1)/(2-1+exp(epsilon))
  
  # get responses
  responses = rep(0,length(selection))
  
  # for every row in the responses generate protected selection based on matrix above.
  for (i in 1:length(selection)){
    set.seed(seed+i)
    responses[i] = ifelse(selection[i] == 0, sample(x = c(1:2)-1,size = 1,prob= P[1,]), sample(x = c(1:2)-1,size = 1,prob=P[2,]))
  }
  
  protected_selection = responses # make responses equal to the selection
  index_0 = which(protected_selection == 0) # select the rownumbers that are equal to 0
  index = which(protected_selection == 1) # select the rownumbers that are equal to 0
  protected_selection = rep(0,length(selection)) # set the selection to zero again
  
  if(top > length(index)){
    protected_selection[sample(index, length(index))] = 1 # sample everyone from index
    protected_selection[sample(index_0, top - length(index))] = 1 # sample from not selected to get equal amount to top (top-length is remainder)
  }else if(top < length(index)){
    protected_selection[sample(index, length(index))] = 1 # sample everyone from index
    protected_selection[sample(index, abs(top - length(index)))] = 0 # sample from not selected to get equal amount to top (top-length is remainder)
  }
  return(protected_selection)
}

bootstrap_strat_2 = function(bootstraps, CATE, CATE_estimates, percentage = seq(0.05, .95, by = 0.05), epsilons = c(0.05, 0.5, 1, 3, 5), seed = 1){
    # Set the number of seeds
    set.seed(seed)
    seeds <- sample(1:1000000, bootstraps, replace = FALSE)
    
    # Initialize a list to store bootstrap results
    bootstrap_results <- data.frame()
    
    # Loop over each bootstrap iteration
    for (b in 1:bootstraps) {
      set.seed(seeds[b])
      # Resample the data with replacement
      bootstrap_data <- CATE[sample(length(CATE), replace = TRUE)]
    
      # Initialize an object to store results for this bootstrap
      percentage_collection <- NULL
    
      # Loop over each percentage level
      for (percent in percentage) {
        set.seed(seeds[b])
        # Apply the protect function using the bootstrap sample
        collection <- protect_CATEs(percent = percent,
                            CATE = CATE,
                            CATE_estimates = CATE_estimates,
                            n = length(CATE_estimates),
                            epsilons = c(0.05, 0.5, 1, 3, 5),
                            seed = seeds[b])
        collection$percent <- percent
        percentage_collection <- rbind(percentage_collection, collection)
      }
    
      # Store the results from this bootstrap iteration
      percentage_collection$bootstrap = b
      bootstrap_results = rbind(bootstrap_results, percentage_collection)
  }
  return(bootstrap_results)
}

policy_overlap = function(data, bootstrap = FALSE){
  if (bootstrap == TRUE){
    overlap = data %>% dplyr::select(customer, selection_true, selection_tau, epsilon_005, epsilon_05,
                                 epsilon_1,epsilon_3,epsilon_5, random, percent,bootstrap) %>%
  group_by(percent, bootstrap) %>%
  filter(percent > 0) %>% filter(percent < 1) %>%
  summarize(overlap_random = table(selection_true, random)[2,2]/sum(selection_true),
            overlap_05 = table(selection_true, epsilon_05)[2,2]/sum(selection_true),
            overlap_005 = table(selection_true, epsilon_005)[2,2]/sum(selection_true),
            overlap_1 = table(selection_true, epsilon_1)[2,2]/sum(selection_true),
            overlap_3 = table(selection_true, epsilon_3)[2,2]/sum(selection_true),
            overlap_5 = table(selection_true, epsilon_5)[2,2]/sum(selection_true), .groups = "keep") %>% 
  pivot_longer(c(overlap_random, overlap_005, overlap_05,overlap_1,overlap_3,overlap_5)) %>%
  group_by(percent,name) %>%
  summarize(mean_overlap = mean(value),
            lower = quantile(value, probs = 0.025),
            upper = quantile(value, probs = 0.975))
  }else{
    overlap = data %>% 
    dplyr::select(customer, selection_true, selection_tau, epsilon_005, epsilon_05,
                  epsilon_1,epsilon_3,epsilon_5, random, percent) %>%
    group_by(percent) %>%
  filter(percent > 0) %>% filter(percent < 1) %>%
    summarize(overlap_random = table(selection_true, random)[2,2]/sum(selection_true),
              overlap_05 = table(selection_true, epsilon_05)[2,2]/sum(selection_true),
              overlap_005 = table(selection_true, epsilon_005)[2,2]/sum(selection_true),
              overlap_1 = table(selection_true, epsilon_1)[2,2]/sum(selection_true),
              overlap_3 = table(selection_true, epsilon_3)[2,2]/sum(selection_true),
              overlap_5 = table(selection_true, epsilon_5)[2,2]/sum(selection_true))
  }
  
  return(overlap)
}

policy_profit = function(data, bootstrap = FALSE){
  if (bootstrap == TRUE){
    profit = data %>% dplyr::select(tau, selection_true, selection_tau, epsilon_005, epsilon_05,
                                                                 epsilon_1,epsilon_3,epsilon_5, random, percent, bootstrap) %>%
  pivot_longer(c(selection_true, selection_tau,  epsilon_005, epsilon_05,
                 epsilon_1,epsilon_3,epsilon_5, random)) %>% 
  group_by(percent,name, bootstrap) %>% 
  summarize(profit = (sum(tau*value))) %>%
  summarize(mean_profit = mean(profit),
            lower = quantile(profit, probs = 0.025),
            upper = quantile(profit, probs = 0.975))
  }else{
    profit = data %>% dplyr::select(tau, selection_true, selection_tau, epsilon_005, epsilon_05,
                                        epsilon_1,epsilon_3,epsilon_5, random, percent) %>%
    pivot_longer(c(selection_true, selection_tau,  epsilon_005, epsilon_05,
                   epsilon_1,epsilon_3,epsilon_5, random)) %>% 
  group_by(percent,name) %>% 
  summarize(profit = (sum(tau*value)))
  }
  return(profit)
}


profit = function(epsilon, CATE, CATE_estimates, phi){
  if (length(CATE) > 0){
    N = length(CATE) 
  }else{
    N = length(CATE_estimates)
  }
  top = round(N * phi)

  if(length(CATE) > 0){
    sorted_indices = order(CATE, decreasing = TRUE)
    selection = CATE >= CATE[sorted_indices[top]]
  }else{
    sorted_indices = order(CATE_estimates, decreasing = TRUE)
    selection = CATE_estimates >= CATE_estimates[sorted_indices[top]]
  }
  
  # for every row in the responses generate protected selection based on matrix above.
  RR = rbinom(n = N, 1, 1 / (1 + exp(epsilon)))
  selection[RR == 1] <- 1 - selection[RR == 1]
  
  # Find targeted and non-targeted indices after RR
  who_is_targeted <- which(selection == 1)
  who_is_not_targeted <- which(selection == 0)
  targetRR_Nphi <- rep(0, N)
  
  if (round(phi * N) < length(who_is_targeted)) {
    # If too many are targeted, randomly select a subset
    random_select <- sample(who_is_targeted, size=round(phi * N), replace=FALSE)
    targetRR_Nphi[random_select] <- 1
  } else {
    # If too few are targeted, add random non-targeted people
    targetRR_Nphi[who_is_targeted] <- 1
    random_select <- sample(who_is_not_targeted, size=round(phi * N) - length(who_is_targeted), replace=FALSE)
    targetRR_Nphi[random_select] <- 1
  }
  
  # Generate a random policy
  randompolicy <- sample(N, round(phi * N))
  randomtarget <- rep(0, N)
  randomtarget[randompolicy] <- 1
  
  # Calculate the profit based on q
  prof_random <- sum(CATE[randomtarget == 1])
  empirical_profit = sum(CATE[targetRR_Nphi == 1])
  return(empirical_profit - prof_random)
}

expected_profit = function(phi, lambda, epsilon, N){
  pr <- exp(epsilon) / (1 + exp(epsilon))
  Prof1 <- (1 - log(phi)) / lambda
  Prof2 <- (phi * log(phi) + 1 - phi) / (lambda * (1 - phi))
  
  # Profit on the target group
  Pi1 <- pr * N * phi * Prof1 + (1 - pr) * N * (1 - phi) * Prof2
  # Profit on the not-target group
  Pi2 <- (1 - pr) * N * phi * Prof1 + pr * N * (1 - phi) * Prof2
  # People in the target group
  N1 <- pr * N * phi + (1 - pr) * N * (1 - phi)
  # People in the non-target group
  N2 <- (1 - pr) * N * phi + pr * N * (1 - phi)
  
  # Theoretical profit
  if (phi < 0.5) {
    p <- (phi * N / N1) * Pi1
    } else {
    p <- Pi1 + (Pi2 * ((N * phi - N1) / N2))
}
return(p)
}

# Define the function after subtracting the random policy
expected_profit_random <- function(phi, lambda, epsilon, N) {
  return(expected_profit(phi, lambda, epsilon, N) - expected_profit(phi, lambda, epsilon = 0, N))
}
