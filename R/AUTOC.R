# TOC and Monte Carlo Integration
# Description: This script calculates the Targeting Operating Characteristic (TOC) for a given dataset
# and estimates the Area Under the Curve (AUTOC) using Monte Carlo integration.
# Calculates TOC for a given phi and dataset
# Args:
#   phi: Quantile for threshold calculation
#   data: Data frame containing 'tau' or CATE, 'y' or outcome, and 'w' or treatment indicator columns
# Returns:
#   TOC value for the given phi


TOC <- function(phi, data) {
  threshold <- quantile(data$tau, 1 - phi)
  filtered_data <- data %>% filter(tau >= threshold) 
  
  # Check if filtered data has enough rows to proceed
  if (nrow(filtered_data) < 1) {
    return(0)  # Return a neutral value or consider other appropriate responses
  }
  
  # Proceed with calculations
  top_phi_mean_diff <- mean(filtered_data$y[filtered_data$w == 1], na.rm = TRUE) - 
    mean(filtered_data$y[filtered_data$w == 0], na.rm = TRUE)
  overall_mean_diff <- mean(data$y[data$w == 1], na.rm = TRUE) - 
    mean(data$y[data$w == 0], na.rm = TRUE)
  
  toc_value <- top_phi_mean_diff - overall_mean_diff
  return(toc_value)
}

monte_carlo_integration <- function(f, lower, upper, samples = nrow(data)) {
  random_samples <- runif(samples, min = lower, max = upper)
  mean(f(random_samples)) * (upper - lower)
}


# Modified function to include Monte Carlo integration
calculate_and_plot_TOC_mc <- function(data, samples = nrow(data)) {
  # Generate random phi values uniformly across the interval [0, 1]
  phi_values <- runif(samples, min = 0.000001, max = 1)
  
  # Calculate TOC values for each randomly generated phi
  toc_values <- sapply(phi_values, function(phi) TOC(phi, data))
  
  # Create a dataframe for plotting
  toc_data <- data.frame(phi = phi_values, toc = toc_values)
  
  # Estimate AUTOC using Monte Carlo integration
  AUTOC_value <- monte_carlo_integration(function(phi) TOC(phi, data), 0, 1, samples)
  
  # Plotting the TOC points and filling the area under the curve
  toc_plot <- ggplot(toc_data, aes(x = phi, y = toc)) +
    geom_point(alpha = 0.4, color = "blue", size = 0.5) +
    labs(x = "Quantile (phi)", y = "TOC Value", title = "Monte Carlo Integration of TOC Curve") +
    theme_minimal() +
    geom_text(aes(x = 0.5, y = min(toc_values), label = paste("AUTOC:", round(AUTOC_value, 3))),
              vjust = -1, color = "red", hjust = 0.5)  # Annotate the plot with the AUTOC value
  
  # Print the plot
  print(toc_plot)
  return(AUTOC_value)
}
