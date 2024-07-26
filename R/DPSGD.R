rm(list=ls())
set.seed(123)

# Define the gradient function
gradient <- function(t, y, y_hat, h) {
  # Compute the gradient for weight update
  # Since t, y, y_hat can be vectors for batch processing, we use outer product to form the gradient matrix when h is a matrix.
  (-2 * (y - y_hat) * t) %*% t(h)
}

# Parameters
l_1 <- 2 # Previous layer dimensionality
l <- 1   # Next layer dimensionality
a <- 0.01 # Learning rate

# Privacy parameters
C <- 1   # Constraint parameter
o = 0.1 # privacy multiplier

iterations = 10000

# Initialize h and theta
h <- cbind(rnorm(n = l_1))
theta_truth <- as.matrix(cbind(rnorm(n = l), rnorm(n = l))) # 1 by 5 matrix
y <- theta_truth %*% h

# Initialize theta and theta_private
theta <- as.matrix(cbind(rnorm(1), rnorm(1)))
theta_private <- theta

# Initialize lists to store theta values for plotting
theta_values <- list(theta = numeric(), theta_private = numeric(), diff = numeric(), diff_private = numeric())

# Perform gradient descent iterations
for (i in 1:iterations) {
  y_hat <- theta %*% h
  
  # Compute the gradient
  t <- c(1)       # Target value for a batch
  g <- gradient(t, y, y_hat, h)
  
  # Update theta
  theta <- theta - (a * g)
  
  # Update theta_private
  grad_priv <- g / max(1, (sqrt(sum(g^2)) / C))
  grad_priv = grad_priv + rnorm(l_1, 0, o^2*C^2)
  
  
  theta_private <- theta_private - (a * grad_priv)
  y_hat_private <- theta_private %*% h
  
  # Store theta values
  theta_values$theta <- rbind(theta_values$theta, theta)
  theta_values$theta_private <- rbind(theta_values$theta_private, theta_private)
  theta_values$diff <- rbind(theta_values$diff, y - y_hat)
  theta_values$diff_private <- rbind(theta_values$diff_private, y - y_hat_private)
}

# Plot convergence of the difference between y and y_hat
par(mfrow = c(1, 1))  # Reset the plotting layout to a single plot
# Plot the difference
plot(1:iterations, theta_values$diff, type = "l", col = "blue", xlab = "Iteration", ylab = "Difference between y and y_hat",
     main = "Convergence of the Difference between y and y_hat")

# Add the second line
lines(1:iterations, theta_values$diff_private, type = "l", col = "red")

# Add legend
legend("topright", legend = c("Theta", "Theta_private"), col = c("blue", "red"), lty = 1)
