import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(50, 1)
y = 4 + 3 * X + np.random.randn(50, 1)


# Add intercept term to X (for bias term)
X_b = np.c_[np.ones((50, 1)), X]



# Gradient Descent parameters
learning_rate = 0.1
n_iterations = 12
m = len(X_b)


# Initialize theta (random starting point)
theta = np.random.randn(2, 1)


# Store theta values at each step for animation
theta_path = [theta.copy()]


# Gradient Descent loop
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T @ (X_b @ theta - y)
    theta = theta - learning_rate * gradients
    theta_path.append(theta.copy())


# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 2)
ax.set_ylim(0, 15)
ax.set_title("Gradient Descent Regression Line Animation")


# Scatter plot of original training data
ax.scatter(X, y, color='blue', label='Training data')
regression_line, = ax.plot([], [], 'k--', label='Regression line')


# Function to update the line for each frame
def update(frame):
    current_theta = theta_path[frame]
    x_vals = np.array([0, 2])
    y_vals = current_theta[0] + current_theta[1] * x_vals
    regression_line.set_data(x_vals, y_vals)
    return regression_line,


# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=len(theta_path), interval=200, blit=True, repeat=False
)


# Save the animation as a GIF
ani.save("gd_regression_animation.gif", writer='pillow', fps=5)

