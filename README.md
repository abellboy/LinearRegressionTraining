# Training Linear Regression

## Stochastic Gradient Decent


When your model needs to adapt in real-time as part of a streaming data pipeline, Stochastic Gradient Descent (SGD) is a great choice.

It’s also ideal for large datasets where closed-form solutions (like OLS) aren’t feasible. I created a visual Python demo that shows how SGD iteratively learns the regression line. 

Since SGD is an iterative method, it must be trained in epochs. Choosing the right number means balancing speed against accuracy.
In my demo, I used 12 epochs so the animation loops quickly and remains visually appealing

Some pros and cons to consider:

✅ Scales well to large datasets  
✅ Supports L1/L2 regularization  
✅ Enables real-time updates  

⚠️ Requires tuning (learning rate, epochs)  
⚠️ May converge slowly or to suboptimal values  



**Code is in:** LinearRegressionTraining_GradientDescent.py

**The learning rate and training epoch hyperparameters can be tuned by seeting:**

**Learning Rate:** learning_rate (Currently set to: 0.1)  
**Training Epoch**: n_iterations (Currently set to 12)


