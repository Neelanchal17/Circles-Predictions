from sklearn.datasets import make_circles
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
n_samples = 1000
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

z = 0.8
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    ) # make the random split reproducible

len(X_train), len(X_test), len(y_train), len(y_test)
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
print(circles.head())
print(type(X), type(y))
print(len(X), len(y))
plt.scatter(x=X_train[:, 0], 
            y=X_train[:, 1], 
            c=y_train, 
            cmap=plt.cm.RdYlBu);


plt.show()
model0 = Sequential([
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')

])
print(model0.summary())
model0.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model0.fit(
    X_train, y_train,
    epochs=100,
)

preds = model0.predict(X_test)

yhat = np.zeros_like(preds)
for i in range(len(preds)):
    if preds[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")

X_test_class0 = X_test[y_test == 0]
X_test_class1 = X_test[y_test == 1]

X_pred_class0 = X_test[yhat.flatten() == 0] 
X_pred_class1 = X_test[yhat.flatten() == 1]

plt.figure(figsize=(10, 6))

plt.scatter(X_test_class0[:, 0], X_test_class0[:, 1], color='red', marker='o', label='True Class 0')
plt.scatter(X_test_class1[:, 0], X_test_class1[:, 1], color='blue', marker='o', label='True Class 1')

plt.scatter(X_pred_class0[:, 0], X_pred_class0[:, 1], color='red', marker='x', label='Predicted Class 0', alpha=0.5)
plt.scatter(X_pred_class1[:, 0], X_pred_class1[:, 1], color='blue', marker='x', label='Predicted Class 1', alpha=0.5)


plt.xlabel('X1')
plt.ylabel('X2')
plt.title('True Labels vs. Predicted Labels')
plt.legend()


plt.show()