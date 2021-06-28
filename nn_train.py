#%%
import pickle
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# %%
# df_train = pd.read_csv('0_10.csv')
# df_train = pd.read_parquet("./no_g_handpick.parquet")
df_train = pd.read_parquet("./quanshaixuan.parquet")
# %%
x_label = ["A1", "A2", "A4"]
X = df_train[x_label]

# y_label = ["30", "23y", "23z"]
y_label = ["30", "ave18_23y", "ave18_23z"]
Y = df_train[y_label]

input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

# input_scaler = StandardScaler()
# output_scaler = StandardScaler()import itertools

x = input_scaler.fit_transform(X)
y = output_scaler.fit_transform(Y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42
)

with open("models/minMax_quanshaixuan.pkl", "wb") as f:
    pickle.dump((input_scaler, output_scaler), f)

# %%
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=5, activation="relu", input_dim=x.shape[1]))
model.add(tf.keras.layers.Dense(units=50, activation="relu"))
model.add(tf.keras.layers.Dense(units=50, activation="relu"))
model.add(tf.keras.layers.Dense(units=y.shape[1], activation="linear"))
model.summary()

#%%
def scheduler(epoch, lr):
    if epoch < 400:
        return lr
    else:
        # return lr * tf.math.exp(-0.1)
        return lr * 0.1


callback_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
callback_early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=100, restore_best_weights=True
)

model.compile(optimizer="adam", loss="mse")
# model.compile(optimizer='sgd', loss='mse')

model.fit(
    x_train,
    y_train,
    epochs=600,
    validation_split=0.1,
    callbacks=[callback_early, callback_rate],
    batch_size=1024,
)

# model.compile(optimizer='sgd', loss='mse')
history = model.fit(
    x_train,
    y_train,
    epochs=600,
    validation_split=0.1,
    callbacks=[callback_early, callback_rate],
    batch_size=256,
)
model.save("models/act_abbc_nn_010.h5")
tf.saved_model.save(model, "saved_model/my_model")
# %%
y_predict = model.predict(x_test, batch_size=1024)
df_predict = pd.DataFrame(output_scaler.inverse_transform(y_predict), columns=y_label)
df_test = pd.DataFrame(output_scaler.inverse_transform(y_test), columns=y_label)
plt.figure(figsize=(12, 4))
for i, out in enumerate(y_label):
    plt.subplot(1, 3, i + 1)
    plt.scatter(df_predict[out], df_test[out])
    plt.title(f"{out}: r2={r2_score(df_predict[out], df_test[out])}")
# %%


# %%
# %%
with open("models/minMax2.pkl", "rb") as f:
    in_scaler, out_scaler = pickle.load(f)


#%%
model.save("saved_model/my_model")
# %%

# %%
df_train["A1"].max()
# %%
